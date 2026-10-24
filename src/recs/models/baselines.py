"""Classic reference recommenders, used as benchmark baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from recs.base import Recommender
from recs.similarity import profile_similarity


class MostPopularBaseline(Recommender):
    """Recommend the products bought by the most distinct customers."""

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        self._popularity = self._interactions.getnnz(axis=0).astype(np.float32)

    def _score(self, positions: np.ndarray) -> np.ndarray:
        return np.tile(self._popularity, (len(positions), 1))


class UserCFBaseline(Recommender):
    """User-based collaborative filtering.

    Scores are the sum of the interaction rows of the customer's
    *top_neighbors* most cosine-similar customers, weighted by similarity.
    """

    def __init__(self, top_neighbors: int = 50) -> None:
        self.top_neighbors = top_neighbors

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        I = self._interactions  # noqa: E741
        self._R = profile_similarity(I, metric="cosine", top_k=self.top_neighbors) @ I


class ItemCFBaseline(Recommender):
    """Item-based collaborative filtering (full item-item cosine similarity)."""

    def __init__(self) -> None:
        self._item_sim: sparse.csr_matrix | None = None

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        I = self._interactions  # noqa: E741
        self._item_sim = profile_similarity(I.T.tocsr(), metric="cosine", top_k=I.shape[1])

    def _score(self, positions: np.ndarray) -> np.ndarray:
        return (self._interactions[positions] @ self._item_sim).toarray()


class SVDBaseline(Recommender):
    """Matrix factorisation via truncated SVD on the customer-product matrix.

    Only customers and products with at least one interaction are factorised;
    the rest are never scored.
    """

    positive_scores_only = False

    def __init__(self, n_components: int = 50, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        I = self._interactions  # noqa: E741
        self._active_customers = I.getnnz(axis=1) > 0
        self._active_products = I.getnnz(axis=0) > 0
        active = I[self._active_customers][:, self._active_products]

        n_comp = max(1, min(self.n_components, active.shape[0] - 1, active.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self._user_factors = np.zeros((I.shape[0], n_comp))
        self._user_factors[self._active_customers] = svd.fit_transform(active)
        self._item_factors = np.zeros((I.shape[1], n_comp))
        self._item_factors[self._active_products] = svd.components_.T

    def _score(self, positions: np.ndarray) -> np.ndarray:
        scores = self._user_factors[positions] @ self._item_factors.T
        scores[:, ~self._active_products] = -np.inf
        scores[~self._active_customers[positions]] = -np.inf
        return scores
