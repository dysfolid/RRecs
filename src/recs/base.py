from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from recs.matrix import interaction_matrix, top_n

RecommenderT = TypeVar("RecommenderT", bound="Recommender")

RESULT_COLUMNS = ["customer_id", "product_id", "score", "rank"]


class Recommender(ABC):
    """Abstract base class for all recommendation models.

    The base class owns the plumbing every model shares: ID indexing, the
    interaction matrix, seen-item exclusion and top-*n* ranking.  A model
    implements:

    * :meth:`_fit` -- build model state.  When it is called,
      ``self._customer_idx``, ``self._product_idx`` and ``self._interactions``
      (the n x m interaction weight matrix *I*) are already set.
    * :meth:`_score` -- dense scores for a batch of customers.  The default
      reads rows of ``self._R``, so a model that precomputes a sparse score
      matrix in :meth:`_fit` needs nothing else.

    Class attributes
    ----------------
    positive_scores_only : bool
        Only recommend products with a score > 0.  Models whose scores can
        be negative (e.g. SVD) set this to ``False``.
    recommend_batch_size : int
        Customers scored per :meth:`_score` call (controls peak memory).
    show_progress : bool
        Show a progress bar during :meth:`recommend`.
    """

    positive_scores_only: bool = True
    recommend_batch_size: int = 1024
    show_progress: bool = False

    _customer_idx: pd.Index | None = None
    _product_idx: pd.Index | None = None
    _interactions: sparse.csr_matrix | None = None
    _R: sparse.csr_matrix | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self: RecommenderT,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> RecommenderT:
        """Build internal state from input tables.

        Parameters
        ----------
        customers : DataFrame
            Must contain a ``customer_id`` column plus metadata columns.
        products : DataFrame
            Must contain a ``product_id`` column plus metadata columns.
        interactions : DataFrame
            Must contain ``customer_id``, ``product_id``, and ``weight`` columns.
        """
        self._customer_idx = pd.Index(customers["customer_id"].values)
        self._product_idx = pd.Index(products["product_id"].values)
        self._interactions = interaction_matrix(
            interactions, self._customer_idx, self._product_idx
        )
        self._fit(customers, products, interactions)
        return self

    @abstractmethod
    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        """Model-specific fitting (see the class docstring)."""

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        customer_ids: list | pd.Index | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Return top-*n* product recommendations per customer.

        Parameters
        ----------
        customer_ids
            Subset of customers to score.  ``None`` means all customers
            seen during :meth:`fit`.
        n
            Number of products to return per customer.
        exclude_seen
            If ``True``, products the customer already interacted with are
            removed from the results.  Set to ``False`` to include them.

        Returns
        -------
        DataFrame
            Columns ``customer_id``, ``product_id``, ``score``, ``rank``.
        """
        if self._interactions is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if customer_ids is None:
            customer_ids = self._customer_idx
        customer_ids = pd.Index(customer_ids)

        positions = self._customer_idx.get_indexer(customer_ids)
        if (positions == -1).any():
            unknown = customer_ids[positions == -1].tolist()
            raise KeyError(f"Unknown customer_ids: {unknown}")

        frames = []
        with tqdm(
            total=len(positions),
            desc=f"{type(self).__name__} recommend",
            unit="user",
            disable=not self.show_progress,
        ) as bar:
            for start in range(0, len(positions), self.recommend_batch_size):
                end = start + self.recommend_batch_size
                frames.append(
                    self._recommend_batch(
                        customer_ids[start:end], positions[start:end], n, exclude_seen
                    )
                )
                bar.update(len(positions[start:end]))

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _recommend_batch(
        self,
        customer_ids: pd.Index,
        positions: np.ndarray,
        n: int,
        exclude_seen: bool,
    ) -> pd.DataFrame:
        scores = self._score(positions)
        if exclude_seen:
            # The masked value must fail the `keep` test below.
            rows, cols = self._seen(positions).nonzero()
            scores[rows, cols] = 0.0 if self.positive_scores_only else -np.inf

        top = top_n(scores, n)
        top_scores = np.take_along_axis(scores, top, axis=1)
        keep = top_scores > 0 if self.positive_scores_only else top_scores > -np.inf

        rows, ranks = np.nonzero(keep)
        return pd.DataFrame({
            "customer_id": customer_ids[rows],
            "product_id": self._product_idx[top[rows, ranks]],
            "score": top_scores[rows, ranks],
            "rank": ranks + 1,
        })

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def _score(self, positions: np.ndarray) -> np.ndarray:
        """Return a new dense (len(positions), n_products) score array.

        The caller modifies the returned array in place.
        """
        return self._R[positions].toarray()

    def _seen(self, positions: np.ndarray) -> sparse.csr_matrix:
        """Products removed when ``exclude_seen=True`` (non-zero entries)."""
        return self._interactions[positions]
