"""Weighted-similarity recommendation model.

Scores are computed as:

    R = (w_cm * CSM + w_ci * ICSM) @ I @ (w_pm * PSM + w_pi * IPSM)

where all matrices are sparse (top-k neighbours only).  Any weight set to
zero skips the corresponding similarity computation entirely, and a side
whose weights are all zero is left out of the product.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from scipy import sparse

from recs.base import Recommender
from recs.similarity import Metric, metadata_similarity, profile_similarity


class WeightedSimilarity(Recommender):
    """Blended metadata + interaction similarity recommender.

    Parameters
    ----------
    w_product_metadata : float
        Weight for the product metadata similarity matrix (PSM).
    w_product_interactions : float
        Weight for the interaction-based product similarity matrix (IPSM).
    w_customer_metadata : float
        Weight for the customer metadata similarity matrix (CSM).
    w_customer_interactions : float
        Weight for the interaction-based customer similarity matrix (ICSM).
    metric : ``"cosine"`` | ``"overlap"``
        Similarity metric (see :mod:`recs.similarity`).
    top_k : int
        Default neighbours retained per row in similarity matrices.
    top_k_psm, top_k_ipsm, top_k_csm, top_k_icsm : int | None
        Per-matrix overrides.  When ``None``, falls back to *top_k*.
    """

    def __init__(
        self,
        w_product_metadata: float = 0.25,
        w_product_interactions: float = 0.25,
        w_customer_metadata: float = 0.25,
        w_customer_interactions: float = 0.25,
        metric: Metric = "cosine",
        top_k: int = 100,
        top_k_psm: int | None = None,
        top_k_ipsm: int | None = None,
        top_k_csm: int | None = None,
        top_k_icsm: int | None = None,
    ) -> None:
        self.w_product_metadata = w_product_metadata
        self.w_product_interactions = w_product_interactions
        self.w_customer_metadata = w_customer_metadata
        self.w_customer_interactions = w_customer_interactions
        self.metric = metric
        self.top_k = top_k
        self.top_k_psm = top_k_psm
        self.top_k_ipsm = top_k_ipsm
        self.top_k_csm = top_k_csm
        self.top_k_icsm = top_k_icsm

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        I = self._interactions  # noqa: E741
        cust_meta_cols = [c for c in customers.columns if c != "customer_id"]
        prod_meta_cols = [c for c in products.columns if c != "product_id"]

        customer_blend = _blend(
            (self.w_customer_metadata, lambda: metadata_similarity(
                customers, cust_meta_cols, self.metric, self._k(self.top_k_csm))),
            (self.w_customer_interactions, lambda: profile_similarity(
                I, self.metric, self._k(self.top_k_icsm))),
        )
        product_blend = _blend(
            (self.w_product_metadata, lambda: metadata_similarity(
                products, prod_meta_cols, self.metric, self._k(self.top_k_psm))),
            (self.w_product_interactions, lambda: profile_similarity(
                I.T.tocsr(), self.metric, self._k(self.top_k_ipsm))),
        )

        R = I if customer_blend is None else customer_blend @ I
        self._R = R if product_blend is None else R @ product_blend

    def _k(self, override: int | None) -> int:
        return self.top_k if override is None else override


def _blend(
    *terms: tuple[float, Callable[[], sparse.csr_matrix]],
) -> sparse.csr_matrix | None:
    """Weighted sum of lazily built matrices; zero-weight terms are never built.

    Returns ``None`` when every weight is zero.
    """
    parts = [w * build() for w, build in terms if w]
    if not parts:
        return None
    return sum(parts[1:], parts[0])
