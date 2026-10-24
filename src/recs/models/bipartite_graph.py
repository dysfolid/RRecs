"""Bipartite-graph hop recommender.

Scores products by counting / weighting multi-hop paths in the
customer-product bipartite graph:

    R = sum(w_k * C^((k-1)/2) @ I   for each odd k in hop_weights)

where C = I @ I^T (customer co-purchase matrix, top-k sparsified).

Supports strict group filtering and soft per-feature penalties on the
intermediate customer nodes, and popularity-based cold start.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from recs.base import Recommender
from recs.matrix import sparse_topk


class BipartiteGraphRec(Recommender):
    """Graph-based recommender using multi-hop bipartite paths.

    Parameters
    ----------
    hop_weights : dict[int, float]
        Mapping of odd hop counts to their blending weights,
        e.g. ``{3: 1.0, 5: 0.3}``.
    aggregation : ``"multiply"`` | ``"count"``
        ``"multiply"`` uses raw interaction weights along each path
        (path score = product of edge weights; implemented via
        :math:`C^{(k-1)/2} I`).
        ``"count"`` binarises the interaction matrix so every path
        contributes exactly 1 (same matrix formulation).
    top_k : int
        Neighbours retained per row in C (and in each power of C).
    customer_group_cols : list[str] | None
        Columns defining strict customer groups.  Paths cannot
        traverse customers from different groups.
    customer_penalties : dict[str, float] | float | None
        Soft penalties for crossing customer feature boundaries.
        A dict maps feature columns to penalty factors (0-1).
        A single float applies the same penalty to every metadata
        column.  ``None`` disables penalties.
    cold_start_n : int | None
        If set, customers with **no** interactions get recommendations by
        pretending their basket is the top-*cold_start_n* products by
        **sum of interaction weights** (global).  If *customer_group_cols*
        is set, popularity is computed **within each group** first; if that
        group has no data, **global** popularity is used.  Synthetic basket
        entries are treated like real ones for *exclude_seen*.  Only hop
        keys **1** and **3** from *hop_weights* contribute to the cold-start
        score (5/7-hop terms are skipped for that user).
    hop_normalization : ``"none"`` | ``"row"``
        ``"row"`` scales each hop's scores to sum to 1 per customer before
        weighting, so *hop_weights* set each hop's share instead of being
        swamped by the much larger path counts of longer hops.
    """

    def __init__(
        self,
        hop_weights: dict[int, float] | None = None,
        aggregation: Literal["multiply", "count"] = "multiply",
        top_k: int = 100,
        customer_group_cols: list[str] | None = None,
        customer_penalties: dict[str, float] | float | None = None,
        cold_start_n: int | None = None,
        hop_normalization: Literal["none", "row"] = "none",
    ) -> None:
        self.hop_weights = hop_weights or {3: 1.0, 5: 0.3}
        self.aggregation = aggregation
        self.top_k = top_k
        self.customer_group_cols = customer_group_cols
        self.customer_penalties = customer_penalties
        self.cold_start_n = cold_start_n
        self.hop_normalization = hop_normalization

        self._I_graph: sparse.csr_matrix | None = None
        self._cold_baskets: sparse.csr_matrix | None = None

        for k in self.hop_weights:
            if k < 1 or k % 2 == 0:
                raise ValueError(
                    f"hop_weights keys must be positive odd integers, got {k}"
                )

        if aggregation not in ("multiply", "count"):
            raise ValueError(
                f"aggregation must be 'multiply' or 'count', got {aggregation!r}"
            )

        if hop_normalization not in ("none", "row"):
            raise ValueError(
                f"hop_normalization must be 'none' or 'row', got {hop_normalization!r}"
            )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        I = self._interactions  # noqa: E741
        if self.aggregation == "count":
            I = (I > 0).astype(np.float32)  # noqa: E741
        self._I_graph = I

        # --- Customer co-occurrence, with group / penalty constraints ---
        C = sparse_topk(I @ I.T, self.top_k, zero_diag=True)
        C = _constrain(
            C, customers, "customer_id", self._customer_idx,
            self.customer_group_cols, self.customer_penalties,
        )

        # --- Multi-hop score accumulation ---
        # k-hop = C^((k-1)/2) @ I for odd k; powers of C built incrementally.
        max_c_power = (max(self.hop_weights) - 1) // 2
        c_powers = {1: C}
        for p in range(2, max_c_power + 1):
            c_powers[p] = sparse_topk(c_powers[p - 1] @ C, self.top_k)

        R = sparse.csr_matrix(I.shape, dtype=np.float32)
        for k, w in self.hop_weights.items():
            if w:
                c_exp = (k - 1) // 2
                R = R + w * self._normalize_hop(I if c_exp == 0 else c_powers[c_exp] @ I)
        self._R = R

        self._cold_baskets = (
            self._build_cold_baskets(customers, interactions)
            if (self.cold_start_n or 0) > 0 else None
        )

    # ------------------------------------------------------------------
    # recommend hooks
    # ------------------------------------------------------------------

    def _score(self, positions: np.ndarray) -> np.ndarray:
        scores = super()._score(positions)
        if self._cold_baskets is not None:
            baskets = self._cold_baskets[positions]
            cold = np.flatnonzero(np.diff(baskets.indptr))
            if len(cold):
                scores[cold] = self._cold_scores(baskets[cold])
        return scores

    def _seen(self, positions: np.ndarray) -> sparse.csr_matrix:
        seen = super()._seen(positions)
        if self._cold_baskets is None:
            return seen
        baskets = self._cold_baskets[positions]
        baskets.data = np.ones_like(baskets.data)  # whole basket counts as seen
        return seen + baskets

    # ------------------------------------------------------------------
    # cold start
    # ------------------------------------------------------------------

    def _build_cold_baskets(
        self,
        customers: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> sparse.csr_matrix | None:
        """Synthetic popular basket for every customer without interactions.

        Returns an (n x m) matrix whose rows are empty for warm customers,
        or ``None`` if no customer gets a basket.
        """
        n, m = self._interactions.shape
        cold = np.flatnonzero(np.diff(self._interactions.indptr) == 0)
        pop_global, pop_by_group, group_keys = _cold_popularity_tables(
            interactions, customers, self._customer_idx, self._product_idx,
            self.customer_group_cols,
        )
        n_take = min(self.cold_start_n, m)

        def popularity(pos: int) -> tuple[str | None, np.ndarray]:
            if group_keys is not None and pop_by_group:
                key = str(group_keys.iloc[pos])
                gpop = pop_by_group.get(key)
                if gpop is not None and np.any(gpop > 0):
                    return key, gpop
            return None, pop_global

        baskets: dict[str | None, tuple[np.ndarray, np.ndarray] | None] = {}
        rows, cols, vals = [], [], []
        for pos in cold:
            key, pop = popularity(pos)
            if key not in baskets:
                if np.max(pop) <= 0:
                    baskets[key] = None
                else:
                    top = np.argpartition(pop, -n_take)[-n_take:]
                    if self.aggregation == "count":
                        weights = np.ones(len(top), dtype=np.float32)
                    else:
                        weights = pop[top].astype(np.float32)
                    baskets[key] = (top, weights)
            if baskets[key] is None:
                continue
            top, weights = baskets[key]
            rows.append(np.full(len(top), pos))
            cols.append(top)
            vals.append(weights)

        if not rows:
            return None
        return sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, m),
        )

    def _cold_scores(self, baskets: sparse.csr_matrix) -> np.ndarray:
        """Hop-1 and hop-3 scores for synthetic baskets (5+ hops skipped)."""
        I_g = self._I_graph
        scores = np.zeros(baskets.shape, dtype=np.float64)
        hw = self.hop_weights
        if hw.get(1):
            scores += float(hw[1]) * self._normalize_hop(baskets).toarray()
        if hw.get(3):
            hop3 = self._normalize_hop((baskets @ I_g.T) @ I_g)
            scores += float(hw[3]) * hop3.toarray()
        return scores.astype(np.float32)

    def _normalize_hop(self, R_k: sparse.csr_matrix) -> sparse.csr_matrix:
        """Apply *hop_normalization* to one hop's (customer x product) scores."""
        if self.hop_normalization == "none":
            return R_k
        totals = np.asarray(R_k.sum(axis=1)).ravel()
        inv = np.divide(1.0, totals, out=np.zeros_like(totals, dtype=np.float64),
                        where=totals != 0)
        return (sparse.diags(inv) @ R_k).astype(np.float32).tocsr()


# ======================================================================
# Helpers
# ======================================================================


def _constrain(
    C: sparse.csr_matrix,
    entity_df: pd.DataFrame,
    id_col: str,
    idx: pd.Index,
    group_cols: list[str] | None,
    penalties: dict[str, float] | float | None,
) -> sparse.csr_matrix:
    """Scale the entries of the entity-entity matrix *C* by metadata agreement.

    ``C[i, j]`` is zeroed when *i* and *j* differ on the composite
    *group_cols* key (strict groups), and multiplied by ``penalties[col]``
    for every penalty column on which they differ (soft penalties).  Only
    existing non-zero entries are visited.
    """
    if not group_cols and penalties is None:
        return C

    df = entity_df.set_index(id_col).reindex(idx)
    coo = C.tocoo()
    factors = np.ones(coo.nnz, dtype=np.float32)

    if group_cols:
        keys = df[group_cols].astype(str).agg("|".join, axis=1).values
        factors[keys[coo.row] != keys[coo.col]] = 0.0

    if penalties is not None:
        if isinstance(penalties, (int, float)):
            penalties = {c: float(penalties) for c in df.columns}
        for col, factor in penalties.items():
            if col not in df.columns:
                continue
            values = df[col].values
            factors[values[coo.row] != values[coo.col]] *= factor

    scale = sparse.csr_matrix((factors, (coo.row, coo.col)), shape=C.shape)
    out = C.multiply(scale).tocsr()
    out.eliminate_zeros()
    return out


def _cold_popularity_tables(
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    customer_idx: pd.Index,
    product_idx: pd.Index,
    customer_group_cols: list[str] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, pd.Series | None]:
    """Per-product sum of weights (global and optional per customer group)."""

    def product_totals(df: pd.DataFrame) -> np.ndarray:
        totals = np.zeros(len(product_idx), dtype=np.float64)
        col = product_idx.get_indexer(df["product_id"])
        w = df["weight"].values.astype(np.float64)
        ok = col >= 0
        np.add.at(totals, col[ok], w[ok])
        return totals

    pop_global = product_totals(interactions)
    if not customer_group_cols:
        return pop_global, None, None

    df_c = customers.set_index("customer_id").reindex(customer_idx)
    group_keys = df_c[customer_group_cols].astype(str).agg("|".join, axis=1)
    merged = interactions.merge(
        customers[["customer_id", *customer_group_cols]],
        on="customer_id",
        how="left",
    )
    keys = merged[customer_group_cols].astype(str).agg("|".join, axis=1)
    by_group = {
        str(gk): product_totals(merged[keys == gk]) for gk in keys.dropna().unique()
    }
    return pop_global, by_group, group_keys
