"""Item–item belief propagation for collaborative filtering.

Sum–product loopy BP on a sparse **item co-occurrence graph**.  Each item node
carries a binary variable *relevant* vs *not*; unary terms encode the user's
basket (evidence) and a prior for every other item; pairwise factors propagate
affinity along co-purchase edges.  Scores are the belief that an item is
relevant.

Messages for all edges of a user's subgraph are updated at once (numpy).  A
binary message is stored as its magnetisation ``p(1) - p(0)``; for an edge
a -> b with cavity log-odds *nu* at *a* (everything *a* hears except from
*b*) the sum-product update has a closed form, so an iteration costs
O(edges).  Messages are always normalised (log-odds), so high-degree nodes
cannot underflow.

The defaults reproduce the original model; see the parameters for the
variants studied on the Online Retail benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recs.base import Recommender
from recs.matrix import sparse_topk

_COUPLINGS = ("spend", "count", "cosine")
_POTENTIALS = ("ising", "positive")
_SCORES = ("prob", "logodds")


def _symmetrize_max(S: sparse.csr_matrix) -> sparse.csr_matrix:
    """S' = max(S, S.T) so the graph is undirected with shared weights."""
    St = S.transpose().tocsr()
    return S.maximum(St)


def _expand_item_subgraph(
    basket: set[int],
    S: sparse.csr_matrix,
    max_nodes: int,
) -> set[int]:
    """Basket ∪ highest-weight neighbours until *max_nodes* (always keep basket)."""
    if len(basket) > max_nodes:
        basket = set(sorted(basket)[:max_nodes])

    scores: dict[int, float] = {}
    for i in basket:
        row = S.getrow(i)
        for j, w in zip(row.indices, row.data):
            if j in basket:
                continue
            scores[j] = scores.get(j, 0.0) + float(w)

    V = set(basket)
    for j, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        if len(V) >= max_nodes:
            break
        V.add(j)
    return V


def _directed_edges(
    S: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(src, dst, weight, rev)`` for every stored entry of symmetric *S*.

    Edge *e* carries the message ``src[e] -> dst[e]``; ``rev[e]`` is the
    edge ``dst[e] -> src[e]``.
    """
    S = S.tocsr().copy()
    S.sort_indices()
    n = S.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(S.indptr))
    dst = S.indices.astype(np.int64)
    keys = src * n + dst
    rev = np.searchsorted(keys, dst * n + src)
    return src, dst, S.data.astype(np.float64), rev


class BeliefPropagationRec(Recommender):
    """Loopy sum–product BP on a sparse item–item co-occurrence graph.

    Parameters
    ----------
    top_k : int
        Each item keeps this many neighbours in the item graph (the graph is
        then symmetrised by taking the larger weight).
    beta : float
        Coupling strength: edge coupling ``J = beta * edge weight``.
    max_iter : int
        Maximum BP iterations.
    damping : float
        Message update damping in ``[0, 1)`` (higher = slower change).
    max_subgraph_nodes : int
        Maximum item nodes in the subgraph built around a user’s basket
        (for latency).
    show_progress : bool
        If ``True``, show a tqdm bar over customers during *recommend*.
    coupling : ``"spend"`` | ``"count"`` | ``"cosine"``
        Edge weights: ``I^T I`` on interaction weights, scaled by the largest
        weight in the user's subgraph (original model); shared-buyer counts,
        scaled the same way; or shared-buyer cosine
        ``shared / sqrt(buyers_i * buyers_j)``, used as is.
    evidence : ``"spend"`` | float
        P(relevant) of basket items: ``0.5 + 0.5 * weight / max weight``
        (original model), or the same fixed value for every basket item
        (``1.0`` clamps them as relevant).
    prior : float
        P(relevant) of items outside the basket (``0.5`` = no prior).
    potential : ``"ising"`` | ``"positive"``
        ``"ising"``: ``psi = [[e^J, e^-J], [e^-J, e^J]]`` (original model), so
        a likely-irrelevant neighbour pushes down as hard as a relevant one
        pushes up.  ``"positive"``: ``psi = [[1, 1], [1, e^J]]``, so only
        relevant neighbours raise relevance.
    score : ``"prob"`` | ``"logodds"``
        Rank by P(relevant) (original model) or by its log-odds, which gives
        the same order but does not saturate at 1.0 in floating point.
    tol : float | None
        Stop early once no message changes by more than *tol*.
    """

    # BP runs per customer, so small batches keep the progress bar moving.
    recommend_batch_size = 32

    def __init__(
        self,
        top_k: int = 100,
        beta: float = 1.0,
        max_iter: int = 15,
        damping: float = 0.5,
        max_subgraph_nodes: int = 4000,
        show_progress: bool = False,
        coupling: str = "spend",
        evidence: str | float = "spend",
        prior: float = 0.5,
        potential: str = "ising",
        score: str = "prob",
        tol: float | None = None,
    ) -> None:
        for name, value, allowed in [
            ("coupling", coupling, _COUPLINGS),
            ("potential", potential, _POTENTIALS),
            ("score", score, _SCORES),
        ]:
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if evidence != "spend" and not 0.0 < float(evidence) <= 1.0:
            raise ValueError(f"evidence must be 'spend' or in (0, 1], got {evidence!r}")
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior!r}")

        self.top_k = top_k
        self.beta = beta
        self.max_iter = max_iter
        self.damping = damping
        self.max_subgraph_nodes = max_subgraph_nodes
        self.show_progress = show_progress
        self.coupling = coupling
        self.evidence = evidence
        self.prior = prior
        self.potential = potential
        self.score = score
        self.tol = tol
        # Log-odds can be negative; items outside the subgraph score -inf.
        self.positive_scores_only = score == "prob"

        self._S: sparse.csr_matrix | None = None
        self._edges: tuple[np.ndarray, ...] | None = None

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
        if self.coupling == "spend":
            W = I.T @ I
        else:
            B = (I > 0).astype(np.float64)
            W = (B.T @ B).tocsr()
            if self.coupling == "cosine":
                buyers = W.diagonal()
                inv = np.divide(
                    1.0, np.sqrt(buyers), out=np.zeros_like(buyers), where=buyers > 0
                )
                W = sparse.diags(inv) @ W @ sparse.diags(inv)
        S = sparse_topk(W, self.top_k, zero_diag=True)
        self._S = _symmetrize_max(S).tocsr()
        self._edges = _directed_edges(self._S)

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _score(self, positions: np.ndarray) -> np.ndarray:
        m = self._S.shape[0]
        return np.vstack([
            self._scores_for_user(self._interactions.getrow(pos), self._S, m)
            for pos in positions
        ])

    def _scores_for_user(
        self,
        user_row: sparse.csr_matrix,
        S: sparse.csr_matrix,
        m: int,
    ) -> np.ndarray:
        """Full length-*m* score vector; items outside the subgraph score
        0 (``score="prob"``) or -inf (``score="logodds"``)."""
        scores = np.full(m, 0.0 if self.score == "prob" else -np.inf)
        uh, uw = user_row.indices, user_row.data
        if len(uh) == 0:
            return scores

        basket = set(int(i) for i in uh)
        w_map = {int(i): float(w) for i, w in zip(uh, uw)}
        max_w = max(w_map.values())
        if max_w <= 0:
            max_w = 1.0

        V = _expand_item_subgraph(basket, S, self.max_subgraph_nodes)
        if V <= basket:
            # No neighbour beyond basket: no pairwise signal for new items
            return scores

        nodes = np.fromiter(V, dtype=np.int64)
        in_v = np.zeros(m, dtype=bool)
        in_v[nodes] = True

        # --- edges with both ends in the subgraph ---
        src_all, dst_all, w_all, rev_all = self._edges
        act = np.flatnonzero(in_v[src_all] & in_v[dst_all])
        local = np.full(len(src_all), -1, dtype=np.int64)
        local[act] = np.arange(len(act))
        src, dst, rev = src_all[act], dst_all[act], local[rev_all[act]]

        w = w_all[act]
        if self.coupling != "cosine" and len(w):
            w = w / max(1e-9, float(w.max()))
        J = self.beta * w
        tanh_J = np.tanh(J)

        # --- unary log-odds h (inf for a basket item clamped at 1.0) ---
        p1 = np.full(m, float(self.prior))
        for g in basket:
            if in_v[g]:
                p1[g] = (
                    0.5 + 0.5 * min(1.0, w_map[g] / max_w)
                    if self.evidence == "spend" else float(self.evidence)
                )
        with np.errstate(divide="ignore"):
            h = np.log(p1) - np.log1p(-p1)

        # --- synchronous damped sum-product ---
        mag = np.zeros(len(act))  # every message starts at [0.5, 0.5]
        for _ in range(self.max_iter):
            log_odds = 2.0 * np.arctanh(mag)
            total = h + np.bincount(dst, weights=log_odds, minlength=m)
            cavity = total[src] - log_odds[rev]
            if self.potential == "ising":
                out = tanh_J * np.tanh(0.5 * cavity)
            else:  # message log-odds log((1 + e^(J + nu)) / (1 + e^nu)), in [0, J]
                nu = np.clip(cavity, -700.0, 700.0)
                out = np.tanh(0.5 * (np.logaddexp(0.0, J + nu) - np.logaddexp(0.0, nu)))
            new = self.damping * mag + (1.0 - self.damping) * out
            converged = self.tol is not None and (
                len(new) == 0 or np.abs(new - mag).max() < self.tol
            )
            mag = new
            if converged:
                break

        total = h + np.bincount(dst, weights=2.0 * np.arctanh(mag), minlength=m)
        if self.score == "logodds":
            scores[nodes] = total[nodes]
        else:
            scores[nodes] = 0.5 * (1.0 + np.tanh(0.5 * total[nodes]))
        return scores
