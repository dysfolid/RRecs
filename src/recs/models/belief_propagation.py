"""Item–item belief propagation for collaborative filtering.

Sum–product loopy BP on a sparse **item co-occurrence graph** (``I^T @ I``).
Each item node carries a binary variable *relevant* vs *not*; unary terms pin
items in the user’s basket; pairwise Ising-like factors propagate affinity
along co-purchase edges.  Scores are the marginal P(relevant).

This follows the spirit of PMRF / BP recommender work (marginals via message
passing) while keeping the implementation compact and dependency-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recs.base import Recommender
from recs.matrix import sparse_topk


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


def _psi_matrix(w_norm: float, beta: float) -> np.ndarray:
    """2×2 pairwise potential; ferromagnetic when *beta* > 0."""
    J = beta * float(w_norm)
    e_p, e_m = np.exp(J), np.exp(-J)
    return np.array([[e_p, e_m], [e_m, e_p]], dtype=np.float64)


def _run_loopy_bp(
    neighbors: list[list[int]],
    edge_w: dict[tuple[int, int], float],
    phi: np.ndarray,
    *,
    beta: float,
    max_iter: int,
    damping: float,
) -> np.ndarray:
    """Return P(x_i = 1) for each local node *i* (shape (L,))."""
    L = len(neighbors)
    # Directed messages (j -> i): only for edges present
    msgs: dict[tuple[int, int], np.ndarray] = {}
    for i in range(L):
        for j in neighbors[i]:
            msgs[(j, i)] = np.array([0.5, 0.5], dtype=np.float64)

    for _ in range(max_iter):
        new_msgs: dict[tuple[int, int], np.ndarray] = {}
        for i in range(L):
            for j in neighbors[i]:
                w_key = (j, i) if j < i else (i, j)
                w_ij = edge_w.get(w_key, 0.0)
                psi = _psi_matrix(w_ij, beta)
                out = np.zeros(2, dtype=np.float64)
                for xi in (0, 1):
                    acc = 0.0
                    for xj in (0, 1):
                        prod = phi[j, xj]
                        for k in neighbors[j]:
                            if k == i:
                                continue
                            prod *= msgs[(k, j)][xj]
                        acc += psi[xi, xj] * prod
                    out[xi] = acc
                s = out.sum()
                if s > 1e-15:
                    out /= s
                old = msgs[(j, i)]
                new_msgs[(j, i)] = damping * old + (1.0 - damping) * out

        msgs = new_msgs

    beliefs = np.zeros((L, 2), dtype=np.float64)
    for i in range(L):
        b = phi[i].copy()
        for j in neighbors[i]:
            b *= msgs[(j, i)]
        s = b.sum()
        if s > 1e-15:
            b /= s
        else:
            b = np.array([0.5, 0.5], dtype=np.float64)
        beliefs[i] = b

    return beliefs[:, 1]


class BeliefPropagationRec(Recommender):
    """Loopy sum–product BP on a sparse item–item co-occurrence graph.

    Parameters
    ----------
    top_k : int
        Rows of ``I^T @ I`` are sparsified to this many neighbours per item.
    beta : float
        Strength of pairwise agreement along an edge (after normalizing weights).
    max_iter : int
        BP iterations.
    damping : float
        Message update damping in ``[0, 1)`` (higher = slower change).
    max_subgraph_nodes : int
        Maximum item nodes in the subgraph built around a user’s basket
        (for latency).
    show_progress : bool
        If ``True``, show a tqdm bar over customers during *recommend*.
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
    ) -> None:
        self.top_k = top_k
        self.beta = beta
        self.max_iter = max_iter
        self.damping = damping
        self.max_subgraph_nodes = max_subgraph_nodes
        self.show_progress = show_progress

        self._S: sparse.csr_matrix | None = None

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        I = self._interactions  # noqa: E741
        S = sparse_topk(I.T @ I, self.top_k, zero_diag=True)
        self._S = _symmetrize_max(S).tocsr()

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
        """Full length-*m* score vector (mostly zeros if disconnected)."""
        scores = np.zeros(m, dtype=np.float64)
        uh, uw = user_row.indices, user_row.data
        if len(uh) == 0:
            return scores

        basket = set(int(i) for i in uh)
        w_map = {int(i): float(w) for i, w in zip(uh, uw)}
        max_w = max(w_map.values()) if w_map else 1.0
        if max_w <= 0:
            max_w = 1.0

        V = _expand_item_subgraph(basket, S, self.max_subgraph_nodes)
        if len(V) <= len(basket) and not any(j not in basket for j in V):
            # No neighbour beyond basket: no pairwise signal for new items
            return scores

        glob = sorted(V)
        loc = {g: i for i, g in enumerate(glob)}
        L = len(glob)

        neighbors: list[list[int]] = [[] for _ in range(L)]
        edge_w: dict[tuple[int, int], float] = {}
        w_max = 1e-9

        for g in glob:
            row = S.getrow(g)
            for h, w in zip(row.indices, row.data):
                if h not in loc:
                    continue
                a, b = loc[g], loc[h]
                if a == b:
                    continue
                i, j = (a, b) if a < b else (b, a)
                if (i, j) not in edge_w:
                    edge_w[(i, j)] = float(w)
                    w_max = max(w_max, float(w))
                if b not in neighbors[a]:
                    neighbors[a].append(b)
                if a not in neighbors[b]:
                    neighbors[b].append(a)

        if w_max <= 0:
            w_max = 1.0
        for key in edge_w:
            edge_w[key] /= w_max

        phi = np.ones((L, 2), dtype=np.float64) * 0.5
        for g in basket:
            if g not in loc:
                continue
            li = loc[g]
            w = w_map[g]
            p1 = 0.5 + 0.5 * min(1.0, w / max_w)
            phi[li, 1] = p1
            phi[li, 0] = 1.0 - p1

        p_relevant = _run_loopy_bp(
            neighbors,
            edge_w,
            phi,
            beta=self.beta,
            max_iter=self.max_iter,
            damping=self.damping,
        )

        for g, pr in zip(glob, p_relevant):
            scores[g] = pr

        return scores
