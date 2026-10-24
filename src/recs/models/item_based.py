"""Item-item models: a customer's score for item *i* sums ``W[j, i]`` over
the items *j* in their basket.

* :class:`ItemKNNRec` -- top-k shared-buyer similarity (item-based CF)
* :class:`EASERec` -- closed-form linear item model (Steck, 2019)
* :class:`RP3BetaRec` -- popularity-penalised 3-step random walk
  (Paudel et al., 2016)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recs.base import Recommender
from recs.matrix import sparse_topk

_BASKETS = ("binary", "weight", "log")


class _ItemItemRecommender(Recommender):
    """Scores = basket rows @ ``self._W`` (an m x m matrix set in ``_fit``)."""

    basket: str = "binary"
    _W: sparse.csr_matrix | np.ndarray | None = None

    def _binary(self) -> sparse.csr_matrix:
        return (self._interactions > 0).astype(np.float64)

    def _basket_rows(self, positions: np.ndarray) -> sparse.csr_matrix:
        rows = self._interactions[positions].astype(np.float64)
        if self.basket == "binary":
            rows.data = (rows.data > 0).astype(np.float64)
        elif self.basket == "log":
            rows.data = np.log1p(rows.data)
        return rows

    def _score(self, positions: np.ndarray) -> np.ndarray:
        scores = self._basket_rows(positions) @ self._W
        return scores.toarray() if sparse.issparse(scores) else np.asarray(scores)


def _keep_top_k(W: sparse.csr_matrix, top_k: int, symmetrize: bool) -> sparse.csr_matrix:
    """Restrict *W* to each row's *top_k* entries (or either direction's)."""
    mask = sparse_topk(W, top_k)
    if symmetrize:
        mask = (mask + mask.T).tocsr()
    mask.data[:] = 1.0
    return W.multiply(mask).tocsr()


class ItemKNNRec(_ItemItemRecommender):
    """Item-based k-nearest-neighbour CF on shared buyers.

    ``W[j, i] = shared(j, i) / (buyers(j)^alpha * buyers(i)^(1 - alpha) + shrink)``,
    then each item keeps its *top_k* most similar items.

    Parameters
    ----------
    top_k : int
        Neighbours kept per item.
    alpha : float
        Asymmetric cosine exponent (Aiolli, 2013).  ``0.5`` is cosine;
        larger values divide less by the candidate's popularity, favouring
        popular candidates (``1.0`` gives P(i bought | j bought)); smaller
        values favour niche candidates.
    shrink : float
        Added to the denominator, damping similarities built on few buyers.
    symmetrize : bool
        Keep an edge if either item lists the other among its *top_k*
        (values stay ``W[j, i]``).  If ``False``, only the source item's
        own top-k are used.
    basket : ``"binary"`` | ``"weight"`` | ``"log"``
        How the customer's past interactions weight their items: 1 each,
        the raw interaction weight, or ``log1p`` of it.
    """

    def __init__(
        self,
        top_k: int = 25,
        alpha: float = 0.5,
        shrink: float = 0.0,
        symmetrize: bool = True,
        basket: str = "binary",
    ) -> None:
        if basket not in _BASKETS:
            raise ValueError(f"basket must be one of {_BASKETS}, got {basket!r}")
        self.top_k = top_k
        self.alpha = alpha
        self.shrink = shrink
        self.symmetrize = symmetrize
        self.basket = basket

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        B = self._binary()
        shared = (B.T @ B).tocsr()
        shared.setdiag(0)
        shared.eliminate_zeros()
        buyers = np.asarray(B.sum(axis=0)).ravel()

        coo = shared.tocoo()
        denom = (
            buyers[coo.row] ** self.alpha * buyers[coo.col] ** (1.0 - self.alpha)
            + self.shrink
        )
        W = sparse.csr_matrix((coo.data / denom, (coo.row, coo.col)), shape=shared.shape)
        self._W = _keep_top_k(W, self.top_k, self.symmetrize)


class EASERec(_ItemItemRecommender):
    """EASE: closed-form linear item-item model (Steck, 2019).

    Solves ``min ||X - X W||^2 + l2 * ||W||^2`` subject to ``diag(W) = 0``
    for the binary interaction matrix *X*:
    ``W = I - P / diag(P)`` (column-wise) with ``P = (X^T X + l2 * I)^-1``.

    *W* is a dense m x m matrix, so this suits catalogues up to tens of
    thousands of products.

    Parameters
    ----------
    l2 : float
        L2 regularisation strength.
    """

    def __init__(self, l2: float = 500.0) -> None:
        self.l2 = l2

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        X = self._binary()
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        W = -P / np.diag(P)[None, :]
        np.fill_diagonal(W, 0.0)
        self._W = W


class RP3BetaRec(_ItemItemRecommender):
    """RP3beta: popularity-penalised 3-step random walk (Paudel et al., 2016).

    ``W[i, j] = sum_c P(c | i)^alpha * P(j | c)^alpha / buyers(j)^beta``:
    walk item -> customer -> item on the binary purchase graph, sharpen the
    transition probabilities with *alpha*, divide out the destination's
    popularity with *beta*, and keep each item's *top_k* neighbours.

    Parameters
    ----------
    alpha : float
        Exponent on the transition probabilities.
    beta : float
        Popularity penalty; ``0`` is the plain random walk.
    top_k : int
        Neighbours kept per item.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, top_k: int = 100) -> None:
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k

    def _fit(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        X = self._binary().tocsr()
        basket_size = np.asarray(X.sum(axis=1)).ravel()
        buyers = np.asarray(X.sum(axis=0)).ravel()

        p_item = _inverse(basket_size)  # P(j | c) = 1 / basket size
        p_cust = _inverse(buyers)       # P(c | i) = 1 / buyers of i
        P_ci = sparse.diags(p_item ** self.alpha) @ X          # c -> j
        P_ic = sparse.diags(p_cust ** self.alpha) @ X.T.tocsr()  # i -> c

        W = (P_ic @ P_ci).tocsr()
        W = W @ sparse.diags(_inverse(buyers) ** self.beta)
        W = W.tocsr()
        W.setdiag(0)
        W.eliminate_zeros()
        self._W = _keep_top_k(W, self.top_k, symmetrize=False)


def _inverse(x: np.ndarray) -> np.ndarray:
    """Elementwise 1 / x, with 0 where x is 0."""
    return np.divide(1.0, x, out=np.zeros_like(x, dtype=np.float64), where=x > 0)
