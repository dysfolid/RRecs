"""Matrix helpers shared by all models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def interaction_matrix(
    interactions: pd.DataFrame,
    customer_idx: pd.Index,
    product_idx: pd.Index,
) -> sparse.csr_matrix:
    """Build the (n_customers x n_products) interaction weight matrix *I*.

    Rows follow *customer_idx*, columns follow *product_idx*; duplicate
    (customer, product) rows are summed.
    """
    rows = customer_idx.get_indexer(interactions["customer_id"])
    cols = product_idx.get_indexer(interactions["product_id"])
    for positions, col in ((rows, "customer_id"), (cols, "product_id")):
        if (positions == -1).any():
            unknown = interactions[col][positions == -1].unique()[:5].tolist()
            raise ValueError(f"interactions reference unknown {col} values: {unknown}")

    weights = interactions["weight"].values.astype(np.float32)
    return sparse.csr_matrix(
        (weights, (rows, cols)), shape=(len(customer_idx), len(product_idx))
    )


def sparse_topk(
    mat: sparse.spmatrix,
    top_k: int,
    zero_diag: bool = False,
) -> sparse.csr_matrix:
    """Keep only the *top_k* largest stored values in each row.

    With *zero_diag*, the diagonal of a square matrix is dropped first.
    """
    mat = mat.tocsr()
    if zero_diag and mat.shape[0] == mat.shape[1]:
        mat = mat.copy()
        mat.setdiag(0)
        mat.eliminate_zeros()

    if top_k >= mat.shape[1]:
        return mat.astype(np.float32)

    keep = np.ones(mat.nnz, dtype=bool)
    for i in np.flatnonzero(np.diff(mat.indptr) > top_k):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        dropped = np.argpartition(mat.data[start:end], -top_k)[:-top_k]
        keep[start + dropped] = False

    rows = np.repeat(np.arange(mat.shape[0]), np.diff(mat.indptr))
    return sparse.csr_matrix(
        (mat.data[keep], (rows[keep], mat.indices[keep])),
        shape=mat.shape,
        dtype=np.float32,
    )


def top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Column indices of the *n* largest values in each row, best first."""
    if n <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.intp)
    if n < scores.shape[1]:
        top = np.argpartition(scores, -n, axis=1)[:, -n:]
        order = np.argsort(np.take_along_axis(scores, top, axis=1), axis=1)[:, ::-1]
        return np.take_along_axis(top, order, axis=1)
    return np.argsort(scores, axis=1)[:, ::-1][:, :n]
