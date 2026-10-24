"""Similarity computation utilities.

All public functions return a *sparse* CSR similarity matrix that keeps the
top-k neighbours per row, with self-similarity removed.  Row/column *i*
corresponds to row *i* of the input.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

Metric = Literal["cosine", "overlap"]

_DEFAULT_BATCH_SIZE = 2048


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def metadata_similarity(
    df: pd.DataFrame,
    feature_cols: list[str],
    metric: Metric = "cosine",
    top_k: int = 100,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> sparse.csr_matrix:
    """Build a sparse top-k similarity matrix from entity metadata.

    Parameters
    ----------
    df : DataFrame
        One row per entity.
    feature_cols : list[str]
        Categorical columns to compare.
    metric : ``"cosine"`` | ``"overlap"``
        ``"cosine"`` compares one-hot encodings; ``"overlap"`` is the
        fraction of feature columns whose values match.
    top_k : int
        Number of most-similar neighbours to retain per row.
    batch_size : int
        Rows processed per chunk (controls peak memory).
    """
    features = df[feature_cols]
    if metric == "cosine":
        return _batched_topk(_one_hot_encode(features), top_k, batch_size, _cosine_chunk)
    if metric == "overlap":
        return _batched_topk(_category_codes(features), top_k, batch_size, _overlap_chunk)
    raise ValueError(f"Unknown metric {metric!r}")


def profile_similarity(
    profiles: sparse.csr_matrix,
    metric: Metric = "cosine",
    top_k: int = 100,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> sparse.csr_matrix:
    """Build a sparse top-k similarity matrix between interaction profiles.

    Each row of *profiles* is one entity's interaction weights across all
    targets (e.g. ``I`` for customers, ``I.T`` for products).  Two entities
    are similar when they interact with similar targets.

    Parameters
    ----------
    profiles : csr_matrix
        (n_entities x n_targets) interaction weights.
    metric : ``"cosine"`` | ``"overlap"``
        ``"cosine"`` compares the weights; ``"overlap"`` is cosine on the
        binarised profiles (shared targets / sqrt of both profile sizes).
    top_k, batch_size : int
        Same semantics as :func:`metadata_similarity`.
    """
    if metric == "overlap":
        profiles = (profiles > 0).astype(np.float32)
    elif metric != "cosine":
        raise ValueError(f"Unknown metric {metric!r}")
    return _batched_topk(profiles, top_k, batch_size, _cosine_chunk)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _one_hot_encode(features: pd.DataFrame) -> sparse.csr_matrix:
    enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    return enc.fit_transform(features.astype(str)).tocsr().astype(np.float32)


def _category_codes(features: pd.DataFrame) -> np.ndarray:
    """Return an (n, d) int array of category codes (one col per feature)."""
    return np.column_stack(
        [features[c].astype("category").cat.codes.values for c in features.columns]
    )


# ---------------------------------------------------------------------------
# Batched top-k engine
# ---------------------------------------------------------------------------


def _cosine_chunk(
    batch: sparse.spmatrix,
    full: sparse.spmatrix,
) -> np.ndarray:
    """Return dense (batch_size, n) cosine similarities."""
    return cosine_similarity(batch, full).astype(np.float32)


def _overlap_chunk(batch: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Return dense (batch_size, n) overlap similarities.

    Overlap = fraction of columns where values match.
    """
    matches = np.zeros((batch.shape[0], full.shape[0]), dtype=np.int64)
    for col in range(batch.shape[1]):
        matches += batch[:, col, None] == full[None, :, col]
    return (matches / batch.shape[1]).astype(np.float32)


def _batched_topk(
    features: np.ndarray | sparse.spmatrix,
    top_k: int,
    batch_size: int,
    sim_fn,
) -> sparse.csr_matrix:
    """Compute a sparse all-pairs similarity matrix of *features* in chunks.

    For each batch of rows, compute similarities against all rows, drop
    self-similarity, keep only the *top_k* positive values per row, and
    assemble a CSR matrix.
    """
    n = features.shape[0]
    top_k = min(top_k, n)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = sim_fn(features[start:end], features)  # (end - start, n) dense
        local = np.arange(end - start)
        sims[local, local + start] = 0.0

        if top_k < n:
            top_indices = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
        else:
            top_indices = np.broadcast_to(np.arange(n), sims.shape)

        batch_rows = np.repeat(np.arange(start, end), top_k)
        batch_cols = top_indices.ravel()
        batch_vals = np.take_along_axis(sims, top_indices, axis=1).ravel()

        mask = batch_vals > 0
        rows.append(batch_rows[mask])
        cols.append(batch_cols[mask])
        vals.append(batch_vals[mask])

    all_rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
    all_cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
    all_vals = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

    return sparse.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))
