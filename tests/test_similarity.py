import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recs.similarity import metadata_similarity, profile_similarity


def _profiles():
    return sparse.csr_matrix(np.array([
        [1.0, 2.0, 0.0, 0.0],
        [3.0, 0.0, 1.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ], dtype=np.float32))


def _cosine(dense):
    norm = dense / np.linalg.norm(dense, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, 0.0)
    return sim


class TestProfileSimilarity:
    def test_cosine_matches_dense(self):
        out = profile_similarity(_profiles(), metric="cosine", top_k=10).toarray()
        np.testing.assert_allclose(out, _cosine(_profiles().toarray()), rtol=1e-6)

    def test_overlap_is_cosine_on_binary(self):
        out = profile_similarity(_profiles(), metric="overlap", top_k=10).toarray()
        binary = (_profiles().toarray() > 0).astype(float)
        np.testing.assert_allclose(out, _cosine(binary), rtol=1e-6)

    def test_top_k_per_row(self):
        rng = np.random.default_rng(1)
        profiles = sparse.random(60, 30, density=0.3, random_state=rng, format="csr")
        out = profile_similarity(profiles, metric="cosine", top_k=5, batch_size=7)
        assert np.diff(out.indptr).max() <= 5
        assert out.diagonal().sum() == 0

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            profile_similarity(_profiles(), metric="jaccard")


class TestMetadataSimilarity:
    def test_overlap_is_fraction_of_matching_columns(self):
        df = pd.DataFrame({
            "a": ["x", "x", "y"],
            "b": ["u", "v", "v"],
        })
        out = metadata_similarity(df, ["a", "b"], metric="overlap", top_k=10).toarray()
        np.testing.assert_allclose(out, [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ])

    def test_cosine_on_one_hot(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["u", "v", "v"]})
        out = metadata_similarity(df, ["a", "b"], metric="cosine", top_k=10).toarray()
        # each row has two active one-hot features; one shared -> 1/2
        np.testing.assert_allclose(out[0, 1], 0.5, rtol=1e-6)
        assert out[0, 2] == 0
