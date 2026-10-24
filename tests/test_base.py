import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recs.base import RESULT_COLUMNS, Recommender
from recs.matrix import interaction_matrix, sparse_topk, top_n


# ---------------------------------------------------------------------------
# A minimal model with hand-set scores
# ---------------------------------------------------------------------------

class FixedScores(Recommender):
    """Scores every customer with the same fixed product vector."""

    def __init__(self, scores, positive_scores_only=True):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.positive_scores_only = positive_scores_only

    def _fit(self, customers, products, interactions):
        pass

    def _score(self, positions):
        return np.tile(self.scores, (len(positions), 1))


@pytest.fixture()
def tables():
    customers = pd.DataFrame({"customer_id": ["a", "b", "c"]})
    products = pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3"]})
    interactions = pd.DataFrame({
        "customer_id": ["a", "a", "b"],
        "product_id": ["p1", "p3", "p0"],
        "weight": [1.0, 2.0, 1.0],
    })
    return customers, products, interactions


# ---------------------------------------------------------------------------
# Recommender.recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_ranks_and_excludes_seen(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        recs = model.recommend(customer_ids=["a"], n=10)
        assert recs["product_id"].tolist() == ["p0", "p2"]
        assert recs["rank"].tolist() == [1, 2]
        assert recs["score"].tolist() == [4.0, 2.0]

    def test_include_seen(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        recs = model.recommend(customer_ids=["a"], n=10, exclude_seen=False)
        assert recs["product_id"].tolist() == ["p0", "p1", "p2", "p3"]

    def test_drops_non_positive_scores(self, tables):
        model = FixedScores([0.0, -1.0, 2.0, 1.0]).fit(*tables)
        recs = model.recommend(customer_ids=["c"], n=10)
        assert recs["product_id"].tolist() == ["p2", "p3"]

    def test_negative_scores_allowed_when_configured(self, tables):
        model = FixedScores([0.0, -1.0, 2.0, 1.0], positive_scores_only=False)
        model.fit(*tables)
        recs = model.recommend(customer_ids=["a"], n=10)
        # p1 and p3 are seen by "a"; the rest keep their order incl. zero
        assert recs["product_id"].tolist() == ["p2", "p0"]

    def test_n_limits_results(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        recs = model.recommend(customer_ids=["c"], n=2)
        assert recs["product_id"].tolist() == ["p0", "p1"]
        assert model.recommend(customer_ids=["c"], n=0).empty

    def test_batching_does_not_change_results(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        full = model.recommend(n=3)
        model.recommend_batch_size = 1
        pd.testing.assert_frame_equal(model.recommend(n=3), full)

    def test_keeps_requested_order_and_duplicates(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        recs = model.recommend(customer_ids=["c", "a", "c"], n=1)
        assert recs["customer_id"].tolist() == ["c", "a", "c"]

    def test_default_is_all_customers(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        assert model.recommend(n=1)["customer_id"].tolist() == ["a", "b", "c"]

    def test_empty_request(self, tables):
        model = FixedScores([4.0, 3.0, 2.0, 1.0]).fit(*tables)
        recs = model.recommend(customer_ids=[], n=5)
        assert recs.empty
        assert list(recs.columns) == RESULT_COLUMNS

    def test_unknown_customer_raises(self, tables):
        model = FixedScores([1.0] * 4).fit(*tables)
        with pytest.raises(KeyError, match="Unknown customer_ids"):
            model.recommend(customer_ids=["zzz"])

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            FixedScores([1.0]).recommend()

    def test_unknown_interaction_ids_raise(self, tables):
        customers, products, interactions = tables
        bad = pd.concat([interactions, pd.DataFrame(
            {"customer_id": ["ghost"], "product_id": ["p0"], "weight": [1.0]}
        )])
        with pytest.raises(ValueError, match="unknown customer_id"):
            FixedScores([1.0] * 4).fit(customers, products, bad)


# ---------------------------------------------------------------------------
# recs.matrix helpers
# ---------------------------------------------------------------------------

class TestInteractionMatrix:
    def test_layout_and_duplicates_summed(self):
        I = interaction_matrix(  # noqa: E741
            pd.DataFrame({
                "customer_id": ["b", "a", "b"],
                "product_id": ["x", "y", "x"],
                "weight": [1.0, 2.0, 3.0],
            }),
            pd.Index(["a", "b"]),
            pd.Index(["x", "y"]),
        )
        np.testing.assert_array_equal(I.toarray(), [[0.0, 2.0], [4.0, 0.0]])


class TestTopN:
    def test_matches_full_sort(self):
        rng = np.random.default_rng(0)
        scores = rng.random((20, 50))
        expected = np.argsort(-scores, axis=1)[:, :7]
        np.testing.assert_array_equal(top_n(scores, 7), expected)

    def test_n_at_least_width(self):
        scores = np.array([[1.0, 3.0, 2.0]])
        np.testing.assert_array_equal(top_n(scores, 5), [[1, 2, 0]])

    def test_n_zero(self):
        assert top_n(np.ones((3, 4)), 0).shape == (3, 0)


class TestSparseTopk:
    def test_keeps_largest_per_row(self):
        mat = sparse.csr_matrix(np.array([
            [5.0, 1.0, 4.0, 2.0],
            [0.0, 3.0, 0.0, 0.0],
        ]))
        out = sparse_topk(mat, 2).toarray()
        np.testing.assert_array_equal(out, [[5.0, 0.0, 4.0, 0.0], [0.0, 3.0, 0.0, 0.0]])

    def test_zero_diag(self):
        mat = sparse.csr_matrix(np.array([[9.0, 1.0], [2.0, 9.0]]))
        out = sparse_topk(mat, 5, zero_diag=True).toarray()
        np.testing.assert_array_equal(out, [[0.0, 1.0], [2.0, 0.0]])
