import numpy as np
import pandas as pd
import pytest

from recs import ItemCFBaseline, MostPopularBaseline, SVDBaseline, UserCFBaseline


@pytest.fixture()
def tables():
    customers = pd.DataFrame({"customer_id": ["a", "b", "c", "d", "cold"]})
    products = pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3", "never"]})
    interactions = pd.DataFrame({
        "customer_id": ["a", "a", "b", "b", "b", "c", "c", "d"],
        "product_id": ["p0", "p1", "p0", "p1", "p2", "p1", "p2", "p3"],
        "weight": [1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0],
    })
    return customers, products, interactions


def _dense(interactions, customers, products):
    return (
        interactions.pivot_table(index="customer_id", columns="product_id",
                                 values="weight", aggfunc="sum", fill_value=0.0)
        .reindex(index=customers["customer_id"], columns=products["product_id"],
                 fill_value=0.0)
        .to_numpy()
    )


def _cosine(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    return sim


ALL = [MostPopularBaseline, UserCFBaseline, ItemCFBaseline, SVDBaseline]


@pytest.mark.parametrize("cls", ALL)
def test_contract(cls, tables):
    customers, products, interactions = tables
    recs = cls().fit(customers, products, interactions).recommend(n=3)
    assert list(recs.columns) == ["customer_id", "product_id", "score", "rank"]
    seen = set(zip(interactions["customer_id"], interactions["product_id"]))
    assert seen.isdisjoint(zip(recs["customer_id"], recs["product_id"]))
    assert "never" not in set(recs["product_id"])


@pytest.mark.parametrize("cls", ALL)
def test_unknown_customer_raises(cls, tables):
    model = cls().fit(*tables)
    with pytest.raises(KeyError, match="Unknown customer_ids"):
        model.recommend(customer_ids=["zzz"])


def test_most_popular_ranks_by_distinct_customers(tables):
    recs = MostPopularBaseline().fit(*tables).recommend(customer_ids=["cold", "d"], n=2)
    # p1 has 3 buyers, p0 and p2 have 2 each
    assert recs[recs["customer_id"] == "cold"]["product_id"].tolist()[0] == "p1"
    assert recs[recs["customer_id"] == "d"]["score"].tolist() == [3.0, 2.0]


def test_user_cf_matches_dense_computation(tables):
    customers, products, interactions = tables
    I = _dense(interactions, customers, products)  # noqa: E741
    expected = _cosine(I) @ I  # all neighbours kept (top_neighbors >= n)

    model = UserCFBaseline(top_neighbors=10).fit(customers, products, interactions)
    recs = model.recommend(customer_ids=["a"], n=10)
    a = 0
    for pid, score in zip(recs["product_id"], recs["score"]):
        j = products["product_id"].tolist().index(pid)
        assert score == pytest.approx(expected[a, j], rel=1e-5)


def test_item_cf_matches_dense_computation(tables):
    customers, products, interactions = tables
    I = _dense(interactions, customers, products)  # noqa: E741
    expected = I @ _cosine(I.T)

    model = ItemCFBaseline().fit(customers, products, interactions)
    recs = model.recommend(customer_ids=["c"], n=10)
    c = 2
    assert recs["product_id"].tolist()[0] == products["product_id"][np.argmax(
        np.where(I[c] > 0, -np.inf, expected[c]))]
    for pid, score in zip(recs["product_id"], recs["score"]):
        j = products["product_id"].tolist().index(pid)
        assert score == pytest.approx(expected[c, j], rel=1e-5)


def test_svd_skips_customers_without_history(tables):
    model = SVDBaseline(n_components=2).fit(*tables)
    assert model.recommend(customer_ids=["cold"], n=3).empty
    assert len(model.recommend(customer_ids=["d"], n=3)) == 3
