import numpy as np
import pandas as pd
import pytest

from recs import EASERec, ItemKNNRec, RP3BetaRec
from tests.conftest import make_data


def _binary(customers, products, interactions):
    return (
        interactions.assign(one=1.0)
        .pivot_table(index="customer_id", columns="product_id", values="one",
                     aggfunc="max", fill_value=0.0)
        .reindex(index=customers["customer_id"], columns=products["product_id"],
                 fill_value=0.0)
        .to_numpy()
    )


@pytest.fixture()
def small():
    return make_data(n_customers=40, n_products=15, n_interactions=200, seed=11)


def test_item_knn_cosine_matches_dense(small):
    X = _binary(*small)
    shared = X.T @ X
    np.fill_diagonal(shared, 0.0)
    buyers = X.sum(axis=0)
    expected = shared / np.sqrt(np.outer(buyers, buyers))

    model = ItemKNNRec(top_k=100).fit(*small)  # top_k >= catalogue: keep all
    np.testing.assert_allclose(model._W.toarray(), expected)


def test_item_knn_alpha_one_is_conditional_probability(small):
    X = _binary(*small)
    shared = X.T @ X
    np.fill_diagonal(shared, 0.0)
    expected = shared / X.sum(axis=0)[:, None]  # P(i bought | j bought)

    model = ItemKNNRec(top_k=100, alpha=1.0).fit(*small)
    np.testing.assert_allclose(model._W.toarray(), expected)


def test_item_knn_top_k_and_symmetrize(small):
    one_sided = ItemKNNRec(top_k=3, symmetrize=False).fit(*small)
    assert np.diff(one_sided._W.indptr).max() <= 3

    both = ItemKNNRec(top_k=3, symmetrize=True).fit(*small)
    structure = (both._W != 0).toarray()
    np.testing.assert_array_equal(structure, structure.T)


def test_item_knn_scores_sum_basket_rows(small):
    model = ItemKNNRec(top_k=5).fit(*small)
    X = _binary(*small)
    np.testing.assert_allclose(model._score(np.array([0, 1])), X[:2] @ model._W.toarray())


def test_ease_matches_closed_form(small):
    X = _binary(*small)
    P = np.linalg.inv(X.T @ X + 10.0 * np.eye(X.shape[1]))
    expected = -P / np.diag(P)
    np.fill_diagonal(expected, 0.0)

    model = EASERec(l2=10.0).fit(*small)
    np.testing.assert_allclose(model._W, expected, atol=1e-10)


def test_rp3beta_matches_dense(small):
    X = _binary(*small)
    alpha, beta = 0.8, 0.4
    basket = X.sum(axis=1, keepdims=True)
    buyers = X.sum(axis=0, keepdims=True)
    p_item = np.divide(X, basket, out=np.zeros_like(X), where=basket > 0) ** alpha
    p_cust = np.divide(X, buyers, out=np.zeros_like(X), where=buyers > 0).T ** alpha
    expected = (p_cust @ p_item) / np.where(buyers > 0, buyers, 1.0) ** beta
    np.fill_diagonal(expected, 0.0)

    model = RP3BetaRec(alpha=alpha, beta=beta, top_k=100).fit(*small)
    np.testing.assert_allclose(model._W.toarray(), expected, rtol=1e-10)


@pytest.mark.parametrize("model", [ItemKNNRec(), EASERec(l2=50.0), RP3BetaRec()])
def test_contract(model, data):
    customers, products, interactions = data
    recs = model.fit(customers, products, interactions).recommend(n=5)
    assert list(recs.columns) == ["customer_id", "product_id", "score", "rank"]
    assert recs["rank"].max() <= 5
    seen = set(zip(interactions["customer_id"], interactions["product_id"]))
    assert seen.isdisjoint(zip(recs["customer_id"], recs["product_id"]))


def test_item_knn_invalid_basket():
    with pytest.raises(ValueError, match="basket"):
        ItemKNNRec(basket="tfidf")
