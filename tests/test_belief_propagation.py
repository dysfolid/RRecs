import numpy as np
import pandas as pd
import pytest

from recs.models.belief_propagation import BeliefPropagationRec
from tests.conftest import make_data


def test_bp_prefers_co_occurring_item():
    """Item adjacent to the basket in S should score above an isolated SKU."""
    customers = pd.DataFrame({
        "customer_id": ["c0", "c1", "c2", "c3"],
        "x": [0, 0, 0, 0],
    })
    products = pd.DataFrame({
        "product_id": ["p0", "p1", "p2", "p3"],
        "x": [0, 0, 0, 0],
    })
    interactions = pd.DataFrame({
        "customer_id": ["c0", "c1", "c2", "c2", "c3"],
        "product_id": ["p0", "p1", "p0", "p1", "p0"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    })

    model = BeliefPropagationRec(top_k=10, beta=1.5, max_iter=25, damping=0.4)
    model.fit(customers, products, interactions)

    s = model._scores_for_user(
        model._interactions.getrow(model._customer_idx.get_loc("c3")),
        model._S,
        len(model._product_idx),
    )
    i1 = model._product_idx.get_loc("p1")
    i2 = model._product_idx.get_loc("p2")
    assert s[i1] > s[i2]


def test_bp_chain_propagation():
    """p0–p1–p2 chain: buying p0 should lift p2 vs isolated p3."""
    customers = pd.DataFrame({"customer_id": ["c0", "c1", "c2"], "x": [0, 0, 0]})
    products = pd.DataFrame({
        "product_id": ["p0", "p1", "p2", "p3"],
        "x": [0, 0, 0, 0],
    })
    interactions = pd.DataFrame({
        "customer_id": ["c0", "c0", "c1", "c1", "c2", "c2"],
        "product_id": ["p0", "p1", "p1", "p2", "p2", "p3"],
        "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

    model = BeliefPropagationRec(top_k=10, beta=1.2, max_iter=25, damping=0.45)
    model.fit(customers, products, interactions)

    s = model._scores_for_user(
        model._interactions.getrow(0),
        model._S,
        len(model._product_idx),
    )
    # c0 only bought p0; p2 is two hops via p1, p3 only touches c2
    i2 = model._product_idx.get_loc("p2")
    i3 = model._product_idx.get_loc("p3")
    assert s[i2] > s[i3]


def test_fit_recommend_smoke():
    rng = np.random.default_rng(0)
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(20)],
        "z": rng.integers(0, 3, 20),
    })
    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(15)],
        "z": rng.integers(0, 3, 15),
    })
    cids = rng.choice(customers["customer_id"].values, 80)
    pids = rng.choice(products["product_id"].values, 80)
    interactions = pd.DataFrame({
        "customer_id": cids,
        "product_id": pids,
        "weight": rng.uniform(0.5, 2.0, 80),
    }).drop_duplicates(subset=["customer_id", "product_id"])

    model = BeliefPropagationRec(top_k=8, max_iter=8)
    model.fit(customers, products, interactions)
    recs = model.recommend(n=5)
    assert set(recs.columns) == {"customer_id", "product_id", "score", "rank"}
    assert recs["rank"].max() <= 5


def test_not_fitted_raises():
    m = BeliefPropagationRec()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.recommend()


# ---------------------------------------------------------------------------
# Reference engine: the original per-edge Python loop, kept to pin the
# vectorised engine to the same model.
# ---------------------------------------------------------------------------

def _reference_scores(model, user_row):
    """Original implementation of BeliefPropagationRec._scores_for_user."""
    from recs.models.belief_propagation import _expand_item_subgraph

    S, m = model._S, model._S.shape[0]
    scores = np.zeros(m)
    basket = {int(i) for i in user_row.indices}
    w_map = {int(i): float(w) for i, w in zip(user_row.indices, user_row.data)}
    max_w = max(w_map.values())
    V = _expand_item_subgraph(basket, S, model.max_subgraph_nodes)
    if V <= basket:
        return scores
    glob = sorted(V)
    loc = {g: i for i, g in enumerate(glob)}
    L = len(glob)
    neighbors = [[] for _ in range(L)]
    edge_w, w_max = {}, 1e-9
    for g in glob:
        row = S.getrow(g)
        for h, w in zip(row.indices, row.data):
            if h not in loc or loc[g] == loc[h]:
                continue
            a, b = loc[g], loc[h]
            key = (min(a, b), max(a, b))
            if key not in edge_w:
                edge_w[key] = float(w)
                w_max = max(w_max, float(w))
            if b not in neighbors[a]:
                neighbors[a].append(b)
            if a not in neighbors[b]:
                neighbors[b].append(a)
    edge_w = {k: v / w_max for k, v in edge_w.items()}
    phi = np.full((L, 2), 0.5)
    for g in basket:
        p1 = 0.5 + 0.5 * min(1.0, w_map[g] / max_w)
        phi[loc[g]] = [1.0 - p1, p1]

    msgs = {(j, i): np.array([0.5, 0.5]) for i in range(L) for j in neighbors[i]}
    for _ in range(model.max_iter):
        new = {}
        for i in range(L):
            for j in neighbors[i]:
                J = model.beta * edge_w[(min(i, j), max(i, j))]
                psi = np.array([[np.exp(J), np.exp(-J)], [np.exp(-J), np.exp(J)]])
                cavity = phi[j].copy()
                for k in neighbors[j]:
                    if k != i:
                        cavity *= msgs[(k, j)]
                out = psi @ cavity
                new[(j, i)] = model.damping * msgs[(j, i)] + (1 - model.damping) * out / out.sum()
        msgs = new
    for i in range(L):
        b = phi[i].copy()
        for j in neighbors[i]:
            b *= msgs[(j, i)]
        scores[glob[i]] = b[1] / b.sum()
    return scores


def test_matches_reference_engine():
    customers, products, interactions = make_data(n_customers=60, n_products=25,
                                                  n_interactions=300, seed=3)
    for kw in [dict(top_k=6, max_iter=6, max_subgraph_nodes=15),
               dict(top_k=5, beta=1.4, max_iter=5, damping=0.3, max_subgraph_nodes=12)]:
        model = BeliefPropagationRec(**kw).fit(customers, products, interactions)
        m = len(model._product_idx)
        for pos in range(20):
            row = model._interactions.getrow(pos)
            if row.nnz == 0:
                continue
            np.testing.assert_allclose(
                model._scores_for_user(row, model._S, m),
                _reference_scores(model, row),
                atol=1e-12,
            )


def test_high_degree_hub_does_not_underflow():
    """A hub with 120 neighbours: products of 120 raw messages ~ 1e-36 used to
    collapse every belief to 0.5."""
    n = 121
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)]})
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(n + 1)]})
    rows = [("c0", "p0")] + [(f"c{i}", p) for i in range(1, n) for p in ("p0", f"p{i}")]
    interactions = pd.DataFrame(rows, columns=["customer_id", "product_id"]).assign(weight=1.0)

    model = BeliefPropagationRec(top_k=200, max_iter=10).fit(customers, products, interactions)
    s = model._scores_for_user(model._interactions.getrow(0), model._S, n + 1)
    neighbours = s[1:n]
    assert np.all(neighbours > 0.5)
    assert s[n] == 0.0  # p<n> is never bought: outside the subgraph


def test_logodds_ranks_like_prob():
    customers, products, interactions = make_data(n_customers=60, n_products=25,
                                                  n_interactions=300, seed=5)
    kw = dict(top_k=8, coupling="cosine", evidence=1.0, prior=0.05, potential="positive",
              beta=0.3)
    prob = BeliefPropagationRec(**kw).fit(customers, products, interactions)
    logodds = BeliefPropagationRec(score="logodds", **kw).fit(customers, products, interactions)
    a, b = prob.recommend(n=5), logodds.recommend(n=5)
    assert a["product_id"].tolist() == b["product_id"].tolist()
    assert (b["score"] < 0).any()  # low prior: log-odds below zero still ranked


def test_positive_potential_never_pushes_down():
    """With psi = [[1, 1], [1, e^J]], no item falls below its prior."""
    customers, products, interactions = make_data(n_customers=60, n_products=25,
                                                  n_interactions=300, seed=6)
    model = BeliefPropagationRec(top_k=8, potential="positive", evidence=1.0, prior=0.2,
                                 coupling="count", beta=2.0).fit(customers, products, interactions)
    s = model._scores_for_user(model._interactions.getrow(0), model._S, len(model._product_idx))
    assert s[s > 0].min() >= 0.2 - 1e-12


def test_invalid_options_raise():
    for kw in [dict(coupling="pmi"), dict(potential="potts"), dict(score="rank"),
               dict(prior=0.0), dict(evidence=1.5)]:
        with pytest.raises(ValueError):
            BeliefPropagationRec(**kw)
