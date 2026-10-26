"""Quick start: fit a recommender, get recommendations, compare models.

Uses a small synthetic shop, so it runs in a few seconds without downloads.
Run from the repository root::

    python examples/quickstart.py

For the full benchmark protocol on real data (validation-based model
selection, more metrics) see ``python -m benchmarks.run_benchmark``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recs import EASERec, ItemKNNRec, MostPopularBaseline, RP3BetaRec


def make_tables(
    n_customers: int = 2000,
    n_products: int = 300,
    n_tastes: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A toy shop: each customer mostly buys from one 'taste' group of products."""
    rng = np.random.default_rng(seed)
    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n_customers)],
        "region": rng.choice(["north", "south", "east", "west"], n_customers),
    })
    products = pd.DataFrame({
        "product_id": [f"p{j}" for j in range(n_products)],
        "category": [f"taste{j % n_tastes}" for j in range(n_products)],
    })

    taste_of_product = np.arange(n_products) % n_tastes
    popularity = 1.0 / np.arange(1, n_products + 1) ** 0.5
    rows = []
    for i, taste in enumerate(rng.integers(n_tastes, size=n_customers)):
        affinity = popularity * np.where(taste_of_product == taste, 8.0, 1.0)
        basket = rng.choice(n_products, size=rng.integers(3, 15), replace=False,
                            p=affinity / affinity.sum())
        rows += [(f"c{i}", f"p{j}", round(rng.uniform(1, 50), 2)) for j in basket]
    interactions = pd.DataFrame(rows, columns=["customer_id", "product_id", "weight"])
    return customers, products, interactions


def hold_out_one(
    interactions: pd.DataFrame, seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Hide one random purchase per customer; returns (train, hidden product per customer)."""
    hidden = interactions.groupby("customer_id").sample(n=1, random_state=seed)
    train = interactions.drop(hidden.index)
    return train, hidden.set_index("customer_id")["product_id"]


def hit_rate(model, customers, products, train, hidden, n=10) -> float:
    """Share of customers whose hidden purchase is in their top-*n*."""
    model.fit(customers, products, train)
    recs = model.recommend(customer_ids=hidden.index, n=n)
    hits = recs.merge(hidden.rename("product_id").reset_index(),
                      on=["customer_id", "product_id"])
    return len(hits) / len(hidden)


def main() -> None:
    customers, products, interactions = make_tables()
    print(f"{len(customers)} customers, {len(products)} products, "
          f"{len(interactions)} interactions\n")

    # 1. Fit a model and recommend.  Every model takes the same three tables.
    model = EASERec(l2=250).fit(customers, products, interactions)
    print("Top 5 for two customers:")
    print(model.recommend(customer_ids=["c0", "c1"], n=5).to_string(index=False))

    # 2. Already-bought products are excluded by default; opt back in if needed.
    with_seen = model.recommend(customer_ids=["c0"], n=5, exclude_seen=False)
    print(f"\nWith already-bought products allowed: {with_seen['product_id'].tolist()}")

    # 3. Compare models on a hold-out: hide one purchase per customer.
    train, hidden = hold_out_one(interactions)
    print("\nHit rate@10 (hidden purchase in the top 10):")
    for name, candidate in [
        ("Popular", MostPopularBaseline()),
        ("ItemKNN", ItemKNNRec(top_k=10, alpha=0.7)),
        ("RP3beta", RP3BetaRec(alpha=0.6, beta=0.6)),
        ("EASE", EASERec(l2=250)),
    ]:
        print(f"  {name:8s} {hit_rate(candidate, customers, products, train, hidden):.3f}")


if __name__ == "__main__":
    main()
