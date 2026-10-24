import numpy as np
import pandas as pd
import pytest


def make_data(n_customers=100, n_products=50, n_interactions=500, seed=42):
    """Random customers / products / interactions tables for model tests."""
    rng = np.random.default_rng(seed)

    customers = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n_customers)],
        "region": rng.choice(["north", "south", "east", "west"], n_customers),
        "segment": rng.choice(["budget", "mid", "premium"], n_customers),
    })

    products = pd.DataFrame({
        "product_id": [f"p{i}" for i in range(n_products)],
        "category": rng.choice(["catA", "catB", "catC", "catD"], n_products),
    })

    cids = rng.choice(customers["customer_id"].values, n_interactions)
    pids = rng.choice(products["product_id"].values, n_interactions)
    interactions = pd.DataFrame({
        "customer_id": cids,
        "product_id": pids,
        "weight": rng.uniform(0.1, 5.0, n_interactions).round(2),
    }).drop_duplicates(subset=["customer_id", "product_id"])

    return customers, products, interactions


@pytest.fixture()
def data():
    return make_data()
