import pandas as pd
import pytest

from benchmarks.datasets import BenchmarkData, _split_by_time
from benchmarks.metrics import evaluate
from benchmarks.runner import select_and_test
from recs import ItemKNNRec, MostPopularBaseline
from tests.conftest import make_data


def test_split_by_time_holds_out_latest_rows():
    dated = pd.DataFrame({
        "CustomerID": ["a"] * 5 + ["b"] * 2 + ["c"],
        "StockCode": list("pqrst") + ["p", "q"] + ["r"],
        "last_date": pd.to_datetime([1, 2, 3, 4, 5, 1, 2, 1], unit="D"),
    })
    train, held_out = _split_by_time(dated, 0.4)
    assert train.groupby("CustomerID")["StockCode"].apply(list).to_dict() == {
        "a": ["p", "q", "r"], "b": ["p"], "c": ["r"],
    }
    assert held_out.groupby("CustomerID")["StockCode"].apply(list).to_dict() == {
        "a": ["s", "t"], "b": ["q"],
    }


def test_evaluate_coverage_and_popularity():
    recs = {"a": ["p", "q"], "b": ["p"]}
    truth = {"a": {"p"}, "b": {"r"}}
    out = evaluate(recs, truth, ["a", "b"], k=2,
                   item_popularity={"p": 10, "q": 2}, n_items=4)
    assert out["Coverage@2"] == pytest.approx(2 / 4)
    assert out["AvgPop@2"] == pytest.approx((10 + 2 + 10) / 3)
    assert out["Precision@2"] == pytest.approx((1 / 2 + 0) / 2)


def _benchmark(seed):
    customers, products, interactions = make_data(seed=seed)
    test = interactions.sample(frac=0.2, random_state=seed)
    train = interactions.drop(test.index)
    truth = test.groupby("customer_id")["product_id"].apply(set).to_dict()
    return BenchmarkData(customers, products, train, truth, sorted(truth), name=f"s{seed}")


def test_select_and_test_picks_best_per_class_on_validation():
    data = _benchmark(0)
    data.validation = _benchmark(1)
    configs = [
        ("popular", MostPopularBaseline()),
        ("knn k=2", ItemKNNRec(top_k=2)),
        ("knn k=20", ItemKNNRec(top_k=20)),
    ]
    val, test = select_and_test(configs, data, k=5)
    assert len(val) == 3
    best_knn = val[val["model_class"] == "ItemKNNRec"].iloc[0]["name"]
    assert sorted(test["name"]) == sorted(["popular", best_knn])
