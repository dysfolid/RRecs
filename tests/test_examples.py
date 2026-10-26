"""Smoke tests: the scripts in examples/ must keep running."""

import runpy
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def test_quickstart_runs(capsys):
    runpy.run_path(str(QUICKSTART), run_name="__main__")
    out = capsys.readouterr().out
    assert "Top 5 for two customers" in out
    assert "Hit rate@10" in out


def test_quickstart_item_models_beat_popularity():
    ex = runpy.run_path(str(QUICKSTART))
    customers, products, interactions = ex["make_tables"]()
    train, hidden = ex["hold_out_one"](interactions)

    def rate(model):
        return ex["hit_rate"](model, customers, products, train, hidden)

    popular = rate(ex["MostPopularBaseline"]())
    for model in [ex["ItemKNNRec"](top_k=10, alpha=0.7), ex["RP3BetaRec"](alpha=0.6, beta=0.6),
                  ex["EASERec"](l2=250)]:
        assert rate(model) > popular
