# recs

Modular, extensible recommendation systems library.

## Install

```bash
pip install -e ".[dev]"            # library + pytest
pip install -e ".[dev,benchmark]"  # + matplotlib / openpyxl for benchmarks
```

## Quick start

```python
import pandas as pd
from recs import WeightedSimilarity

customers = pd.DataFrame({
    "customer_id": ["c0", "c1", "c2"],
    "region":      ["north", "south", "north"],
    "segment":     ["premium", "budget", "mid"],
})

products = pd.DataFrame({
    "product_id": ["p0", "p1", "p2", "p3"],
    "category":   ["electronics", "books", "electronics", "books"],
})

interactions = pd.DataFrame({
    "customer_id": ["c0", "c0", "c1", "c1", "c2", "c2"],
    "product_id":  ["p0", "p1", "p0", "p2", "p1", "p3"],
    "weight":      [5.0,  3.0,  4.0,  2.0,  1.0,  2.0],
})

model = WeightedSimilarity(
    w_product_metadata=0.25, w_product_interactions=0.25,    # PSM, IPSM
    w_customer_metadata=0.25, w_customer_interactions=0.25,  # CSM, ICSM
    metric="cosine",   # or "overlap"
    top_k=100,         # neighbours per row in similarity matrices
)

model.fit(customers, products, interactions)
recs = model.recommend(customer_ids=["c0", "c1"], n=5)
#   customer_id product_id     score  rank
# 0          c0         p2  0.311521     1
# 1          c0         p3  0.060060     2
# 2          c1         p3  0.189281     1
# 3          c1         p1  0.177552     2
```

Every model takes the same three tables in `fit()` and returns the same
`customer_id, product_id, score, rank` table from `recommend()`.

## Layout

```
src/recs/
  base.py          Recommender base class: indexing, interaction matrix,
                   seen-item exclusion, batched top-n ranking
  matrix.py        interaction_matrix, sparse_topk, top_n
  similarity.py    sparse top-k metadata / interaction-profile similarity
  models/
    weighted_similarity.py  WeightedSimilarity
    bipartite_graph.py      BipartiteGraphRec
    belief_propagation.py   BeliefPropagationRec
    baselines.py            MostPopular / UserCF / ItemCF / SVD baselines
benchmarks/        dataset loader, metrics, sweep runner, plotting
notebooks/         demo + exploratory benchmark notebooks
tests/
```

## Models

| Model | Scores |
|-------|--------|
| `WeightedSimilarity` | `R = (c·CSM + d·ICSM) @ I @ (a·PSM + b·IPSM)` |
| `BipartiteGraphRec` | `R = Σ w_k · C^((k-1)/2) @ I` over odd hop counts *k*, with `C = I Iᵀ` |
| `BeliefPropagationRec` | marginals of loopy BP on the item co-occurrence graph `IᵀI` |
| `MostPopularBaseline`, `UserCFBaseline`, `ItemCFBaseline`, `SVDBaseline` | classic references |

`WeightedSimilarity` blends five matrices built from the three tables:

| Matrix | Shape | Source |
|--------|-------|--------|
| **I**    | n x m | interaction weights |
| **PSM**  | m x m | product metadata similarity |
| **CSM**  | n x n | customer metadata similarity |
| **IPSM** | m x m | product interaction-profile similarity |
| **ICSM** | n x n | customer interaction-profile similarity |

All similarity matrices are **sparse** (top-k neighbours per row), so memory
and compute scale with `k * n` rather than `n^2`.

## Adding a new model

Subclass `recs.Recommender`.  The base class builds `self._customer_idx`,
`self._product_idx` and the sparse interaction matrix `self._interactions`
before calling your hook, and handles `recommend()` for you:

```python
from recs import Recommender

class MyModel(Recommender):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _fit(self, customers, products, interactions):
        I = self._interactions
        self._R = self.alpha * (I @ I.T) @ I   # precomputed sparse scores

    # Optional: score on the fly instead of precomputing self._R.
    # Must return a new dense (len(positions), n_products) array.
    # def _score(self, positions):
    #     ...
```

Set the class attribute `positive_scores_only = False` if your scores can be
negative.

## Benchmarks

UCI Online Retail, per-customer time split (80 / 20), Precision / Recall /
NDCG @ 10:

```bash
python -m benchmarks.run_benchmark
```

Results are written to `benchmarks/results/` (CSV + best-per-model plot).
Add configurations in `build_configs()` in `benchmarks/run_benchmark.py`.

## Tests

```bash
python -m pytest
```
