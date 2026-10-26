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
from recs import EASERec

customers = pd.DataFrame({
    "customer_id": ["c0", "c1", "c2"],
    "region":      ["north", "south", "north"],
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

# l2 is EASE's regulariser: it grows with data size (250 was best on the
# 4k-customer benchmark), so tune it on a validation split.
model = EASERec(l2=1.0)
model.fit(customers, products, interactions)
recs = model.recommend(customer_ids=["c0", "c1"], n=5)
#   customer_id product_id     score  rank
# 0          c0         p3  0.230769     1
# 1          c0         p2  0.230769     2
# 2          c1         p1  0.200000     1
```

Every model takes the same three tables in `fit()` and returns the same
`customer_id, product_id, score, rank` table from `recommend()`, so models
are interchangeable.  A runnable, longer example (recommending, including
already-bought products, comparing models on a hold-out) is in
[examples/quickstart.py](examples/quickstart.py):

```bash
python examples/quickstart.py
```

## Input data

| Table | Required columns | Notes |
|-------|------------------|-------|
| `customers` | `customer_id` (unique) | Every customer you want to score must be listed.  Other columns are optional metadata. |
| `products` | `product_id` (unique) | The catalogue that can be recommended.  Other columns are optional metadata. |
| `interactions` | `customer_id`, `product_id`, `weight` | One row per purchase, view, rating, … with a **positive** weight (spend, quantity, rating, or just 1). |

- Duplicate (customer, product) rows are summed.  An interaction with an id
  missing from `customers` / `products` raises `ValueError`.
- **Weights** matter only to some models.  `EASERec`, `RP3BetaRec`,
  `ItemKNNRec` (by default), `BipartiteGraphRec(aggregation="count")` and
  `MostPopularBaseline` use only *whether* a customer bought a product.
  `WeightedSimilarity`, `UserCFBaseline`, `ItemCFBaseline`, `SVDBaseline` and
  `BeliefPropagationRec` (by default) use the weights themselves.
- **Metadata** columns are used only by `WeightedSimilarity` (metadata
  similarity) and `BipartiteGraphRec` (customer groups, penalties and
  cold-start groups).  Every other model ignores them.
- `recommend()` excludes products the customer already has unless
  `exclude_seen=False`, and returns at most `n` rows per customer.  It can
  return fewer: products with no positive score are never recommended
  (except by `SVDBaseline` and `BeliefPropagationRec(score="logodds")`,
  whose scores can be negative).  An unknown `customer_id` raises `KeyError`.
- **Customers with no interactions get no recommendations** from the
  collaborative models.  Use `MostPopularBaseline` as a fallback, or
  `BipartiteGraphRec(cold_start_n=...)`, or `WeightedSimilarity` with customer
  metadata weight.

## Choosing a model

Test results on UCI Online Retail (4,078 customers, 3,149 products), with
each model's hyperparameters picked on the validation split (see
[Benchmarks](#benchmarks)).  Coverage is the share of the catalogue
recommended to at least one customer.  Times are single-process on a desktop
CPU; "at 30k products" is synthetic data shaped like Online Retail (50k
customers, average basket 50).

| Model (selected config) | NDCG@10 | Coverage | Fit / per-customer, Online Retail | Fit / per-customer, at 30k products |
|-------------------------|---------|----------|-----------------------------------|-------------------------------------|
| `EASERec(l2=250)` | **0.207** | 47% | 1 s / 0.1 ms | not run: needs ~22 GB (see below) |
| `RP3BetaRec(alpha=0.6, beta=0.6)` | 0.184 | 71% | 0.4 s / 0.05 ms | 5 s / 0.4 ms |
| `ItemKNNRec(top_k=10, alpha=0.7)` | 0.173 | 58% | 0.6 s / 0.06 ms | 12 s / 0.4 ms |
| `WeightedSimilarity` (product interactions only, overlap, `top_k=10`) | 0.164 | 76% | 0.1 s / 0.04 ms | 12 s / 0.4 ms |
| `BeliefPropagationRec` (positive potential, cosine couplings) | 0.140 | 45% | 0.4 s / 25 ms | 10 s / 75 ms |
| `BipartiteGraphRec` (3-hop count, `top_k=200`) | 0.122 | 17% | 0.7 s / 0.02 ms | 159 s / 0.3 ms, 8.5 GB score matrix |
| `UserCFBaseline(top_neighbors=100)` | 0.092 | 23% | 0.3 s / 0.02 ms | |
| `SVDBaseline(n_components=80)` | 0.080 | 20% | 0.1 s / 0.03 ms | |
| `MostPopularBaseline()` | 0.058 | 1% | < 0.1 s | |

Recommendations:

- **Default to `EASERec`.**  It is the most accurate here and fits in about a
  second.  It holds three dense products × products matrices, so memory
  grows with the square of the catalogue: about 2.4 GB at 10k products
  (fit 8.5 s), about 22 GB at 30k.  Treat 20–30k products as its practical
  limit.
- **For larger catalogues, use `RP3BetaRec` or `ItemKNNRec`.**  Both are
  sparse, fit in seconds at 30k products and score a customer in under a
  millisecond.  Their cost grows with the number of product pairs bought
  together, not with the catalogue squared.
- **For wider coverage of the catalogue, use `RP3BetaRec`.**  It reaches 71%
  of the catalogue against EASE's 47%; its `beta` penalises popular products.
- **Avoid `BipartiteGraphRec` and `WeightedSimilarity` with customer-side
  similarity on large data.**  They precompute a customers × products score
  matrix that becomes nearly dense (1 billion entries at 50k × 30k).
- **`BeliefPropagationRec` is experimental.**  It is roughly 200–500× slower
  per customer than the item models and less accurate than its own linear
  approximation (`ItemKNNRec` on the same graph).
- **Re-tune on your data.**  These hyperparameters were selected on Online
  Retail; use `benchmarks.runner.select_and_test` to pick yours on a
  validation split.

## Layout

```
src/recs/
  base.py          Recommender base class: indexing, interaction matrix,
                   seen-item exclusion, batched top-n ranking
  matrix.py        interaction_matrix, sparse_topk, top_n
  similarity.py    sparse top-k metadata / interaction-profile similarity
  models/
    item_based.py           ItemKNNRec, EASERec, RP3BetaRec
    weighted_similarity.py  WeightedSimilarity
    bipartite_graph.py      BipartiteGraphRec
    belief_propagation.py   BeliefPropagationRec
    baselines.py            MostPopular / UserCF / ItemCF / SVD baselines
benchmarks/        dataset loader, metrics, sweep runner, plotting
examples/          runnable usage examples
notebooks/         demo notebook
tests/
```

## Models

| Model | Scores |
|-------|--------|
| `EASERec` | basket @ closed-form linear item model (Steck, 2019) |
| `RP3BetaRec` | basket @ popularity-penalised 3-step random walk (Paudel et al., 2016) |
| `ItemKNNRec` | basket @ top-k shared-buyer similarity (asymmetric cosine, shrinkage) |
| `WeightedSimilarity` | `R = (c·CSM + d·ICSM) @ I @ (a·PSM + b·IPSM)` |
| `BipartiteGraphRec` | `R = Σ w_k · C^((k-1)/2) @ I` over odd hop counts *k*, with `C = I Iᵀ` |
| `BeliefPropagationRec` | beliefs of loopy BP on the item co-occurrence graph (vectorised; see its docstring for the coupling / evidence / prior / potential options) |
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

UCI Online Retail, per-customer time split (80 / 20) into train / test, with
the last 20% of each customer's training history held out again as a
validation split.  Every config is scored on validation; the best config of
each model class is refitted on the full training data and scored **once**
on test.  Metrics: Precision / Recall / NDCG @ 10, plus catalogue coverage
and the average popularity of recommended items.

```bash
python -m benchmarks.run_benchmark               # 4 parallel workers
python -m benchmarks.run_benchmark --jobs 8 --include-bp
```

Results are written to `benchmarks/results/` (`validation_*.csv`,
`test_*.csv` and a best-per-model plot).  Add configurations in
`build_configs()` in `benchmarks/run_benchmark.py`.  The parsed Excel file is
cached as `benchmarks/data/Online Retail.pkl`.

## Tests

```bash
python -m pytest
```

The tests include smoke tests that run the scripts in `examples/`.
