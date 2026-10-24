"""Benchmark script: model selection on validation, one test run per model.

Every config is scored on the validation split (the last part of each
customer's training history); the best config of each model class is then
refitted on the full training data and scored once on the test split.

Run from the project root::

    python -m benchmarks.run_benchmark [--jobs N] [--include-bp]

Requires matplotlib for the summary plot (``pip install matplotlib``).
"""

from __future__ import annotations

import argparse

from benchmarks.datasets import load_online_retail
from benchmarks.plotting import plot_best_per_family
from benchmarks.runner import select_and_test
from recs import (
    BeliefPropagationRec,
    BipartiteGraphRec,
    EASERec,
    ItemCFBaseline,
    ItemKNNRec,
    MostPopularBaseline,
    Recommender,
    RP3BetaRec,
    SVDBaseline,
    UserCFBaseline,
    WeightedSimilarity,
)

K = 10
RESULTS_DIR = "benchmarks/results"

# --- Grid constants (edit here to explore further) ---
WS_OVERLAP_K = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # fine steps where optimum likely lies
WS_OVERLAP_K_TAIL = [60, 75]  # check decay after 50
WS_COSINE_K = [30, 50, 100]  # sparse: cosine was uniformly weak vs overlap
WS_PRODUCT_ONLY_K = [10, 15, 25, 50]  # item-side similarity alone (item-based CF)

GRAPH_3HOP_COUNT_K = [75, 100, 125, 150, 175, 200, 250]  # extend past 150; anchor low
GRAPH_3HOP_MULT_K = [30, 50, 100]  # multiply underperforms; few reference points

ITEM_KNN_K = [10, 25, 50, 100]
ITEM_KNN_ALPHA = [0.3, 0.5, 0.7]  # < 0.5 favours niche, > 0.5 popular candidates
EASE_L2 = [50, 100, 250, 500, 1000, 2500]
RP3_ALPHA = [0.6, 1.0]
RP3_BETA = [0.0, 0.3, 0.6]


def _ws_interactions_only(**kwargs) -> WeightedSimilarity:
    """WeightedSimilarity using only the two interaction-similarity matrices."""
    return WeightedSimilarity(
        w_product_metadata=0,
        w_product_interactions=1,
        w_customer_metadata=0,
        w_customer_interactions=1,
        **kwargs,
    )


def build_configs(include_bp: bool = False) -> list[tuple[str, Recommender]]:
    """Return (name, model) pairs for the benchmark sweep."""
    configs: list[tuple[str, Recommender]] = []

    # ---- Classic baselines ----
    configs.append(("Popular", MostPopularBaseline()))
    for n_neigh in [30, 40, 50, 100]:
        configs.append((f"UserCF nn={n_neigh}", UserCFBaseline(top_neighbors=n_neigh)))
    configs.append(("ItemCF", ItemCFBaseline()))
    for n_comp in [20, 50, 80]:
        configs.append((f"SVD n={n_comp}", SVDBaseline(n_components=n_comp)))

    # ---- WeightedSimilarity: overlap, fine top_k in 10–50, then the tail ----
    for top_k in WS_OVERLAP_K + WS_OVERLAP_K_TAIL:
        configs.append((
            f"WS overlap k={top_k}",
            _ws_interactions_only(metric="overlap", top_k=top_k),
        ))

    # ---- WeightedSimilarity: cosine (sparse reference only) ----
    for top_k in WS_COSINE_K:
        configs.append((
            f"WS cosine k={top_k}",
            _ws_interactions_only(metric="cosine", top_k=top_k),
        ))

    # ---- WeightedSimilarity: small metadata blend (overlap) ----
    for meta_w in [0.02, 0.05, 0.08]:
        configs.append((
            f"WS overlap meta={meta_w} k=50",
            WeightedSimilarity(
                w_product_metadata=meta_w,
                w_product_interactions=1 - meta_w,
                w_customer_metadata=meta_w,
                w_customer_interactions=1 - meta_w,
                metric="overlap",
                top_k=50,
            ),
        ))

    # ---- WeightedSimilarity: per-matrix top_k (overlap, interaction-only) ----
    for k_ic, k_ip in [(25, 100), (30, 100), (40, 120), (50, 100), (50, 150)]:
        configs.append((
            f"WS ovl k_ic={k_ic} k_ip={k_ip}",
            _ws_interactions_only(
                metric="overlap", top_k=50, top_k_icsm=k_ic, top_k_ipsm=k_ip,
            ),
        ))

    # ---- WeightedSimilarity: product interactions only (no customer side) ----
    for top_k in WS_PRODUCT_ONLY_K:
        configs.append((
            f"WS ovl product-only k={top_k}",
            WeightedSimilarity(
                w_product_metadata=0,
                w_product_interactions=1,
                w_customer_metadata=0,
                w_customer_interactions=0,
                metric="overlap",
                top_k=top_k,
            ),
        ))

    # ---- BipartiteGraphRec: 3-hop count, extend top_k ----
    for top_k in GRAPH_3HOP_COUNT_K:
        configs.append((
            f"Graph 3hop count k={top_k}",
            BipartiteGraphRec(
                hop_weights={3: 1.0},
                aggregation="count",
                top_k=top_k,
            ),
        ))

    # ---- BipartiteGraphRec: 3-hop multiply (reference only) ----
    for top_k in GRAPH_3HOP_MULT_K:
        configs.append((
            f"Graph 3hop multiply k={top_k}",
            BipartiteGraphRec(
                hop_weights={3: 1.0},
                aggregation="multiply",
                top_k=top_k,
            ),
        ))

    # ---- BipartiteGraphRec: lighter multi-hop (5-hop down-weighted vs prior sweep) ----
    for w5 in [0.05, 0.1]:
        configs.append((
            f"Graph 3+5 count w5={w5} k=100",
            BipartiteGraphRec(
                hop_weights={3: 1.0, 5: w5},
                aggregation="count",
                top_k=100,
            ),
        ))
    configs.append((
        "Graph 3+5+7 count k=100",
        BipartiteGraphRec(
            hop_weights={3: 1.0, 5: 0.15, 7: 0.05},
            aggregation="count",
            top_k=100,
        ),
    ))

    # ---- BipartiteGraphRec: multi-hop with each hop row-normalised ----
    for w5 in [0.1, 0.3]:
        configs.append((
            f"Graph 3+5 count w5={w5} k=150 rownorm",
            BipartiteGraphRec(
                hop_weights={3: 1.0, 5: w5},
                aggregation="count",
                top_k=150,
                hop_normalization="row",
            ),
        ))
    configs.append((
        "Graph 3+5+7 count k=150 rownorm",
        BipartiteGraphRec(
            hop_weights={3: 1.0, 5: 0.2, 7: 0.1},
            aggregation="count",
            top_k=150,
            hop_normalization="row",
        ),
    ))

    # ---- BipartiteGraphRec: penalties & groups (aligned with strong top_k) ----
    for pen in [0.4, 0.5, 0.6]:
        configs.append((
            f"Graph 3hop cnt k=150 pen={pen}",
            BipartiteGraphRec(
                hop_weights={3: 1.0},
                aggregation="count",
                top_k=150,
                customer_penalties={"country": pen},
            ),
        ))

    configs.append((
        "Graph 3hop cnt k=150 grp",
        BipartiteGraphRec(
            hop_weights={3: 1.0},
            aggregation="count",
            top_k=150,
            customer_group_cols=["country"],
        ),
    ))

    # ---- ItemKNNRec: neighbourhood size x asymmetric cosine ----
    for top_k in ITEM_KNN_K:
        for alpha in ITEM_KNN_ALPHA:
            configs.append((
                f"ItemKNN k={top_k} a={alpha}",
                ItemKNNRec(top_k=top_k, alpha=alpha),
            ))
    configs.append(("ItemKNN k=25 a=0.5 one-sided", ItemKNNRec(top_k=25, symmetrize=False)))
    configs.append(("ItemKNN k=25 a=0.5 shrink=10", ItemKNNRec(top_k=25, shrink=10)))
    configs.append(("ItemKNN k=25 a=0.5 log basket", ItemKNNRec(top_k=25, basket="log")))

    # ---- EASERec: regularisation ----
    for l2 in EASE_L2:
        configs.append((f"EASE l2={l2}", EASERec(l2=l2)))

    # ---- RP3BetaRec: walk sharpening x popularity penalty ----
    for alpha in RP3_ALPHA:
        for beta in RP3_BETA:
            configs.append((
                f"RP3beta a={alpha} b={beta} k=100",
                RP3BetaRec(alpha=alpha, beta=beta, top_k=100),
            ))
    for top_k in [50, 200]:
        configs.append((
            f"RP3beta a=1.0 b=0.3 k={top_k}",
            RP3BetaRec(alpha=1.0, beta=0.3, top_k=top_k),
        ))

    # ---- BeliefPropagationRec (slow: opt in with --include-bp) ----
    # 500-item subgraphs match 4000-item ones in accuracy at ~1/8 the cost.
    if include_bp:
        configs.append((
            "BP original model nodes=500",
            BeliefPropagationRec(top_k=100, max_subgraph_nodes=500),
        ))
        for beta in [0.1, 0.3]:
            configs.append((
                f"BP positive cosine prior=0.01 b={beta} nodes=500",
                BeliefPropagationRec(
                    top_k=100, beta=beta, max_subgraph_nodes=500,
                    coupling="cosine", evidence=1.0, prior=0.01,
                    potential="positive", score="logodds",
                ),
            ))

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4,
                        help="configs evaluated in parallel (default 4)")
    parser.add_argument("--include-bp", action="store_true",
                        help="also sweep BeliefPropagationRec (adds several minutes)")
    args = parser.parse_args()

    print("Loading dataset …")
    data = load_online_retail()

    configs = build_configs(include_bp=args.include_bp)
    print(f"\nRunning {len(configs)} configs (K={K}, jobs={args.jobs}) …\n")

    val, test = select_and_test(
        configs, data, k=K, output_dir=RESULTS_DIR, n_jobs=args.jobs,
    )

    cols = [c for c in test.columns if c not in ("params_json",)]
    print(f"\n{'=' * 80}\nTest results (best validation config per model class)\n")
    print(test[cols].to_string(index=False))

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("\nInstall matplotlib to generate the summary plot: pip install matplotlib")
        return

    plot_path = plot_best_per_family(
        test,
        output_path=f"{RESULTS_DIR}/best_per_family.png",
        title=f"Test NDCG@{K}: best validation config per model",
    )
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
