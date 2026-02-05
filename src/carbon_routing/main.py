from __future__ import annotations
import random
import networkx as nx

# -------------------------------
# Config & core components
# -------------------------------
from carbon_routing.config import RunConfig
from carbon_routing.topology.generator import generate_as_topology
from carbon_routing.topology.stats import topology_summary
from carbon_routing.ci.synthetic import SyntheticCIProvider

# -------------------------------
# Metrics & routing
# -------------------------------
from carbon_routing.metrics.path_cost import compute_path_cost
from carbon_routing.routing.shortest_path import (
    select_baseline_latency_sp,
    select_carbon_sp,
    select_weighted_sp,
)
from carbon_routing.routing.candidates import (
    k_shortest_paths_latency,
    path_latency_ms,
)
from carbon_routing.routing.select_from_candidates import (
    select_min_carbon,
    select_weighted,
    select_min_carbon_with_latency_bound,
)

# -------------------------------
# Benchmark runner
# -------------------------------
from carbon_routing.benchmark.runner import run_benchmark


def main() -> None:
    cfg = RunConfig()

    # ============================================================
    # Step 1: Topology + CI
    # ============================================================
    g = generate_as_topology(cfg.topology)
    summary = topology_summary(g)

    print("=== Step 1: Topology summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    ci = SyntheticCIProvider(
        n_as=cfg.topology.n_as,
        cfg=cfg.ci,
    )

    t0, t1 = 0, 1
    all_ci_t0 = [ci.get_ci(asn, t0) for asn in range(cfg.topology.n_as)]
    all_ci_t1 = [ci.get_ci(asn, t1) for asn in range(cfg.topology.n_as)]

    def stats(xs):
        return min(xs), sum(xs) / len(xs), max(xs)

    print("\n=== Step 1: Synthetic CI sanity check ===")
    print(f"CI[t=0] min/mean/max: {stats(all_ci_t0)}")
    print(f"CI[t=1] min/mean/max: {stats(all_ci_t1)}")

    print("\n=== Step 1: Sample AS CI series (first 6 hours) ===")
    for asn in [0, cfg.topology.n_as // 2, cfg.topology.n_as - 1]:
        region = ci.get_region(asn)
        series = list(ci.get_series(asn))[:6]
        print(f"AS{asn:04d} (region={region}) CI[0..5] = {[round(x,2) for x in series]}")

    # ============================================================
    # Step 2: Path cost sanity
    # ============================================================
    print("\n=== Step 2: Path cost sanity checks ===")
    rng = random.Random(cfg.topology.seed)

    for _ in range(3):
        src, dst = rng.randrange(0, cfg.topology.n_as), rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        path = nx.shortest_path(g, src, dst)
        cost0 = compute_path_cost(g, ci, path, hour=0)
        cost1 = compute_path_cost(g, ci, path, hour=1)

        print(f"\nPath AS{src:04d} -> AS{dst:04d}")
        print(f"  t=0 carbon={cost0.carbon:.2f}, latency={cost0.latency_ms:.2f}")
        print(f"  t=1 carbon={cost1.carbon:.2f}, latency={cost1.latency_ms:.2f}")

    # ============================================================
    # Step 3: Single-path algorithm comparison
    # ============================================================
    print("\n=== Step 3: Algorithm selection checks ===")
    alpha = 0.7

    for _ in range(5):
        src, dst = rng.randrange(0, cfg.topology.n_as), rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        base = select_baseline_latency_sp(g, src, dst)
        carb = select_carbon_sp(g, ci, src, dst, hour=0)
        wtd  = select_weighted_sp(g, ci, src, dst, hour=0, alpha=alpha)

        bc = compute_path_cost(g, ci, base.path, 0)
        cc = compute_path_cost(g, ci, carb.path, 0)
        wc = compute_path_cost(g, ci, wtd.path, 0)

        print(f"\nPair AS{src:04d} -> AS{dst:04d}")
        print(f"  Baseline: latency={bc.latency_ms:.2f}, carbon={bc.carbon:.2f}")
        print(f"  Carbon  : latency={cc.latency_ms:.2f}, carbon={cc.carbon:.2f}")
        print(f"  Weighted: latency={wc.latency_ms:.2f}, carbon={wc.carbon:.2f}")

    # ============================================================
    # Step 4: Candidate paths
    # ============================================================
    print("\n=== Step 4: K-shortest candidate paths + selection ===")
    k = 8
    stretch = 1.15

    for _ in range(3):
        src, dst = rng.randrange(0, cfg.topology.n_as), rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        cands = k_shortest_paths_latency(g, src, dst, k)
        base_lat = path_latency_ms(g, cands[0])
        bound = base_lat * stretch

        sel_c = select_min_carbon(g, ci, cands, hour=0)
        sel_w = select_weighted(g, ci, cands, hour=0, alpha=alpha)
        sel_b = select_min_carbon_with_latency_bound(g, ci, cands, hour=0, max_latency_ms=bound)

        print(f"\nPair AS{src:04d} -> AS{dst:04d}, candidates={len(cands)}")
        print(f"  baseline latency={base_lat:.2f}, bound={bound:.2f}")
        print(f"  chosen carbon={compute_path_cost(g, ci, sel_c.chosen_path, 0).carbon:.2f}")
        print(f"  chosen weighted={compute_path_cost(g, ci, sel_w.chosen_path, 0).carbon:.2f}")
        print(f"  chosen bounded={compute_path_cost(g, ci, sel_b.chosen_path, 0).carbon:.2f}")

    # ============================================================
    # Step 5: Batch benchmark
    # ============================================================
    print("\n=== Step 5: Batch benchmark summary ===")
    _, summaries = run_benchmark(
        g=g,
        ci=ci,
        n_demands=200,
        k=8,
        hour=0,
        alpha=0.7,
        stretch=1.15,
        seed=cfg.topology.seed,
    )

    for name, s in summaries.items():
        print(f"\nAlgorithm: {name}")
        print(f"  mean_carbon: {s.mean_carbon:.2f}")
        print(f"  mean_latency_ms: {s.mean_latency_ms:.2f}")
        print(f"  mean_hops: {s.mean_hops:.2f}")
        print(f"  mean_carbon_reduction_vs_base(%): {s.mean_carbon_reduction_pct_vs_base:.2f}")
        print(f"  mean_latency_increase_vs_base(%): {s.mean_latency_increase_pct_vs_base:.2f}")
        print(f"  pct_paths_changed_vs_base(%): {s.pct_paths_changed_vs_base:.2f}")


if __name__ == "__main__":
    main()
