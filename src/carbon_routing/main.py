from __future__ import annotations
from benchmark.  runner import run_benchmark
import random
import networkx as nx
import sys
from pathlib import Path
if __package__ is None:
    # Path(__file__).parent -> carbon_routing dir; .parents[1] -> src dir
    src_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_dir))

from carbon_routing.config import RunConfig
from carbon_routing.topology.generator import generate_as_topology
from carbon_routing.topology.stats import topology_summary
from carbon_routing.ci.synthetic import SyntheticCIProvider

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


def run_steps_1_to_4(cfg: RunConfig) -> None:
    # -------------------------
    # Step 1: Topology + CI
    # -------------------------
    g = generate_as_topology(cfg.topology)
    summary = topology_summary(g)

    print("=== Step 1: Topology summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    ci = SyntheticCIProvider(n_as=cfg.topology.n_as, cfg=cfg.ci)

    t0 = 0
    t1 = 1 if cfg.ci.horizon_hours > 1 else 0

    all_ci_t0 = [ci.get_ci(asn, t0) for asn in range(cfg.topology.n_as)]
    all_ci_t1 = [ci.get_ci(asn, t1) for asn in range(cfg.topology.n_as)]

    def stats(xs):
        return min(xs), sum(xs) / len(xs), max(xs)

    mn0, av0, mx0 = stats(all_ci_t0)
    mn1, av1, mx1 = stats(all_ci_t1)

    print("\n=== Step 1: Synthetic CI sanity check ===")
    print(f"CI[t={t0}] min/mean/max: {mn0:.2f} / {av0:.2f} / {mx0:.2f} (gCO2/kWh)")
    print(f"CI[t={t1}] min/mean/max: {mn1:.2f} / {av1:.2f} / {mx1:.2f} (gCO2/kWh)")

    sample_as = [0, cfg.topology.n_as // 2, cfg.topology.n_as - 1]
    print("\n=== Step 1: Sample AS CI series (first 6 hours) ===")
    for asn in sample_as:
        region = ci.get_region(asn)
        series6 = list(ci.get_series(asn))[:6]
        print(f"AS{asn:04d} (region={region}) CI[0..5] = {[round(x,2) for x in series6]}")

    # -------------------------
    # Step 2: Path cost sanity
    # -------------------------
    print("\n=== Step 2: Path cost sanity checks ===")

    rng = random.Random(cfg.topology.seed)
    checks = 3
    done = 0

    while done < checks:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        path = nx.shortest_path(g, src, dst)
        cost_t0 = compute_path_cost(g, ci, path, hour=t0)
        cost_t1 = compute_path_cost(g, ci, path, hour=t1)

        print(f"\nPath AS{src:04d} -> AS{dst:04d}: length={len(path)-1} hops")
        print(f"  path: {path[:8]}{'...' if len(path) > 8 else ''}")
        print(f"  t={t0}: carbon={cost_t0.carbon:.2f}, latency_ms={cost_t0.latency_ms:.2f}")
        print(f"  t={t1}: carbon={cost_t1.carbon:.2f}, latency_ms={cost_t1.latency_ms:.2f}")

        # latency should not change across time
        if abs(cost_t0.latency_ms - cost_t1.latency_ms) > 1e-9:
            raise RuntimeError("Latency changed across time steps — bug in latency calculation.")

        done += 1

    # -------------------------
    # Step 3: Algorithm sanity
    # -------------------------
    print("\n=== Step 3: Algorithm selection checks (Baseline vs Carbon vs Weighted) ===")
    t = 0
    alpha = 0.7

    rng = random.Random(cfg.topology.seed)
    checks = 5
    done = 0

    while done < checks:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        base = select_baseline_latency_sp(g, src, dst)
        carb = select_carbon_sp(g, ci, src, dst, hour=t)
        wtd = select_weighted_sp(g, ci, src, dst, hour=t, alpha=alpha)

        base_cost = compute_path_cost(g, ci, base.path, hour=t)
        carb_cost = compute_path_cost(g, ci, carb.path, hour=t)
        wtd_cost = compute_path_cost(g, ci, wtd.path, hour=t)

        print(f"\nPair AS{src:04d} -> AS{dst:04d}")
        print(f"  Baseline: hops={base_cost.hops}, latency={base_cost.latency_ms:.2f}, carbon={base_cost.carbon:.2f}")
        print(f"  Carbon  : hops={carb_cost.hops}, latency={carb_cost.latency_ms:.2f}, carbon={carb_cost.carbon:.2f}")
        print(f"  Weighted(alpha={alpha}): hops={wtd_cost.hops}, latency={wtd_cost.latency_ms:.2f}, carbon={wtd_cost.carbon:.2f}")

        # Typically expected (not guaranteed): carbon path <= baseline carbon
        if carb_cost.carbon - base_cost.carbon > 1e-6:
            print("  [WARN] Carbon path has higher carbon than baseline (possible but should be rare).")

        done += 1

    # -------------------------
    # Step 4: K-shortest candidates
    # -------------------------
    print("\n=== Step 4: K-shortest candidate paths + selection ===")
    t = 0
    k = 8
    alpha = 0.7
    stretch = 1.15

    rng = random.Random(cfg.topology.seed)
    checks = 3
    done = 0

    while done < checks:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        cands = k_shortest_paths_latency(g, src, dst, k=k)
        base_lat = path_latency_ms(g, cands[0])
        bound = base_lat * stretch

        sel_carbon = select_min_carbon(g, ci, cands, hour=t)
        sel_weight = select_weighted(g, ci, cands, hour=t, alpha=alpha)
        sel_bound = select_min_carbon_with_latency_bound(g, ci, cands, hour=t, max_latency_ms=bound)

        print(f"\nPair AS{src:04d} -> AS{dst:04d}, candidates={len(cands)}")
        print(f"  baseline latency (best-lat candidate): {base_lat:.2f} ms")
        print(f"  latency bound ({stretch*100:.0f}%): {bound:.2f} ms")

        print("  Top-3 candidates (latency, carbon):")
        for i, p in enumerate(cands[:3]):
            pc = compute_path_cost(g, ci, p, hour=t)
            print(f"    {i}: lat={pc.latency_ms:.2f}, carbon={pc.carbon:.2f}, hops={pc.hops}")

        pc_c = compute_path_cost(g, ci, sel_carbon.chosen_path, hour=t)
        pc_w = compute_path_cost(g, ci, sel_weight.chosen_path, hour=t)
        pc_b = compute_path_cost(g, ci, sel_bound.chosen_path, hour=t)

        print(f"  Select min-carbon: lat={pc_c.latency_ms:.2f}, carbon={pc_c.carbon:.2f}")
        print(f"  Select weighted  : lat={pc_w.latency_ms:.2f}, carbon={pc_w.carbon:.2f}")
        print(f"  Select bounded  : lat={pc_b.latency_ms:.2f}, carbon={pc_b.carbon:.2f} ({sel_bound.reason})")

        done += 1
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


def main() -> None:
    cfg = RunConfig()  # n_as=120 by default
    run_steps_1_to_4(cfg)


if __name__ == "__main__":
    main()
