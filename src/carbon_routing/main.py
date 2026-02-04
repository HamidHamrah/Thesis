from __future__ import annotations

import random
import networkx as nx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root to path

from src.carbon_routing.config import RunConfig
from src.carbon_routing.topology.generator import generate_as_topology
from src.carbon_routing.topology.stats import topology_summary
from src.carbon_routing.ci.synthetic import SyntheticCIProvider
from src.carbon_routing.metrics.path_cost import compute_path_cost
from src.carbon_routing.routing.shortest_path import (
    select_baseline_latency_sp,
    select_carbon_sp,
    select_weighted_sp,
)


def run_step1(cfg: RunConfig, g: nx.Graph, ci: SyntheticCIProvider) -> None:
    summary = topology_summary(g)

    print("=== Step 1: Topology summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

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
        print(f"AS{asn:04d} (region={region}) CI[0..5] = {[round(x, 2) for x in series6]}")


def run_step2(cfg: RunConfig, g: nx.Graph, ci: SyntheticCIProvider) -> None:
    print("\n=== Step 2: Path cost sanity checks ===")
    t0 = 0
    t1 = 1 if cfg.ci.horizon_hours > 1 else 0

    rng = random.Random(cfg.topology.seed)
    checks = 3

    for _ in range(checks):
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        while dst == src:
            dst = rng.randrange(0, cfg.topology.n_as)

        path = nx.shortest_path(g, src, dst)  # hop-count shortest, for cost sanity only

        cost_t0 = compute_path_cost(g, ci, path, hour=t0)
        cost_t1 = compute_path_cost(g, ci, path, hour=t1)

        print(f"\nPath AS{src:04d} -> AS{dst:04d}: length={len(path)-1} hops")
        print(f"  path: {path[:8]}{'...' if len(path) > 8 else ''}")
        print(f"  t={t0}: carbon={cost_t0.carbon:.2f}, latency_ms={cost_t0.latency_ms:.2f}")
        print(f"  t={t1}: carbon={cost_t1.carbon:.2f}, latency_ms={cost_t1.latency_ms:.2f}")

        # Latency should not change with time
        if abs(cost_t0.latency_ms - cost_t1.latency_ms) > 1e-9:
            raise RuntimeError("Latency changed across time steps — bug in latency calculation.")


def run_step3(cfg: RunConfig, g: nx.Graph, ci: SyntheticCIProvider) -> None:
    print("\n=== Step 3: Algorithm selection checks (Baseline vs Carbon vs Weighted) ===")

    t = 0
    alpha = 0.7  # we will sweep later; for now a strong carbon preference
    rng = random.Random(cfg.topology.seed)
    checks = 5

    for _ in range(checks):
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        while dst == src:
            dst = rng.randrange(0, cfg.topology.n_as)

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

        # This should usually hold. If it doesn't, it can still be possible in edge cases,
        # but we flag it to inspect.
        if carb_cost.carbon > base_cost.carbon + 1e-6:
            print("  [WARN] Carbon path has higher carbon than baseline (inspect / rare).")


def main() -> None:
    cfg = RunConfig()

    # Build topology + CI once (shared across steps)
    g = generate_as_topology(cfg.topology)
    ci = SyntheticCIProvider(n_as=cfg.topology.n_as, cfg=cfg.ci)

    run_step1(cfg, g, ci)
    run_step2(cfg, g, ci)
    run_step3(cfg, g, ci)

    print("\n=== Done (Steps 1–3 completed) ===")


if __name__ == "__main__":
    main()
