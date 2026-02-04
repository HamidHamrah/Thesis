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

def run_step1_and_2(cfg: RunConfig) -> None:
    # 1) Generate topology
    g = generate_as_topology(cfg.topology)
    summary = topology_summary(g)

    print("=== Step 1: Topology summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # 2) Generate CI time series (synthetic)
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

    # ---- Step 2: Path cost sanity checks ----
    print("\n=== Step 2: Path cost sanity checks ===")

    rng = random.Random(cfg.topology.seed)
    checks = 3
    done = 0

    while done < checks:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue
        # shortest path by hop count for now (baseline path used only for cost testing)
        path = nx.shortest_path(g, src, dst)

        cost_t0 = compute_path_cost(g, ci, path, hour=t0)
        cost_t1 = compute_path_cost(g, ci, path, hour=t1)

        print(f"\nPath AS{src:04d} -> AS{dst:04d}: length={len(path)-1} hops")
        print(f"  path: {path[:8]}{'...' if len(path) > 8 else ''}")
        print(f"  t={t0}: carbon={cost_t0.carbon:.2f}, latency_ms={cost_t0.latency_ms:.2f}")
        print(f"  t={t1}: carbon={cost_t1.carbon:.2f}, latency_ms={cost_t1.latency_ms:.2f}")

        # Invariant checks:
        # - latency should NOT change with time (CI changes, latency doesn't)
        if abs(cost_t0.latency_ms - cost_t1.latency_ms) > 1e-9:
            raise RuntimeError("Latency changed across time steps — bug in latency calculation.")

        done += 1

def main():
    cfg = RunConfig()  # defaults: n_as=120
    run_step1_and_2(cfg)

if __name__ == "__main__":
    main()
