from __future__ import annotations
import random
import networkx as nx
from .config import RunConfig
from .topology.generator import generate_as_topology
from .topology.stats import topology_summary, sample_path_exists
from .ci.synthetic import SyntheticCIProvider
from .metrics.path_cost import compute_path_cost
from .algorithms import BaselineLatency, AlgoContext
from .benchmark.runner import run_benchmark
from .benchmark.time_runner import run_time_benchmark
from .algorithms import BaselineLatency, CIRoCore, AlgoContext
from .benchmark.runner import run_benchmark
from .benchmark.time_runner import run_time_benchmark
from .algorithms import BaselineLatency, LowCarbBGP, AlgoContext
from .benchmark.runner import run_benchmark
from .benchmark.time_runner import run_time_benchmark
from .device import generate_router_params
from .algorithms.ospf_metrics import OspfMetricRouting, build_table1_metric_specs
from .benchmark.paper_runner import run_paper_benchmark



def run_step1(cfg: RunConfig, g: nx.Graph, ci: SyntheticCIProvider) -> None:
    summary = topology_summary(g)

    print("\n=== Step 1: Topology summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    t0 = 0
    t1 = 1 if cfg.ci.horizon_hours > 1 else 0

    all_ci_t0 = [ci.get_ci(asn, t0) for asn in range(cfg.topology.n_as)]
    all_ci_t1 = [ci.get_ci(asn, t1) for asn in range(cfg.topology.n_as)]

    def stats(xs):
        return min(xs), sum(xs) / len(xs), max(xs)

    print("\n=== Step 1: Synthetic CI sanity check ===")
    print(f"CI[t=0] min/mean/max: {stats(all_ci_t0)}")
    print(f"CI[t=1] min/mean/max: {stats(all_ci_t1)}")

    sample_as = [0, cfg.topology.n_as // 2, cfg.topology.n_as - 1]
    print("\n=== Step 1: Sample AS CI series (first 6 hours) ===")
    for asn in sample_as:
        region = ci.get_region(asn)
        series6 = list(ci.get_series(asn))[:6]
        print(f"AS{asn:04d} (region={region}) CI[0..5] = {[round(x, 2) for x in series6]}")

    print("\n=== Step 1: Sample path existence checks ===")
    rng = random.Random(cfg.topology.seed)
    pairs_checked = 0
    while pairs_checked < 3:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue
        ok = sample_path_exists(g, src, dst)
        print(f"path exists AS{src:04d} -> AS{dst:04d}: {ok}")
        pairs_checked += 1


def run_step2(cfg: RunConfig, g: nx.Graph, ci: SyntheticCIProvider) -> None:
    print("\n=== Step 2: Path cost sanity checks ===")
    rng = random.Random(cfg.topology.seed)

    t0 = 0
    t1 = 1 if cfg.ci.horizon_hours > 1 else 0

    checks = 3
    done = 0
    while done < checks:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue

        # Use hop-based path only for sanity checks
        path = nx.shortest_path(g, src, dst)

        cost_t0 = compute_path_cost(g, ci, path, hour=t0)
        cost_t1 = compute_path_cost(g, ci, path, hour=t1)

        print(f"\nPath AS{src:04d} -> AS{dst:04d}")
        print(f"  t={t0} carbon={cost_t0.carbon:.2f}, latency={cost_t0.latency_ms:.2f}")
        print(f"  t={t1} carbon={cost_t1.carbon:.2f}, latency={cost_t1.latency_ms:.2f}")

        # Invariant: latency must not change across time
        if abs(cost_t0.latency_ms - cost_t1.latency_ms) > 1e-9:
            raise RuntimeError("Latency changed across time steps — bug in latency calculation.")

        done += 1


def run_step7_baseline_only(cfg: RunConfig, g: nx.Graph, ci: SyntheticCIProvider) -> None:
    print("\n=== Step 7: Baseline-only via algorithm module ===")

    ctx = AlgoContext(
        k_paths=8,
        alpha=0.7,
        stretch=1.15,
    )
    algos = [BaselineLatency()]

    # Batch summary (hour=0)
    _, summaries = run_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        baseline_name="baseline_latency",
        ctx=ctx,
        n_demands=200,
        hour=0,
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

    # Time-series check (24h)
    ts = run_time_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        ctx=ctx,
        n_demands=200,
        horizon_hours=cfg.ci.horizon_hours,
        seed=cfg.topology.seed,
    )

    b = ts["baseline_latency"]
    avg_reroute = sum(b.reroute_rate_by_hour[1:]) / max(1, len(b.reroute_rate_by_hour) - 1)

    print("\nBaseline time-series quick check:")
    print(f"  mean_latency (t=0..2): {[round(x, 2) for x in b.mean_latency_by_hour[:3]]}")
    print(f"  reroute_rate% (t=1..3): {[round(x, 2) for x in b.reroute_rate_by_hour[1:4]]}")
    print(f"  avg_reroute_rate%: {avg_reroute:.2f}")

# Step 7: baseline isolated as its own algorithm module
# Step 8: CIRoCore algorithm module (not shown here, see algorithms/cir_o_core.py)
    print("\n=== Step 8: Baseline vs CIRo-Core (forecast-aware) ===")
    ctx = AlgoContext(k_paths=8, forecast_window_hours=4)

    algos = [BaselineLatency(), CIRoCore()]

    _, summaries = run_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        baseline_name="baseline_latency",
        ctx=ctx,
        n_demands=200,
        hour=0,
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

    ts = run_time_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        ctx=ctx,
        n_demands=200,
        horizon_hours=cfg.ci.horizon_hours,
        seed=cfg.topology.seed,
    )

    cc = ts["ciro_core"]
    avg_reroute = sum(cc.reroute_rate_by_hour[1:]) / max(1, len(cc.reroute_rate_by_hour)-1)
    print("\nCIRo-Core time-series quick check:")
    print(f"  mean_carbon (t=0..2): {[round(x,2) for x in cc.mean_carbon_by_hour[:3]]}")
    print(f"  mean_latency (t=0..2): {[round(x,2) for x in cc.mean_latency_by_hour[:3]]}")
    print(f"  reroute_rate% (t=1..3): {[round(x,2) for x in cc.reroute_rate_by_hour[1:4]]}")
    print(f"  avg_reroute_rate% over day: {avg_reroute:.2f}")
# ---------------------------------------------------------------------------   
# step 9: Low-Carb BGP algorithm module (not shown here, see algorithms/lowcarb_bgp.py)
# ---------------------------------------------------------------------------
    print("\n=== Step 9: Baseline vs Low-Carb BGP (LCB) ===")
    ctx = AlgoContext(k_paths=8)  # LCB uses instantaneous CIM; no forecast window needed

    algos = [BaselineLatency(), LowCarbBGP()]

    _, summaries = run_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        baseline_name="baseline_latency",
        ctx=ctx,
        n_demands=200,
        hour=0,
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

    ts = run_time_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        ctx=ctx,
        n_demands=200,
        horizon_hours=cfg.ci.horizon_hours,
        seed=cfg.topology.seed,
    )

    lcb = ts["lowcarb_bgp"]
    avg_reroute = sum(lcb.reroute_rate_by_hour[1:]) / max(1, len(lcb.reroute_rate_by_hour)-1)
    print("\nLow-Carb BGP time-series quick check:")
    print(f"  mean_carbon (t=0..2): {[round(x,2) for x in lcb.mean_carbon_by_hour[:3]]}")
    print(f"  mean_latency (t=0..2): {[round(x,2) for x in lcb.mean_latency_by_hour[:3]]}")
    print(f"  reroute_rate% (t=1..3): {[round(x,2) for x in lcb.reroute_rate_by_hour[1:4]]}")
    print(f"  avg_reroute_rate% over day: {avg_reroute:.2f}")

# ---------------------------------------------------------------------------
# step 10: OSPF Metric Routing algorithm module (not shown here, see algorithms/ospf_metrics.py)
# ---------------------------------------------------------------------------
    print("\n=== Step 10: Paper Table-1 OSPF-metric routing (OSPF vs C vs C+IncD) ===")

    router_params = generate_router_params(cfg.topology.n_as, seed=cfg.topology.seed)
    specs = build_table1_metric_specs(router_params, ci)

    algos = [
        OspfMetricRouting(specs["OSPF"]),
        OspfMetricRouting(specs["C"]),
        OspfMetricRouting(specs["C+IncD"]),
    ]

    ctx = AlgoContext(k_paths=8)  # not used here, but keep consistent API

    _, summaries = run_benchmark(
        g=g,
        ci=ci,
        algorithms=algos,
        baseline_name="OSPF",
        ctx=ctx,
        n_demands=200,
        hour=0,
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
    

    #step 10.2
    print("\n=== Step 10.2: Paper metric evaluation (IncD emissions proxy) ===")

    # print lambda sanity (make sure max prints too)
    lams = [router_params[a].incd_w_per_mbps for a in router_params]
    print("IncD(lambda) sanity: min/mean/max =",
        round(min(lams), 6), round(sum(lams)/len(lams), 6), round(max(lams), 6))

    _, paper_summaries = run_paper_benchmark(
        g=g,
        ci=ci,
        router_params=router_params,
        algorithms=algos,          # same 3 algos: OSPF, C, C+IncD
        baseline_name="OSPF",
        ctx=ctx,
        n_demands=200,
        hour=0,
        demand_mbps=100.0,         # fixed synthetic demand for Step 10.2
        seed=cfg.topology.seed,
    )

    for name, s in paper_summaries.items():
        print(f"\nAlgorithm: {name}")
        print(f"  mean_emissions: {s.mean_emissions:.6f}")
        print(f"  mean_latency_ms: {s.mean_latency_ms:.2f}")
        print(f"  mean_hops: {s.mean_hops:.2f}")
        print(f"  mean_emissions_reduction_vs_base(%): {s.mean_emissions_reduction_pct_vs_base:.2f}")
        print(f"  mean_latency_increase_vs_base(%): {s.mean_latency_increase_pct_vs_base:.2f}")
        print(f"  pct_paths_changed_vs_base(%): {s.pct_paths_changed_vs_base:.2f}")




    

def main() -> None:
    cfg = RunConfig()

    # Build topology + CI provider once
    g = generate_as_topology(cfg.topology)
    ci = SyntheticCIProvider(n_as=cfg.topology.n_as, cfg=cfg.ci)

    # Steps
    run_step1(cfg, g, ci)
    run_step2(cfg, g, ci)

    # Step 7: baseline isolated as its own algorithm module
    run_step7_baseline_only(cfg, g, ci)


if __name__ == "__main__":
    main()
