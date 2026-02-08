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
from .algorithms import cate as cate_mod
from .device import generate_router_params
from .algorithms.ospf_metrics import OspfMetricRouting, build_table1_metric_specs
from .benchmark.paper_runner import run_paper_benchmark
from .traffic import generate_synthetic_tm
from .algorithms.cate import run_cate
from .benchmark.cate_time_runner import run_cate_over_day
from .algorithms.ce import run_ce
from .metrics.utilization import compute_link_utilization
from .benchmark.all_runner import run_all
from .benchmark.registry import (
    BaselineRunner, CiroCoreRunner, LowCarbBGPRunner,
    OspfRunner, CRunner, CIncDRunner, CERunner
)
from carbon_routing.metrics.emissions_proxy import mean_emissions_proxy_for_paths
import random
from carbon_routing.metrics.emissions_proxy import mean_emissions_proxy_for_paths




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

    # ---------------------------------------------------------------------------
    # step 11: CATE algorithm module (not shown here, see algorithms/cate.py)
    # ---------------------------------------------------------------------------
    print("\n=== Step 11: CATE (centralized TE + link shutdown) ===")

    # router_params already created in Step 10; if not, create here:
    # router_params = generate_router_params(cfg.topology.n_as, seed=cfg.topology.seed)

    tm = generate_synthetic_tm(
        n_nodes=cfg.topology.n_as,
        n_demands=600,
        demand_mbps_range=(50.0, 500.0),
        seed=cfg.topology.seed,
    )

    # Scale TM down to a target max utilization (paper-style calibration).
    default_capacity_mbps = 2000.0
    target_max_util = 0.8
    scale_graph = g.copy()
    cate_mod._ensure_edge_capacity(scale_graph, default_capacity_mbps=default_capacity_mbps)
    _, _, dir_load, feas = cate_mod._route_tm_and_intensities(
        scale_graph, ci, router_params, tm, hour=0
    )
    if not feas:
        raise RuntimeError("Initial TM is not routable on the starting graph.")

    # Compute true max utilization (don't early-return like _capacity_ok).
    max_util = 0.0
    for (a, b), load in dir_load.items():
        if not scale_graph.has_edge(a, b):
            raise RuntimeError("TM routing used a non-existent edge.")
        cap = float(scale_graph[a][b].get("capacity_mbps", 0.0))
        if cap <= 0:
            raise RuntimeError("Edge capacity must be positive for CATE.")
        util = float(load) / cap
        if util > max_util:
            max_util = util

    if max_util > target_max_util and max_util > 0:
        scale = target_max_util / max_util
        tm = {k: v * scale for k, v in tm.items()}
        print(
            f"Scaled TM by {scale:.3f} to target max util {target_max_util:.2f} "
            f"(was {max_util:.3f})."
        )

    cate_res = run_cate(
        g=g,
        ci=ci,
        router_params=router_params,
        tm=tm,
        hour=0,
        default_capacity_mbps=default_capacity_mbps,
        port_beta_kw_per_link=0.005,
        max_iterations=g.number_of_edges(),
    )


    print(f"edges initial: {cate_res.edges_initial}")
    print(f"edges final  : {cate_res.edges_final}")
    print(f"links disabled: {cate_res.links_disabled}")
    print(f"connected final: {cate_res.connected_final}")
    print(f"max link utilization: {cate_res.max_link_utilization:.3f}")

    print("\nEmissions breakdown (paper-style proxy):")
    print(f"  initial dynamic_node: {cate_res.emissions_initial.dynamic_node:.6f}")
    print(f"  initial idle_ports  : {cate_res.emissions_initial.idle_ports:.6f}")
    print(f"  initial total       : {cate_res.emissions_initial.total:.6f}")

    print(f"  final dynamic_node  : {cate_res.emissions_final.dynamic_node:.6f}")
    print(f"  final idle_ports    : {cate_res.emissions_final.idle_ports:.6f}")
    print(f"  final total         : {cate_res.emissions_final.total:.6f}")

    print(f"\nTotal reduction vs initial: {cate_res.emissions_reduction_pct:.2f}%")
    print(f"Accepted removals: {len(cate_res.accepted_removals)} (show first 10) -> {cate_res.accepted_removals[:10]}")
    print(f"stop_reason: {cate_res.stop_reason}")
    print(f"history (last 5): {cate_res.history[-5:]}")
    # ---------------------------------------------------------------------------
    # Step 12A: 24h CATE evaluation (time-intensive)
    # Set RUN_STEP12 = True when you want to re-enable this section.
    # ---------------------------------------------------------------------------
    RUN_STEP12 = False
    if RUN_STEP12:
        print("\n=== Step 12A: 24h CATE evaluation ===")

        day = run_cate_over_day(
            base_graph=g,
            ci=ci,
            router_params=router_params,
            tm=tm,  # reuse the same TM you generated for Step 11 (important!)
            hours=24,
            default_capacity_mbps=2000.0,
            port_beta_kw_per_link=0.005,
            max_iterations=1000,
        )

        # print compact table-like output
        for r in day.rows[:5]:
            print(f"h={r.hour:02d} edges={r.edges_final:3d} disabled={r.links_disabled:3d} "
                f"util={r.max_util:.3f} total={r.total:.2f} stop={r.stop_reason}")

        print("...")

        for r in day.rows[-3:]:
            print(f"h={r.hour:02d} edges={r.edges_final:3d} disabled={r.links_disabled:3d} "
                f"util={r.max_util:.3f} total={r.total:.2f} stop={r.stop_reason}")

        print("\nReroute rate % (h->h+1) first 5:", [round(x,2) for x in day.reroute_rate_pct[:5]])
        print("Reroute rate % (h->h+1) last 5 :", [round(x,2) for x in day.reroute_rate_pct[-5:]])
        print("Avg reroute rate % over day     :", round(sum(day.reroute_rate_pct)/len(day.reroute_rate_pct), 2))

        print("\nTopology Jaccard (edge-set) first 5:", [round(x,3) for x in day.topo_change_jaccard[:5]])
        print("Topology Jaccard (edge-set) last 5 :", [round(x,3) for x in day.topo_change_jaccard[-5:]])
        print("Avg topology Jaccard over day      :", round(sum(day.topo_change_jaccard)/len(day.topo_change_jaccard), 3))

    # ---------------------------------------------------------------------------
    # Step 12B: CE algorithm module (not shown here, see algorithms/ce.py)
    # ---------------------------------------------------------------------------
    print("\n=== Step 12B: CE (utilization-aware C+IncD) over 24h ===")

    # 1) Build a fixed set of src->dst pairs (same as you benchmark elsewhere)
    # Sample source/destination pairs (repeatable).
    n_pairs = 200
    rng = random.Random(cfg.topology.seed)
    pairs = []
    while len(pairs) < n_pairs:
        s = rng.randrange(0, g.number_of_nodes())
        d = rng.randrange(0, g.number_of_nodes())
        if s != d:
            pairs.append((s, d))

    # 2) Helper: compute directed link loads from paths assuming unit demand per pair
    def _dir_load_from_unit_pairs(paths: dict[tuple[int, int], list[int]]) -> dict[tuple[int, int], float]:
        load: dict[tuple[int, int], float] = {}
        for _sd, p in paths.items():
            for a, b in zip(p[:-1], p[1:]):
                load[(a, b)] = load.get((a, b), 0.0) + 1.0
        return load

    # 3) Run over day: hour t uses utilization from hour t-1
    prev_util_undir: dict[tuple[int, int], float] = {}
    prev_paths: dict[tuple[int, int], list[int]] | None = None

    rows = []
    reroute_rates = []

    ce_ctx = AlgoContext(g=g, ci=ci, router_params=router_params)
    for h in range(24):
        # Run CE at hour h
        ce_res = run_ce(
            ctx=ce_ctx,
            pairs=pairs,
            hour=h,
            prev_util_undir=prev_util_undir,
            gamma=cfg.ce.gamma if hasattr(cfg, "ce") else 1.0,
        )
        paths = ce_res.paths

        # Compute mean carbon + latency using your existing helpers
        carb_vals = []
        lat_vals = []
        for (s, d), p in paths.items():
            cost = compute_path_cost(g, ci, p, hour=h)
            carb_vals.append(cost.carbon)
            lat_vals.append(cost.latency_ms)

        mean_c = sum(carb_vals) / len(carb_vals)
        mean_l = sum(lat_vals) / len(lat_vals)

        rr = 0.0
        if prev_paths is not None:
            changed = sum(1 for k in paths if paths[k] != prev_paths.get(k))
            rr = 100.0 * changed / len(paths)
        reroute_rates.append(rr)

        # Update utilization for next hour (based on unit-demand flows)
        dir_load = _dir_load_from_unit_pairs(paths)
        prev_util_undir = compute_link_utilization(g, dir_load)

        rows.append((h, mean_c, mean_l, rr))
        prev_paths = paths

    # Print summary
    for h, mc, ml, rr in rows[:5]:
        print(f"h={h:02d} mean_carbon={mc:.2f} mean_latency={ml:.2f} reroute%={rr:.2f}")
    print("...")
    for h, mc, ml, rr in rows[-3:]:
        print(f"h={h:02d} mean_carbon={mc:.2f} mean_latency={ml:.2f} reroute%={rr:.2f}")

    print("\nAvg reroute rate % over day:", round(sum(reroute_rates[1:]) / 23, 2))
    print("Mean carbon (h=0..2):", [round(rows[i][1], 2) for i in range(3)])
    print("Mean latency (h=0..2):", [round(rows[i][2], 2) for i in range(3)])

    # ----------------------------------------------------------------------------
    # Step 13 Split: Implement 13A
    # ----------------------------------------------------------------------------
    
    print("\n=== Step 13A: Unified tables (Inter-domain family) ===")

    runners_A = [
        BaselineRunner(name="baseline_latency", g=g, pairs=pairs),
        CiroCoreRunner(name="ciro_core", g=g, ci=ci, router_params=router_params, pairs=pairs),
        LowCarbBGPRunner(name="lowcarb_bgp", g=g, ci=ci, router_params=router_params, pairs=pairs),
    ]

    resA = run_all(g=g, ci=ci, pairs=pairs, runners=runners_A, hours=24)
    _print_step13_tables("=== Step 13A Results ===", resA)


    print("\n=== Step 13B: Unified tables (OSPF-metric family) ===")

    runners_B = [
        OspfRunner(name="OSPF", g=g, ci=ci, router_params=router_params, pairs=pairs),
        CRunner(name="C", g=g, ci=ci, router_params=router_params, pairs=pairs),
        CIncDRunner(name="C+IncD", g=g, ci=ci, router_params=router_params, pairs=pairs),
        CERunner(name="CE", g=g, ci=ci, router_params=router_params, pairs=pairs, gamma=1.0),
    ]

    resB = run_all(g=g, ci=ci, pairs=pairs, runners=runners_B, hours=24)
    _print_step13_tables("=== Step 13B Results ===", resB)
    # ----------------------------------------------------------------------------
    
    # Get actual paths and create traffic that matches their length
    paths_ospf_t0 = resB.hour0_paths_by_name["OSPF"]
    actual_pairs = list(paths_ospf_t0.keys())
    
    # Create traffic with matching length
    rng = random.Random(7)
    traffic = [rng.uniform(0.5, 1.5) for _ in actual_pairs]  # stable

    # Create incd_lambda_by_node from router_params
    incd_lambda_by_node = {node: router_params[node].incd_w_per_mbps for node in router_params}

    ci_t0 = {n: ci.get_ci(n, 0) for n in g.nodes}
    
    # snaity check
    # Sanity: emissions must differ if paths differ
    # Get the 3rd pair from the paths dictionary
    paths_os = resB.hour0_paths_by_name["OSPF"]
    paths_c = resB.hour0_paths_by_name["C"]
    third_pair = list(paths_os.keys())[2] if len(paths_os) > 2 else None
    
    if third_pair is not None:
        p_os = paths_os[third_pair]
        p_c = paths_c[third_pair]
        if p_os != p_c:
            # Find index of third pair in actual_pairs
            third_idx = actual_pairs.index(third_pair)
            e_os = mean_emissions_proxy_for_paths([p_os], ci_t0, [traffic[third_idx]], incd_lambda_by_node)
            e_c  = mean_emissions_proxy_for_paths([p_c ], ci_t0, [traffic[third_idx]], incd_lambda_by_node)
            print(f"[DEBUG] single-pair emissions OSPF={e_os:.6f} C={e_c:.6f} (should differ)")

    print("\n=== Step 13B.1: Emissions proxy columns (paper metric) ===")

    ospf_paths_list_t0 = ordered_path_list(resB.hour0_paths_by_name["OSPF"], actual_pairs)
    e_ospf_0 = mean_emissions_proxy_for_paths(ospf_paths_list_t0, ci_t0, traffic, incd_lambda_by_node)

    for name, paths_by_pair in resB.hour0_paths_by_name.items():
        paths_list = ordered_path_list(paths_by_pair, actual_pairs)
        e0 = mean_emissions_proxy_for_paths(paths_list, ci_t0, traffic, incd_lambda_by_node)
        dE = 100.0 * (e_ospf_0 - e0) / max(e_ospf_0, 1e-9)
        print(f"{name:10s} em0={e0:.2f} ΔE0%={dE:.2f}")
    e_ospf_sum = 0.0
    e_by_algo_sum = {name: 0.0 for name in resB.day_paths_by_name.keys()}

    for t in range(24):
        ci_tt = {n: ci.get_ci(n, t) for n in g.nodes}

        ospf_paths_list = ordered_path_list(resB.day_paths_by_name["OSPF"][t], actual_pairs)
        e_ospf_sum += mean_emissions_proxy_for_paths(ospf_paths_list, ci_tt, traffic, incd_lambda_by_node)

        for name in e_by_algo_sum:
            paths_list = ordered_path_list(resB.day_paths_by_name[name][t], actual_pairs)
            e_by_algo_sum[name] += mean_emissions_proxy_for_paths(paths_list, ci_tt, traffic, incd_lambda_by_node)

    e_ospf_24 = e_ospf_sum / 24.0
    for name, s in e_by_algo_sum.items():
        e24 = s / 24.0
        dE24 = 100.0 * (e_ospf_24 - e24) / max(e_ospf_24, 1e-9)
        print(f"{name:10s} em24={e24:.2f} ΔE24%={dE24:.2f}")
    print("[DEBUG13B1]", name, "p2:", paths_list[2])
    print("[DEBUG13B1] p2 OSPF:", ospf_paths_list_t0[2])
    print("[DEBUG13B1] p2 C   :", ordered_path_list(resB.hour0_paths_by_name["C"], actual_pairs)[2])



def ordered_path_list(paths_by_pair: dict, pair_order: list):
    return [paths_by_pair[pair] for pair in pair_order]


def _print_step13_tables(title: str, res) -> None:
    print(f"\n{title}\n")

    print("Hour-0 summary:")
    for row in res.hour0_vs_base:
        print(
            f"{row['name']:12s} carbon={row['mean_carbon']:.2f} "
            f"lat={row['mean_latency_ms']:.2f} hops={row['mean_hops']:.2f} "
            f"ΔC%={row['mean_carbon_reduction_vs_base(%)']:.2f} "
            f"ΔL%={row['mean_latency_increase_vs_base(%)']:.2f} "
            f"chg%={row['pct_paths_changed_vs_base(%)']:.2f}"
        )

    print("\n24h summary:")
    for row in res.day_vs_base:
        print(
            f"{row['name']:12s} carbon24={row['mean_carbon_24h']:.2f} "
            f"lat24={row['mean_latency_24h']:.2f} hops24={row['mean_hops_24h']:.2f} "
            f"reroute%={row['avg_reroute_rate_pct']:.2f} "
            f"ΔC24%={row['carbon_reduction_vs_base_24h(%)']:.2f} "
            f"ΔL24%={row['latency_increase_vs_base_24h(%)']:.2f}"
        )

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
