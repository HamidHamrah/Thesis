from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import random

import networkx as nx

from ..ci.base import CIProvider
from ..algorithms.base import RoutingAlgorithm, AlgoContext
from ..device.models import RouterParams
from ..metrics.paper_emissions import compute_paper_emissions_cost

@dataclass(frozen=True)
class PaperDemandResult:
    src: int
    dst: int
    baseline_path: Sequence[int]
    chosen_path: Sequence[int]
    emissions: float
    latency_ms: float
    hops: int
    baseline_emissions: float
    baseline_latency_ms: float
    path_changed: bool

@dataclass(frozen=True)
class PaperSummary:
    mean_emissions: float
    mean_latency_ms: float
    mean_hops: float
    mean_emissions_reduction_pct_vs_base: float
    mean_latency_increase_pct_vs_base: float
    pct_paths_changed_vs_base: float

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

def run_paper_benchmark(
    g: nx.Graph,
    ci: CIProvider,
    router_params: dict[int, RouterParams],
    algorithms: List[RoutingAlgorithm],
    baseline_name: str,
    ctx: AlgoContext,
    n_demands: int = 200,
    hour: int = 0,
    demand_mbps: float = 100.0,
    seed: int = 42,
) -> Tuple[Dict[str, List[PaperDemandResult]], Dict[str, PaperSummary]]:
    rng = random.Random(seed)
    n_as = g.number_of_nodes()

    algomap = {a.name: a for a in algorithms}
    if baseline_name not in algomap:
        raise ValueError(f"baseline_name '{baseline_name}' not found in algorithms list")

    results: Dict[str, List[PaperDemandResult]] = {a.name: [] for a in algorithms}

    # fixed demands
    demands = []
    while len(demands) < n_demands:
        s = rng.randrange(0, n_as)
        d = rng.randrange(0, n_as)
        if s != d:
            demands.append((s, d))

    for (src, dst) in demands:
        baseline_algo = algomap[baseline_name]
        baseline_path = baseline_algo.select_path(g, ci, src, dst, hour, ctx)
        base_cost = compute_paper_emissions_cost(
            g=g, ci=ci, router_params=router_params,
            path=baseline_path, hour=hour, demand_mbps=demand_mbps
        )

        for algo in algorithms:
            chosen_path = algo.select_path(g, ci, src, dst, hour, ctx)
            pc = compute_paper_emissions_cost(
                g=g, ci=ci, router_params=router_params,
                path=chosen_path, hour=hour, demand_mbps=demand_mbps
            )

            results[algo.name].append(
                PaperDemandResult(
                    src=src,
                    dst=dst,
                    baseline_path=baseline_path,
                    chosen_path=chosen_path,
                    emissions=pc.emissions,
                    latency_ms=pc.latency_ms,
                    hops=pc.hops,
                    baseline_emissions=base_cost.emissions,
                    baseline_latency_ms=base_cost.latency_ms,
                    path_changed=(list(chosen_path) != list(baseline_path)),
                )
            )

    summaries: Dict[str, PaperSummary] = {}
    for tag, rows in results.items():
        em  = [r.emissions for r in rows]
        lat = [r.latency_ms for r in rows]
        hops = [float(r.hops) for r in rows]

        em_red = []
        lat_inc = []
        changed = 0

        for r in rows:
            if r.baseline_emissions > 0:
                em_red.append(100.0 * (r.baseline_emissions - r.emissions) / r.baseline_emissions)
            if r.baseline_latency_ms > 0:
                lat_inc.append(100.0 * (r.latency_ms - r.baseline_latency_ms) / r.baseline_latency_ms)
            if r.path_changed:
                changed += 1

        summaries[tag] = PaperSummary(
            mean_emissions=_mean(em),
            mean_latency_ms=_mean(lat),
            mean_hops=_mean(hops),
            mean_emissions_reduction_pct_vs_base=_mean(em_red),
            mean_latency_increase_pct_vs_base=_mean(lat_inc),
            pct_paths_changed_vs_base=(100.0 * changed / len(rows)) if rows else 0.0,
        )

    return results, summaries
