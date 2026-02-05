from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import random

import networkx as nx

from ..ci.base import CIProvider
from ..metrics.path_cost import compute_path_cost
from ..algorithms.base import RoutingAlgorithm, AlgoContext

@dataclass(frozen=True)
class DemandResult:
    src: int
    dst: int
    baseline_path: Sequence[int]
    chosen_path: Sequence[int]
    carbon: float
    latency_ms: float
    hops: int
    baseline_carbon: float
    baseline_latency_ms: float
    path_changed: bool

@dataclass(frozen=True)
class Summary:
    mean_carbon: float
    mean_latency_ms: float
    mean_hops: float
    mean_carbon_reduction_pct_vs_base: float
    mean_latency_increase_pct_vs_base: float
    pct_paths_changed_vs_base: float

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

def run_benchmark(
    g: nx.Graph,
    ci: CIProvider,
    algorithms: List[RoutingAlgorithm],
    baseline_name: str,
    ctx: AlgoContext,
    n_demands: int = 200,
    hour: int = 0,
    seed: int = 42,
) -> Tuple[Dict[str, List[DemandResult]], Dict[str, Summary]]:
    rng = random.Random(seed)
    n_as = g.number_of_nodes()

    algomap = {a.name: a for a in algorithms}
    if baseline_name not in algomap:
        raise ValueError(f"baseline_name '{baseline_name}' not found in algorithms list")

    results: Dict[str, List[DemandResult]] = {a.name: [] for a in algorithms}

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
        base_cost = compute_path_cost(g, ci, baseline_path, hour)

        for algo in algorithms:
            chosen_path = algo.select_path(g, ci, src, dst, hour, ctx)
            pc = compute_path_cost(g, ci, chosen_path, hour)

            results[algo.name].append(
                DemandResult(
                    src=src,
                    dst=dst,
                    baseline_path=baseline_path,
                    chosen_path=chosen_path,
                    carbon=pc.carbon,
                    latency_ms=pc.latency_ms,
                    hops=pc.hops,
                    baseline_carbon=base_cost.carbon,
                    baseline_latency_ms=base_cost.latency_ms,
                    path_changed=(list(chosen_path) != list(baseline_path)),
                )
            )

    summaries: Dict[str, Summary] = {}
    for tag, rows in results.items():
        carb = [r.carbon for r in rows]
        lat  = [r.latency_ms for r in rows]
        hops = [float(r.hops) for r in rows]

        carb_red = []
        lat_inc = []
        changed = 0

        for r in rows:
            if r.baseline_carbon > 0:
                carb_red.append(100.0 * (r.baseline_carbon - r.carbon) / r.baseline_carbon)
            if r.baseline_latency_ms > 0:
                lat_inc.append(100.0 * (r.latency_ms - r.baseline_latency_ms) / r.baseline_latency_ms)
            if r.path_changed:
                changed += 1

        summaries[tag] = Summary(
            mean_carbon=_mean(carb),
            mean_latency_ms=_mean(lat),
            mean_hops=_mean(hops),
            mean_carbon_reduction_pct_vs_base=_mean(carb_red),
            mean_latency_increase_pct_vs_base=_mean(lat_inc),
            pct_paths_changed_vs_base=(100.0 * changed / len(rows)) if rows else 0.0,
        )

    return results, summaries
