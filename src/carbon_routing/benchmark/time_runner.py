from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import random

import networkx as nx

from ..ci.base import CIProvider
from ..metrics.path_cost import compute_path_cost
from ..algorithms.base import RoutingAlgorithm, AlgoContext

@dataclass(frozen=True)
class TimeSeriesSummary:
    mean_carbon_by_hour: List[float]
    mean_latency_by_hour: List[float]
    reroute_rate_by_hour: List[float]  # % changes vs previous hour (hour>=1)

def run_time_benchmark(
    g: nx.Graph,
    ci: CIProvider,
    algorithms: List[RoutingAlgorithm],
    ctx: AlgoContext,
    n_demands: int = 200,
    horizon_hours: int = 24,
    seed: int = 42,
) -> Dict[str, TimeSeriesSummary]:
    rng = random.Random(seed)
    n_as = g.number_of_nodes()

    algomap = {a.name: a for a in algorithms}

    # fixed demands
    demands: List[Tuple[int, int]] = []
    while len(demands) < n_demands:
        s = rng.randrange(0, n_as)
        d = rng.randrange(0, n_as)
        if s != d:
            demands.append((s, d))

    chosen_paths: Dict[str, List[List[Sequence[int]]]] = {
        name: [[[] for _ in range(n_demands)] for _ in range(horizon_hours)]
        for name in algomap.keys()
    }

    mean_carbon: Dict[str, List[float]] = {name: [0.0]*horizon_hours for name in algomap.keys()}
    mean_latency: Dict[str, List[float]] = {name: [0.0]*horizon_hours for name in algomap.keys()}

    for hour in range(horizon_hours):
        carb_sum = {name: 0.0 for name in algomap.keys()}
        lat_sum  = {name: 0.0 for name in algomap.keys()}

        for i, (src, dst) in enumerate(demands):
            for name, algo in algomap.items():
                p = algo.select_path(g, ci, src, dst, hour, ctx)
                pc = compute_path_cost(g, ci, p, hour)
                chosen_paths[name][hour][i] = p
                carb_sum[name] += pc.carbon
                lat_sum[name]  += pc.latency_ms

        for name in algomap.keys():
            mean_carbon[name][hour] = carb_sum[name] / n_demands
            mean_latency[name][hour] = lat_sum[name] / n_demands

    reroute_rate: Dict[str, List[float]] = {name: [0.0]*horizon_hours for name in algomap.keys()}
    for name in algomap.keys():
        for hour in range(1, horizon_hours):
            changes = 0
            for i in range(n_demands):
                if list(chosen_paths[name][hour][i]) != list(chosen_paths[name][hour-1][i]):
                    changes += 1
            reroute_rate[name][hour] = 100.0 * changes / n_demands

    out: Dict[str, TimeSeriesSummary] = {}
    for name in algomap.keys():
        out[name] = TimeSeriesSummary(
            mean_carbon_by_hour=mean_carbon[name],
            mean_latency_by_hour=mean_latency[name],
            reroute_rate_by_hour=reroute_rate[name],
        )
    return out
