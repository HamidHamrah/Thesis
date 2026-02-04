from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Callable
import networkx as nx
from ..ci.base import CIProvider


# --- Shortest Path Selection Strategies ---
# Implement various shortest path selection strategies
# for different objectives: latency-only, carbon-only,
# and weighted multi-objective.
# ------------------------------------------------

@dataclass(frozen=True)
class SelectedPath:
    path: Sequence[int]
    objective_value: float

def select_baseline_latency_sp(g: nx.Graph, src: int, dst: int) -> SelectedPath:
    """
    Baseline: minimize latency only.
    """
    path = nx.shortest_path(g, src, dst, weight="latency_ms")
    # Objective value = total latency
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += float(g.edges[u, v].get("latency_ms", 0.0))
    return SelectedPath(path=path, objective_value=total)

def select_carbon_sp(g: nx.Graph, ci: CIProvider, src: int, dst: int, hour: int) -> SelectedPath:
    """
    Carbon-only: minimize sum of node CI along path.
    Implemented by transforming into an edge weight problem:
      weight(u->v) = CI(v)  (and add CI(src) once)
    """
    ci_src = ci.get_ci(src, hour)

    def edge_weight(u: int, v: int, attrs: dict) -> float:
        return float(ci.get_ci(v, hour))

    path = nx.shortest_path(g, src, dst, weight=edge_weight)
    total = ci_src + sum(ci.get_ci(asn, hour) for asn in path[1:])
    return SelectedPath(path=path, objective_value=float(total))

def select_weighted_sp(
    g: nx.Graph,
    ci: CIProvider,
    src: int,
    dst: int,
    hour: int,
    alpha: float,
) -> SelectedPath:
    """
    Weighted multi-objective shortest path:
      minimize alpha * carbon + (1-alpha) * latency
    where:
      carbon = sum CI over nodes
      latency = sum latency_ms over edges

    Implemented as edge weights:
      w(u->v) = alpha * CI(v) + (1-alpha) * latency(u,v)
      plus alpha * CI(src) added once.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")

    ci_src = ci.get_ci(src, hour)

    def edge_weight(u: int, v: int, attrs: dict) -> float:
        lat = float(attrs.get("latency_ms", 0.0))
        return float(alpha * ci.get_ci(v, hour) + (1.0 - alpha) * lat)

    path = nx.shortest_path(g, src, dst, weight=edge_weight)

    total_latency = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total_latency += float(g.edges[u, v].get("latency_ms", 0.0))

    total_carbon = ci_src + sum(ci.get_ci(asn, hour) for asn in path[1:])
    total_obj = alpha * total_carbon + (1.0 - alpha) * total_latency

    return SelectedPath(path=path, objective_value=float(total_obj))
