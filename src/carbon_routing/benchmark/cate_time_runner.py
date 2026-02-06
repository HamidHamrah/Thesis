from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence, Set
import math

import networkx as nx

from ..ci.base import CIProvider
from ..device.models import RouterParams
from ..algorithms.cate import run_cate, route_tm_paths, CateResult
from ..metrics.cate_emissions import CateEmissions


@dataclass(frozen=True)
class CateHourRow:
    hour: int
    edges_final: int
    links_disabled: int
    max_util: float
    stop_reason: str
    dynamic_node: float
    idle_ports: float
    total: float


@dataclass(frozen=True)
class CateDayResult:
    rows: List[CateHourRow]
    reroute_rate_pct: List[float]          # len=23 (hour t -> t+1)
    topo_change_jaccard: List[float]       # len=23


def _edge_set(g: nx.Graph) -> Set[Tuple[int, int]]:
    s: Set[Tuple[int, int]] = set()
    for u, v in g.edges():
        a, b = (u, v) if u < v else (v, u)
        s.add((a, b))
    return s


def _apply_disabled_edges(base: nx.Graph, disabled: Sequence[Tuple[int, int]]) -> nx.Graph:
    g = base.copy()
    for u, v in disabled:
        if g.has_edge(u, v):
            g.remove_edge(u, v)
    return g


def _reroute_rate(paths_a: Dict[Tuple[int, int], Sequence[int]],
                  paths_b: Dict[Tuple[int, int], Sequence[int]]) -> float:
    changed = 0
    total = 0
    for k, p1 in paths_a.items():
        p2 = paths_b.get(k)
        if p2 is None:
            continue
        total += 1
        if list(p1) != list(p2):
            changed += 1
    return 100.0 * changed / total if total else 0.0


def _jaccard(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def run_cate_over_day(
    base_graph: nx.Graph,
    ci: CIProvider,
    router_params: Dict[int, RouterParams],
    tm: Dict[Tuple[int, int], float],
    hours: int = 24,
    default_capacity_mbps: float = 2000.0,
    port_beta_kw_per_link: float = 0.005,
    max_iterations: int = 1000,
    target_max_util: float = 0.80,
) -> CateDayResult:
    """
    For each hour:
      - run CATE on base_graph under CI[hour]
      - reconstruct final graph (base - disabled_edges)
      - compute TM paths on that final graph
    Then:
      - reroute_rate between consecutive hours
      - topo edge-set Jaccard between consecutive hours
    """
    hour_rows: List[CateHourRow] = []
    final_graphs: List[nx.Graph] = []
    hour_paths: List[Dict[Tuple[int, int], Sequence[int]]] = []

    # We'll reuse the same TM each hour (forecast TM). If you want a varying TM later,
    # we can generate tm_per_hour.
    for h in range(hours):
        cate_res: CateResult = run_cate(
            g=base_graph,
            ci=ci,
            router_params=router_params,
            tm=tm,
            hour=h,
            default_capacity_mbps=default_capacity_mbps,
            port_beta_kw_per_link=port_beta_kw_per_link,
            max_iterations=max_iterations,
        )

        g_h = _apply_disabled_edges(base_graph, cate_res.disabled_edges)
        final_graphs.append(g_h)

        # route TM on the final graph to compute reroutes later
        paths_h = route_tm_paths(g_h, ci, router_params, tm, hour=h)
        hour_paths.append(paths_h)

        e: CateEmissions = cate_res.emissions_final
        hour_rows.append(
            CateHourRow(
                hour=h,
                edges_final=cate_res.edges_final,
                links_disabled=cate_res.links_disabled,
                max_util=cate_res.max_link_utilization,
                stop_reason=cate_res.stop_reason,
                dynamic_node=e.dynamic_node,
                idle_ports=e.idle_ports,
                total=e.total,
            )
        )

    # reroute + topology change
    rr: List[float] = []
    tj: List[float] = []
    for h in range(hours - 1):
        rr.append(_reroute_rate(hour_paths[h], hour_paths[h + 1]))
        tj.append(_jaccard(_edge_set(final_graphs[h]), _edge_set(final_graphs[h + 1])))

    return CateDayResult(rows=hour_rows, reroute_rate_pct=rr, topo_change_jaccard=tj)
