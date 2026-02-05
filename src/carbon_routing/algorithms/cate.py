from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence, Optional

import networkx as nx

from .base import AlgoContext
from ..ci.base import CIProvider
from ..device.models import RouterParams
from ..metrics.cate_emissions import compute_cate_emissions, CateEmissions


@dataclass(frozen=True)
class CateResult:
    edges_initial: int
    edges_final: int
    links_disabled: int
    connected_final: bool
    emissions_initial: CateEmissions
    emissions_final: CateEmissions
    emissions_reduction_pct: float
    max_link_utilization: float  # max(load/capacity)
    accepted_removals: List[Tuple[int, int]]
    stop_reason: str
    history: List[Tuple[int, float, float]]  # (edges, emissions_total, max_util)


def _ensure_edge_capacity(g: nx.Graph, default_capacity_mbps: float = 10_000.0) -> None:
    """
    Ensure every undirected edge has capacity_mbps attribute.
    """
    for u, v, data in g.edges(data=True):
        if "capacity_mbps" not in data:
            data["capacity_mbps"] = float(default_capacity_mbps)


def _directed_cost_c_plus_incd(
    ci: CIProvider,
    params: Dict[int, RouterParams],
    i: int,
    j: int,
    hour: int,
) -> float:
    # Table-1 style C+IncD link cost: 1 + λ_j * c_j, receiver-based
    return 1.0 + float(params[j].incd_w_per_mbps) * float(ci.get_ci(j, hour))


def _shortest_path_receiver_weighted(
    g: nx.Graph,
    ci: CIProvider,
    params: Dict[int, RouterParams],
    src: int,
    dst: int,
    hour: int,
) -> Optional[Sequence[int]]:
    """
    Dijkstra on a directed view with receiver-based weights.
    Returns None if no path exists.
    """
    dg = nx.DiGraph()
    dg.add_nodes_from(g.nodes())

    for u, v in g.edges():
        dg.add_edge(u, v, weight=_directed_cost_c_plus_incd(ci, params, u, v, hour))
        dg.add_edge(v, u, weight=_directed_cost_c_plus_incd(ci, params, v, u, hour))

    try:
        return nx.shortest_path(dg, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return None


def _route_tm_and_intensities(
    g: nx.Graph,
    ci: CIProvider,
    params: Dict[int, RouterParams],
    tm: Dict[Tuple[int, int], float],
    hour: int,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], bool]:
    """
    Route all demands using receiver-based C+IncD Dijkstra.
    Return:
      - node_intensity_mbps: x_n (sum demand entering/processed at node)
      - undirected_link_intensity_mbps: y_{u,v} (sum over both directions)
      - directed_link_load_mbps: load on each direction (u,v)
      - feasible: False if any demand becomes unroutable
    """
    node_intensity: Dict[int, float] = {n: 0.0 for n in g.nodes()}
    undir_link_intensity: Dict[Tuple[int, int], float] = {}
    dir_load: Dict[Tuple[int, int], float] = {}

    for (s, d), mbps in tm.items():
        path = _shortest_path_receiver_weighted(g, ci, params, s, d, hour)
        if path is None:
            return node_intensity, undir_link_intensity, dir_load, False

        # Receiver processing: each hop adds demand to the receiving node
        for a, b in zip(path[:-1], path[1:]):
            node_intensity[b] += mbps
            dir_load[(a, b)] = dir_load.get((a, b), 0.0) + mbps

            u, v = (a, b) if a < b else (b, a)
            undir_link_intensity[(u, v)] = undir_link_intensity.get((u, v), 0.0) + mbps

    return node_intensity, undir_link_intensity, dir_load, True


def _capacity_ok(g: nx.Graph, dir_load: Dict[Tuple[int, int], float]) -> Tuple[bool, float]:
    """
    Check directed loads against undirected edge capacity (shared for both dirs).
    We enforce per-direction load <= capacity_mbps as a simple constraint.
    Also compute maximum utilization.
    """
    max_util = 0.0
    for (a, b), load in dir_load.items():
        if not g.has_edge(a, b):
            return False, 1.0

        cap = float(g[a][b].get("capacity_mbps", 0.0))
        if cap <= 0:
            return False, 1.0

        util = float(load) / cap
        if util > max_util:
            max_util = util

        if load > cap:
            return False, max_util

    return True, max_util


def cate_rank_links_for_shutdown(
    g: nx.Graph,
    ci: CIProvider,
    params: Dict[int, RouterParams],
    undir_intensity: Dict[Tuple[int, int], float],
    hour: int,
) -> List[Tuple[float, Tuple[int, int]]]:
    """
    Rank links for shutdown using a simple paper-inspired heuristic:
      if y_l == 0 -> cost = +inf (best candidate to disable)
      else cost = (λ_u*c_u + λ_v*c_v) / y_l
    where y_l is undirected link intensity and λ_i*c_i is a receiver-side factor.
    Higher cost removed earlier.
    """
    scored: List[Tuple[float, Tuple[int, int]]] = []
    for u, v in g.edges():
        a, b = (u, v) if u < v else (v, u)
        y = float(undir_intensity.get((a, b), 0.0))
        if y <= 0.0:
            cost = float("inf")
        else:
            num = float(params[u].incd_w_per_mbps) * float(ci.get_ci(u, hour)) \
                + float(params[v].incd_w_per_mbps) * float(ci.get_ci(v, hour))
            cost = num / y
        scored.append((cost, (u, v)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def run_cate(
    g: nx.Graph,
    ci: CIProvider,
    router_params: Dict[int, RouterParams],
    tm: Dict[Tuple[int, int], float],
    hour: int = 0,
    default_capacity_mbps: float = 10_000.0,
    port_beta_kw_per_link: float = 0.05,
    max_iterations: int | None = None,
    improvement_tol: float = 1e-9,
) -> CateResult:
    """
    Centralized CATE loop:
      - route TM (receiver-based C+IncD shortest paths)
      - compute emissions
      - rank links for shutdown and attempt removals greedily
      - accept a removal only if: connected, routable, capacities ok, emissions improve
    """
    g_work = g.copy()
    _ensure_edge_capacity(g_work, default_capacity_mbps)

    # initial route
    x0, y0, dir0, feas0 = _route_tm_and_intensities(g_work, ci, router_params, tm, hour)
    if not feas0:
        raise RuntimeError("Initial TM is not routable on the starting graph.")

    e0 = compute_cate_emissions(
        g_work, ci, router_params, x0, hour, port_beta_kw_per_link=port_beta_kw_per_link
    )
    ok0, max_util0 = _capacity_ok(g_work, dir0)
    if not ok0:
        raise RuntimeError("Initial TM violates capacity constraints (unexpected).")

    best_em = e0.total
    best_graph = g_work
    accepted: List[Tuple[int, int]] = []
    max_util_best = max_util0
    history: List[Tuple[int, float, float]] = [
        (best_graph.number_of_edges(), best_em, max_util_best)
    ]
    stop_reason = "max_iterations_reached"

    if max_iterations is None:
        max_iterations = g_work.number_of_edges()

    improved = True
    iters = 0

    while improved and iters < max_iterations:
        iters += 1
        improved = False

        # recompute intensities for current best graph
        x, y, dir_load, feasible = _route_tm_and_intensities(best_graph, ci, router_params, tm, hour)
        if not feasible:
            break

        candidates = cate_rank_links_for_shutdown(best_graph, ci, router_params, y, hour)

        # try candidates in order until we accept one removal
        for _, (u, v) in candidates:
            if not best_graph.has_edge(u, v):
                continue

            trial = best_graph.copy()
            trial.remove_edge(u, v)

            if not nx.is_connected(trial):
                continue

            x2, y2, dir2, feas2 = _route_tm_and_intensities(trial, ci, router_params, tm, hour)
            if not feas2:
                continue

            ok, max_util = _capacity_ok(trial, dir2)
            if not ok:
                continue

            e2 = compute_cate_emissions(
                trial, ci, router_params, x2, hour, port_beta_kw_per_link=port_beta_kw_per_link
            )

            if e2.total + improvement_tol < best_em:
                # accept
                best_graph = trial
                best_em = e2.total
                accepted.append((u, v))
                max_util_best = max_util
                if len(accepted) % 10 == 0:
                    history.append(
                        (best_graph.number_of_edges(), best_em, max_util_best)
                    )
                improved = True
                break

        if not improved:
            stop_reason = "no_improving_candidate"
            break

    # final
    xf, yf, dirf, feasf = _route_tm_and_intensities(best_graph, ci, router_params, tm, hour)
    if not feasf:
        raise RuntimeError("CATE ended in an unroutable state (should not happen).")

    ef = compute_cate_emissions(
        best_graph, ci, router_params, xf, hour, port_beta_kw_per_link=port_beta_kw_per_link
    )

    red_pct = 0.0
    if e0.total > 0:
        red_pct = 100.0 * (e0.total - ef.total) / e0.total

    return CateResult(
        edges_initial=g.number_of_edges(),
        edges_final=best_graph.number_of_edges(),
        links_disabled=g.number_of_edges() - best_graph.number_of_edges(),
        connected_final=nx.is_connected(best_graph),
        emissions_initial=e0,
        emissions_final=ef,
        emissions_reduction_pct=red_pct,
        max_link_utilization=max_util_best,
        accepted_removals=accepted,
        stop_reason=stop_reason,
        history=history,
    )
