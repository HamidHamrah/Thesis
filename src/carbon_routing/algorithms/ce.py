from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence, List, Any

import networkx as nx

from .base import AlgoContext, AlgorithmResult

Pair = Tuple[int, int]
Paths = Dict[Pair, List[int]]

@dataclass(frozen=True)
class CERouteDecision:
    beta: float  # CE down-scaling factor


def _directed_cost_c_incd(
    ctx: AlgoContext,
    i: int,
    j: int,
    hour: int,
) -> float:
    # C+IncD receiver-based cost (same as Step 10)
    return 1.0 + float(ctx.router_params[j].incd_w_per_mbps) * float(ctx.ci.get_ci(j, hour))


def _beta_max(router_params: Dict[int, Any]) -> float:
    pmax_max = max(float(p.pmax_kw) for p in router_params.values())
    if pmax_max <= 0.0:
        return 1.0
    return 65535.0 / (950.0 * pmax_max)


def _directed_cost_ce(
    ctx: AlgoContext,
    i: int,
    j: int,
    hour: int,
    prev_node_util_mbps: Dict[int, float],
    beta: float,
) -> float:
    # Paper Table-1 CE:
    # 1 + beta * c_j * (Pidle_j + lambda_j * U_{j,Δt})
    # where lambda is W/Mbps and U is Mbps.
    p = ctx.router_params[j]
    c_j = float(ctx.ci.get_ci(j, hour))
    u_prev = float(prev_node_util_mbps.get(j, 0.0))
    dyn_kw = (float(p.incd_w_per_mbps) * u_prev) / 1000.0
    power_kw = float(p.pidle_kw) + dyn_kw
    power_kw = min(power_kw, float(p.pmax_kw))
    return 1.0 + float(beta) * c_j * power_kw


def _shortest_path_ce(
    ctx: AlgoContext,
    src: int,
    dst: int,
    hour: int,
    prev_node_util_mbps: Dict[int, float],
    beta: float,
) -> Optional[Sequence[int]]:
    g = ctx.g
    dg = nx.DiGraph()
    dg.add_nodes_from(g.nodes())

    for u, v in g.edges():
        dg.add_edge(u, v, weight=_directed_cost_ce(ctx, u, v, hour, prev_node_util_mbps, beta))
        dg.add_edge(v, u, weight=_directed_cost_ce(ctx, v, u, hour, prev_node_util_mbps, beta))

    try:
        return nx.shortest_path(dg, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return None


def run_ce(
    ctx: AlgoContext,
    pairs: list[tuple[int, int]],
    hour: int,
    prev_node_util_mbps: Dict[int, float] | None = None,
    beta: float | None = None,
    # Backward-compatible legacy parameters:
    prev_util_undir: Dict[Tuple[int, int], float] | None = None,
    gamma: float | None = None,
) -> AlgorithmResult:
    """
    CE routing with paper Table-1 CE metric.
    Uses previous-interval node utilization U_{j,Δt}.
    """
    if ctx.g is None or ctx.ci is None or ctx.router_params is None:
        raise RuntimeError("CE requires ctx.g, ctx.ci, and ctx.router_params to be set.")

    if prev_node_util_mbps is None:
        prev_node_util_mbps = {}

    # If only legacy link-util map is provided, degrade gracefully using zeros
    # to preserve call compatibility.
    if prev_util_undir is not None and not prev_node_util_mbps:
        prev_node_util_mbps = {}

    if beta is None:
        beta = _beta_max(ctx.router_params)

    chosen_paths: Dict[tuple[int, int], list[int]] = {}

    for s, d in pairs:
        p = _shortest_path_ce(ctx, s, d, hour, prev_node_util_mbps, beta)
        if p is None:
            raise RuntimeError(f"CE: no path {s}->{d} at hour={hour}")
        chosen_paths[(s, d)] = list(p)

    return AlgorithmResult(name=f"CE(beta={beta:.6f})", paths=chosen_paths)


def run_ce_wrapper(
    g: nx.Graph,
    ci: Any,
    router_params: Any,
    pairs: List[Pair],
    hour: int = 0,
    prev_node_util_mbps: Optional[Dict[int, float]] = None,
    beta: float | None = None,
    # Backward-compatible params:
    prev_util_undir: Optional[Dict[Tuple[int, int], float]] = None,
    gamma: float = 1.0,
) -> AlgorithmResult:
    """
    Wrapper function for CE algorithm that matches the registry interface.
    """
    ctx = AlgoContext(g=g, ci=ci, router_params=router_params)
    return run_ce(
        ctx=ctx,
        pairs=pairs,
        hour=hour,
        prev_node_util_mbps=prev_node_util_mbps,
        beta=beta,
        prev_util_undir=prev_util_undir,
        gamma=gamma,
    )
