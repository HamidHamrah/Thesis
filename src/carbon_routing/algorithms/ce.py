from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence

import networkx as nx

from .base import AlgoContext, AlgorithmResult
from ..metrics.utilization import ce_penalty


@dataclass(frozen=True)
class CERouteDecision:
    gamma: float  # weight on congestion penalty


def _directed_cost_c_incd(
    ctx: AlgoContext,
    i: int,
    j: int,
    hour: int,
) -> float:
    # C+IncD receiver-based cost (same as Step 10)
    return 1.0 + float(ctx.router_params[j].incd_w_per_mbps) * float(ctx.ci.get_ci(j, hour))


def _directed_cost_ce(
    ctx: AlgoContext,
    i: int,
    j: int,
    hour: int,
    prev_util_undir: Dict[Tuple[int, int], float],
    gamma: float,
) -> float:
    base = _directed_cost_c_incd(ctx, i, j, hour)
    u, v = (i, j) if i < j else (j, i)
    util = float(prev_util_undir.get((u, v), 0.0))
    return base + gamma * ce_penalty(util)


def _shortest_path_ce(
    ctx: AlgoContext,
    src: int,
    dst: int,
    hour: int,
    prev_util_undir: Dict[Tuple[int, int], float],
    gamma: float,
) -> Optional[Sequence[int]]:
    g = ctx.g
    dg = nx.DiGraph()
    dg.add_nodes_from(g.nodes())

    for u, v in g.edges():
        dg.add_edge(u, v, weight=_directed_cost_ce(ctx, u, v, hour, prev_util_undir, gamma))
        dg.add_edge(v, u, weight=_directed_cost_ce(ctx, v, u, hour, prev_util_undir, gamma))

    try:
        return nx.shortest_path(dg, src, dst, weight="weight")
    except nx.NetworkXNoPath:
        return None


def run_ce(
    ctx: AlgoContext,
    pairs: list[tuple[int, int]],
    hour: int,
    prev_util_undir: Dict[Tuple[int, int], float],
    gamma: float = 1.0,
) -> AlgorithmResult:
    """
    CE routing: carbon-aware + congestion penalty from previous hour utilization.
    """
    if ctx.g is None or ctx.ci is None or ctx.router_params is None:
        raise RuntimeError("CE requires ctx.g, ctx.ci, and ctx.router_params to be set.")
    chosen_paths: Dict[tuple[int, int], list[int]] = {}

    for s, d in pairs:
        p = _shortest_path_ce(ctx, s, d, hour, prev_util_undir, gamma)
        if p is None:
            raise RuntimeError(f"CE: no path {s}->{d} at hour={hour}")
        chosen_paths[(s, d)] = list(p)

    return AlgorithmResult(name=f"CE(gamma={gamma})", paths=chosen_paths)
