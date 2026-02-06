from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import networkx as nx

from ..algorithms.base import AlgoContext
from ..algorithms.ce import run_ce
from ..metrics.path_cost import path_carbon, path_latency, path_dir_load_for_pairs
from ..metrics.utilization import compute_link_utilization


@dataclass(frozen=True)
class CEDayRow:
    hour: int
    mean_carbon: float
    mean_latency_ms: float
    reroute_rate_pct: float


def run_ce_over_day(
    ctx: AlgoContext,
    pairs: List[Tuple[int, int]],
    hours: int = 24,
    gamma: float = 1.0,
) -> List[CEDayRow]:
    """
    Run CE over 24h where hour t uses utilization from hour t-1.
    For hour 0, prev_util = zeros.
    """
    prev_paths: Dict[Tuple[int, int], List[int]] | None = None
    prev_util: Dict[Tuple[int, int], float] = {}  # empty => treated as 0
    rows: List[CEDayRow] = []

    for h in range(hours):
        res = run_ce(ctx, pairs=pairs, hour=h, prev_util_undir=prev_util, gamma=gamma)
        paths = res.paths

        # costs
        carb = []
        lat = []
        for (s, d), p in paths.items():
            carb.append(path_carbon(ctx.g, ctx.ci, p, hour=h))
            lat.append(path_latency(ctx.g, p))

        # reroute rate
        rr = 0.0
        if prev_paths is not None:
            changed = sum(1 for k in paths if paths[k] != prev_paths.get(k))
            rr = 100.0 * changed / len(paths) if paths else 0.0

        # compute directed loads from chosen paths and update utilization for next hour
        dir_load = path_dir_load_for_pairs(paths, ctx.tm)  # if you have TM; else use unit demand
        prev_util = compute_link_utilization(ctx.g, dir_load)

        rows.append(
            CEDayRow(
                hour=h,
                mean_carbon=sum(carb)/len(carb),
                mean_latency_ms=sum(lat)/len(lat),
                reroute_rate_pct=rr,
            )
        )
        prev_paths = paths

    return rows
