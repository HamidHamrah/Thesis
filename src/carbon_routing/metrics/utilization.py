from __future__ import annotations
from typing import Dict, Tuple
import networkx as nx

def compute_link_utilization(
    g: nx.Graph,
    dir_load_mbps: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """
    Returns undirected utilization for each edge: util(u,v) in [0, +inf).
    Uses sum of both directions / capacity as a simple shared-capacity model.
    """
    util: Dict[Tuple[int, int], float] = {}
    # sum both directions on same undirected edge
    sum_undir: Dict[Tuple[int, int], float] = {}
    for (a, b), load in dir_load_mbps.items():
        u, v = (a, b) if a < b else (b, a)
        sum_undir[(u, v)] = sum_undir.get((u, v), 0.0) + float(load)

    for u, v in g.edges():
        a, b = (u, v) if u < v else (v, u)
        cap = float(g[u][v].get("capacity_mbps", 0.0))
        if cap <= 0:
            continue
        load = sum_undir.get((a, b), 0.0)
        util[(a, b)] = load / cap

    return util


def ce_penalty(util: float, eps: float = 1e-6) -> float:
    """
    Convex penalty: util/(1-util) with epsilon for stability.
    """
    util = max(0.0, min(util, 1.0 - eps))
    return util / (1.0 - util + eps)
