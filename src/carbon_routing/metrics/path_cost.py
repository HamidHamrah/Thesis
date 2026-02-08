from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import networkx as nx
from ..ci.base import CIProvider
# --- Path Cost Metrics ---
# Defines functions to compute path costs in terms of
# carbon intensity, latency, and hops.
# ------------------------------------------------------    


@dataclass(frozen=True)
class PathCost:
    carbon: float
    latency_ms: float
    hops: int

def path_latency_ms(g: nx.Graph, path: Sequence[int]) -> float:
    """Sum of edge latencies along the path."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += float(g.edges[u, v].get("latency_ms", 0.0))
    return total

def path_latency(g: nx.Graph, path: Sequence[int]) -> float:
    """Compatibility alias for older callers."""
    return path_latency_ms(g, path)

def path_carbon_cost(ci: CIProvider, path: Sequence[int], hour: int) -> float:
    """
    Carbon cost proxy (Step 2):
    Sum CI of each AS on the path at time 'hour'.

    Note: This is a *proxy* cost (gCO2/kWh summed). Later we can convert to
    actual emissions per bit by multiplying by energy-per-bit.
    """
    return float(sum(ci.get_ci(asn, hour) for asn in path))

def path_carbon(g: nx.Graph, ci: CIProvider, path: Sequence[int], hour: int) -> float:
    """Compatibility alias for older callers (g unused)."""
    return path_carbon_cost(ci, path, hour)

def compute_path_cost(g: nx.Graph, ci: CIProvider, path: Sequence[int], hour: int) -> PathCost:
    carbon = path_carbon_cost(ci, path, hour)
    latency = path_latency_ms(g, path)
    hops = max(0, len(path) - 1)
    return PathCost(carbon=carbon, latency_ms=latency, hops=hops)
