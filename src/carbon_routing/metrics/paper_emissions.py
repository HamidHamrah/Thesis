from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import networkx as nx

from ..ci.base import CIProvider
from ..device.models import RouterParams

@dataclass(frozen=True)
class PaperEmissionCost:
    emissions: float   # proportional units: (W/Mbps)*Mbps*gCO2
    latency_ms: float
    hops: int

def compute_paper_emissions_cost(
    g: nx.Graph,
    ci: CIProvider,
    router_params: dict[int, RouterParams],
    path: Sequence[int],
    hour: int,
    demand_mbps: float = 100.0,
) -> PaperEmissionCost:
    """
    Paper-aligned proxy for traffic-induced node processing emissions:
      E(path) = sum_{j in path} (lambda_j * demand_mbps * CI_j(hour))

    This matches the intent behind IncD and C+IncD where lambda_j represents
    incremental power per Mbps, combined with carbon intensity. 
    """
    emissions = 0.0
    for j in path:
        lam = float(router_params[j].incd_w_per_mbps)
        emissions += lam * float(demand_mbps) * float(ci.get_ci(j, hour))

    latency_ms = 0.0
    for a, b in zip(path[:-1], path[1:]):
        latency_ms += float(g[a][b].get("latency_ms", 1.0))

    hops = max(0, len(path) - 1)
    return PaperEmissionCost(emissions=emissions, latency_ms=latency_ms, hops=hops)
