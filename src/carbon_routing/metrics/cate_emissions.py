from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from ..ci.base import CIProvider
from ..device.models import RouterParams

@dataclass(frozen=True)
class CateEmissions:
    dynamic_node: float  # traffic-induced processing emissions
    idle_ports: float    # link/port emissions for enabled links
    total: float

def compute_cate_emissions(
    g: nx.Graph,
    ci: CIProvider,
    router_params: Dict[int, RouterParams],
    node_intensity_mbps: Dict[int, float],
    hour: int,
    port_beta_kw_per_link: float = 0.05,  # synthetic, can be replaced later
) -> CateEmissions:
    """
    C_tot ≈ Σ_n (x_n * λ_n * c_n) + Σ_l β_l (c_u + c_v)

    - x_n: node intensity (Mbps processed by node n)
    - λ_n: incremental W/Mbps for node n
    - c_n: CI at node n (gCO2/kWh)
    - β_l: link port idle power in kW (synthetic here)
    """
    # Dynamic node term
    dyn = 0.0
    for n, x_mbps in node_intensity_mbps.items():
        lam_w_per_mbps = float(router_params[n].incd_w_per_mbps)
        power_kw = (x_mbps * lam_w_per_mbps) / 1000.0
        dyn += power_kw * float(ci.get_ci(n, hour))

    # Idle port term (enabled links only)
    idle = 0.0
    for u, v in g.edges():
        idle += port_beta_kw_per_link * (float(ci.get_ci(u, hour)) + float(ci.get_ci(v, hour)))

    return CateEmissions(dynamic_node=dyn, idle_ports=idle, total=dyn + idle)
