from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import networkx as nx

from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..device.models import RouterParams

OSPF_MIN = 1
OSPF_MAX = 65535  # 16-bit; 0 forbidden :contentReference[oaicite:7]{index=7}

def _clamp_ospf16(x: float) -> int:
    if x < OSPF_MIN:
        return OSPF_MIN
    if x > OSPF_MAX:
        return OSPF_MAX
    return int(round(x))

@dataclass(frozen=True)
class MetricSpec:
    name: str
    # raw_cost(i, j, hour) : cost of directional link l_{i->j}
    # Per paper: depends on RECEIVING router j :contentReference[oaicite:8]{index=8}
    raw_cost: Callable[[int, int, int], float]

class OspfMetricRouting(RoutingAlgorithm):
    """
    OSPF-style shortest path with Table-1 metric-derived directional link costs.
    - Links are bidirectional but costs differ by direction.
    - l_{i->j} uses metrics of receiving router j. :contentReference[oaicite:9]{index=9}
    """
    def __init__(self, spec: MetricSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def select_path(
        self,
        g: nx.Graph,
        ci: CIProvider,
        src: int,
        dst: int,
        hour: int,
        ctx: AlgoContext,
    ) -> Sequence[int]:
        dg = nx.DiGraph()
        dg.add_nodes_from(g.nodes())

        for u, v, _ in g.edges(data=True):
            w_uv = _clamp_ospf16(self.spec.raw_cost(u, v, hour))
            w_vu = _clamp_ospf16(self.spec.raw_cost(v, u, hour))
            dg.add_edge(u, v, weight=w_uv)
            dg.add_edge(v, u, weight=w_vu)

        return nx.shortest_path(dg, src, dst, weight="weight")

def build_table1_metric_specs(
    params: dict[int, RouterParams],
    ci: CIProvider,
    alpha: float = 200_000.0,   # α scaling for IncD; paper suggests [100k..640k] :contentReference[oaicite:10]{index=10}
) -> dict[str, MetricSpec]:
    # Needed for β scaling in C+Ptyp and CE :contentReference[oaicite:11]{index=11}
    ptyp_max = max(p.ptyp_kw for p in params.values())
    pmax_max = max(p.pmax_kw for p in params.values())

    beta_ptyp = OSPF_MAX / (950.0 * ptyp_max)  # βmax = 64K/(950*Ptyp_max) :contentReference[oaicite:12]{index=12}
    beta_ce   = OSPF_MAX / (950.0 * pmax_max)  # βmax = 64K/(950*Pmax) :contentReference[oaicite:13]{index=13}

    # Metric raw-cost functions (directional, receiver j)
    def OSPF(i, j, h):      return 1.0
    def Ptyp(i, j, h):      return params[j].ptyp_kw
    def E_label(i, j, h):   return float(params[j].elabel_cost)
    def IncD(i, j, h):      return alpha * params[j].incd_w_per_mbps
    def C(i, j, h):         return 1.0 + ci.get_ci(j, h)  # +1 guaranteed min :contentReference[oaicite:14]{index=14}
    def C_Ptyp(i, j, h):    return 1.0 + beta_ptyp * ci.get_ci(j, h) * params[j].ptyp_kw
    def C_Elabel(i, j, h):  return 1.0 + ci.get_ci(j, h) * params[j].elabel_cost
    def C_IncD(i, j, h):    return 1.0 + params[j].incd_w_per_mbps * ci.get_ci(j, h)

    # CE uses utilization from previous interval U_{j,Δt}. We stub U=0 for Step 10,
    # and we will hook it properly when we implement traffic-matrix-based CATE. :contentReference[oaicite:15]{index=15}
    def CE(i, j, h):
        U_prev_mbps = 0.0
        power_kw = params[j].pidle_kw + params[j].incd_w_per_mbps * U_prev_mbps
        if power_kw > params[j].pmax_kw:
            power_kw = params[j].pmax_kw
        return 1.0 + beta_ce * ci.get_ci(j, h) * power_kw

    return {
        "OSPF":      MetricSpec("OSPF", OSPF),
        "Ptyp":      MetricSpec("Ptyp", Ptyp),
        "E-label":   MetricSpec("E-label", E_label),
        "IncD":      MetricSpec("IncD", IncD),
        "C":         MetricSpec("C", C),
        "C+Ptyp":    MetricSpec("C+Ptyp", C_Ptyp),
        "C+E-label": MetricSpec("C+E-label", C_Elabel),
        "C+IncD":    MetricSpec("C+IncD", C_IncD),
        "CE":        MetricSpec("CE", CE),
    }
