from __future__ import annotations
from typing import Sequence, List, Tuple

import networkx as nx

from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency, path_latency_ms

class LowCarbBGP(RoutingAlgorithm):
    """
    Low-Carb BGP (LCB) — simulator-aligned core logic.

    Paper-aligned concept:
    - Each AS contributes an additive carbon metric along the AS-PATH.
    - Route selection prefers lower accumulated carbon (CIM chain),
      with BGP-like tie-breaking.

    In our evaluation sandbox we approximate "available BGP routes"
    as the K candidate AS-paths exposed to the decision-maker.

    Selection rule:
      1) Minimize CIM(path, t) = sum_{AS in path} CI_AS(t)
      2) Tie-break: minimize AS-path length (hops)
      3) Tie-break: minimize latency (sum of link latency_ms)

    Reference: Low-Carb BGP additive CIM chain concept. 
    """

    @property
    def name(self) -> str:
        return "lowcarb_bgp"

    @staticmethod
    def cim(ci: CIProvider, path: Sequence[int], hour: int) -> float:
        # Additive "carbon chain" along AS-path
        return float(sum(ci.get_ci(asn, hour) for asn in path))

    def select_path(
        self,
        g: nx.Graph,
        ci: CIProvider,
        src: int,
        dst: int,
        hour: int,
        ctx: AlgoContext,
    ) -> Sequence[int]:
        cands = k_shortest_paths_latency(g, src, dst, k=ctx.k_paths)

        # Rank by (CIM, hops, latency)
        scored: List[Tuple[float, int, float, Sequence[int]]] = []
        for p in cands:
            cim_val = self.cim(ci, p, hour)
            hops = max(0, len(p) - 1)
            lat = path_latency_ms(g, p)
            scored.append((cim_val, hops, lat, p))

        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        return scored[0][3]
