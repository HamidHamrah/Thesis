from __future__ import annotations
from typing import Sequence
import networkx as nx
from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency

class BaselineLatency(RoutingAlgorithm):
    """
    Baseline (performance-first):
    - generate K candidate paths ordered by latency
    - choose the lowest-latency candidate (candidate[0])

    This is your reference algorithm.
    """

    @property
    def name(self) -> str:
        return "baseline_latency"

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
        return cands[0]
