from __future__ import annotations
from typing import Sequence, Dict, List, Tuple
import networkx as nx
from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency

Pair = Tuple[int, int]
Paths = Dict[Pair, List[int]]

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


def run_baseline_latency(g: nx.Graph, pairs: List[Pair]) -> Paths:
    """
    Wrapper function for baseline latency algorithm.
    Selects the lowest-latency path for each pair.
    """
    ctx = AlgoContext(k_paths=8)
    algo = BaselineLatency()
    result: Paths = {}
    for src, dst in pairs:
        path = algo.select_path(g, None, src, dst, 0, ctx)
        result[(src, dst)] = list(path)
    return result
