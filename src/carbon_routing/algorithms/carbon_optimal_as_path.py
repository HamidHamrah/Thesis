from __future__ import annotations

from typing import Sequence, List, Tuple, Dict, Any

import networkx as nx

from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency

Pair = Tuple[int, int]
Paths = Dict[Pair, List[int]]


class CarbonOptimalASPath(RoutingAlgorithm):
    """
    Carbon Optimal AS-Path Selection.

    Decision rule (pure carbon minimization over identical candidate AS paths):
      C(P) = sum_{AS_i in P} CI(AS_i, t)
      P*   = argmin_P C(P)

    Notes:
    - Operates strictly at AS-path level.
    - Uses the same candidate path generator as other inter-domain algorithms.
    - Ignores BGP policy/transposition constraints by design.
    """

    @property
    def name(self) -> str:
        return "carbon_optimal_as_path"

    @staticmethod
    def path_carbon(ci: CIProvider, path: Sequence[int], hour: int) -> float:
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
        # Candidate set is identical to other algorithms (same topology + k shortest AS paths).
        cands = k_shortest_paths_latency(g, src, dst, k=ctx.k_paths)
        # Pure path evaluation rule: minimize accumulated carbon.
        return min(cands, key=lambda p: self.path_carbon(ci, p, hour))


def run_carbon_optimal_as_path(
    g: nx.Graph,
    ci: CIProvider,
    router_params: Any,
    pairs: List[Pair],
    hour: int = 0,
    k: int = 8,
) -> Paths:
    """
    Wrapper matching registry interface.
    """
    ctx = AlgoContext(k_paths=k, paper_faithful=True)
    algo = CarbonOptimalASPath()
    out: Paths = {}
    for src, dst in pairs:
        out[(src, dst)] = list(algo.select_path(g, ci, src, dst, hour, ctx))
    return out

