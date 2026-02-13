from __future__ import annotations
from typing import Sequence, List, Tuple, Dict, Any
import statistics

import networkx as nx

from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency, path_latency_ms

Pair = Tuple[int, int]
Paths = Dict[Pair, List[int]]

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
    def cim_chain(ci: CIProvider, path: Sequence[int], hour: int, default_cim: float = 1000.0) -> List[float]:
        chain: List[float] = []
        for asn in path:
            try:
                chain.append(float(ci.get_ci(asn, hour)))
            except Exception:
                chain.append(float(default_cim))
        return chain

    @staticmethod
    def aggregate_cim(chain: Sequence[float], aggregate: str) -> float:
        if not chain:
            return float("inf")
        a = aggregate.strip().upper()
        if a == "TCIM":
            return float(sum(chain))
        if a == "MCIM":
            return float(statistics.median(chain))
        if a == "HCIM":
            return float(max(chain))
        return float(sum(chain))

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

        # Paper-faithful policy path:
        # 1) Build CIM chain and aggregate metric (TCIM/MCIM/HCIM)
        # 2) Optional route filtering based on HCIM threshold
        # 3) Transpose aggregate into BGP attribute and run BGP-like best-path order
        scored: List[Tuple[Tuple[float, float, int, float, float], Sequence[int]]] = []
        aggregate = ctx.lcb_aggregate
        transpose = ctx.lcb_transpose_attr.strip().lower()

        for p in cands:
            chain = self.cim_chain(ci, p, hour, default_cim=ctx.lcb_default_cim)
            tcim = self.aggregate_cim(chain, "TCIM")
            mcim = self.aggregate_cim(chain, "MCIM")
            hcim = self.aggregate_cim(chain, "HCIM")
            cim_val = self.aggregate_cim(chain, aggregate)

            # Route-selection style filter (paper policy option)
            if ctx.lcb_hcim_threshold is not None and hcim > float(ctx.lcb_hcim_threshold):
                continue

            hops = max(0, len(p) - 1)
            lat = path_latency_ms(g, p)

            # Transposition of aggregate to an existing BGP attribute
            # Default: convert TCIM->LocalPref with inversion (lower TCIM => higher pref).
            weight = 0.0
            local_pref = 0.0
            med = 0.0
            aspath_metric = float(hops)

            if transpose == "weight":
                weight = max(1.0, float(ctx.lcb_base_pref) - cim_val)
            elif transpose == "local_pref":
                local_pref = max(1.0, float(ctx.lcb_base_pref) - cim_val)
            elif transpose == "med":
                med = cim_val
            elif transpose == "aspath":
                # Carbon-aware AS-PATH penalty proxy.
                aspath_metric = hops + (hcim / 1000.0)
            else:
                local_pref = max(1.0, float(ctx.lcb_base_pref) - cim_val)

            # BGP-like decision order (lower tuple wins):
            # highest Weight, highest LocalPref, shortest AS-PATH, lowest MED, lowest IGP cost
            key = (-weight, -local_pref, int(round(aspath_metric)), med, lat)
            scored.append((key, p))

        if not scored:
            # If all routes filtered, fallback to baseline shortest-latency candidate.
            return cands[0]

        scored.sort(key=lambda x: x[0])
        return scored[0][1]


def run_lowcarb_bgp(
    g: nx.Graph,
    ci: CIProvider,
    router_params: Any,
    pairs: List[Pair],
    hour: int = 0,
    k: int = 8,
    alpha: float = 0.7,
    latency_bound: float = 1.15,
) -> Paths:
    """
    Wrapper function for Low-Carb BGP algorithm.
    """
    ctx = AlgoContext(
        k_paths=k,
        alpha=alpha,
        stretch=latency_bound,
        paper_faithful=True,
        lcb_aggregate="TCIM",
        lcb_transpose_attr="local_pref",
        lcb_base_pref=10_000,
    )
    algo = LowCarbBGP()
    result: Paths = {}
    for src, dst in pairs:
        path = algo.select_path(g, ci, src, dst, hour, ctx)
        result[(src, dst)] = list(path)
    return result
