from __future__ import annotations
from typing import Sequence, Dict, List, Tuple, Any

import networkx as nx

from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency
from ..metrics.cidt import forecast_path_cidt_proxy

Pair = Tuple[int, int]
Paths = Dict[Pair, List[int]]

class CIRoCore(RoutingAlgorithm):
    """
    CIRo-Core (forecast-aware, path-aware selection):

    - Generate K candidate paths (path-aware setting exposes multiple paths)
    - Score each candidate by *forecasted carbon* over a window:
        score(p, t) = average_{h=t..t+W-1} CarbonCost(p, h)
      where CarbonCost is additive across AS hops (our current proxy).

    - Select the path with minimum forecasted score.

    This mirrors CIRo’s key intent: using CI forecasts (day-ahead / time-varying)
    to guide path selection rather than purely instantaneous choices. 
    """

    @property
    def name(self) -> str:
        return "ciro_core"

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

        # Faithful mode: day-ahead CIDT forecast vectors and additive path CIDT proxy.
        # We score each candidate with the first W forecast entries (W <= 24 by default).
        H = max(1, int(ctx.ciro_forecast_horizon_hours))
        W = max(1, int(ctx.forecast_window_hours))
        W = min(W, H)

        def forecast_score(path: Sequence[int]) -> float:
            if ctx.paper_faithful and ctx.router_params is not None:
                vals = forecast_path_cidt_proxy(
                    ci=ci,
                    router_params=ctx.router_params,
                    path=path,
                    start_hour=hour,
                    horizon_hours=H,
                )
            else:
                # Backward-compatible fallback proxy: additive CI sum over path.
                vals = []
                for h in range(hour, hour + H):
                    try:
                        vals.append(float(sum(ci.get_ci(asn, h) for asn in path)))
                    except Exception:
                        break

            if not vals:
                try:
                    return float(sum(ci.get_ci(asn, hour) for asn in path))
                except Exception:
                    return float("inf")
            return float(sum(vals[:W]) / len(vals[:W]))

        best_path = None
        best_score = float("inf")
        for p in cands:
            s = forecast_score(p)
            if s < best_score:
                best_score = s
                best_path = p

        return best_path


def run_ciro_core(
    g: nx.Graph,
    ci: CIProvider,
    router_params: Any,
    pairs: List[Pair],
    hour: int = 0,
    horizon: int = 24,
) -> Paths:
    """
    Wrapper function for CIRo-Core algorithm.
    """
    ctx = AlgoContext(
        k_paths=8,
        forecast_window_hours=horizon,
        ciro_forecast_horizon_hours=24,
        paper_faithful=True,
        g=g,
        ci=ci,
        router_params=router_params,
    )
    algo = CIRoCore()
    result: Paths = {}
    for src, dst in pairs:
        path = algo.select_path(g, ci, src, dst, hour, ctx)
        result[(src, dst)] = list(path)
    return result
