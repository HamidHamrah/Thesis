from __future__ import annotations
from typing import Sequence

import networkx as nx

from .base import RoutingAlgorithm, AlgoContext
from ..ci.base import CIProvider
from ..routing.candidates import k_shortest_paths_latency
from ..metrics.path_cost import compute_path_cost

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

        W = max(1, int(ctx.forecast_window_hours))

        # We don't know the CI provider horizon here, so we just clamp by trying hours
        # (Synthetic provider will raise if out of range; we handle by stopping early).
        def forecast_score(path: Sequence[int]) -> float:
            vals = []
            for h in range(hour, hour + W):
                try:
                    vals.append(compute_path_cost(g, ci, path, hour=h).carbon)
                except Exception:
                    break
            if not vals:
                # fallback: use current hour only
                return compute_path_cost(g, ci, path, hour=hour).carbon
            return float(sum(vals) / len(vals))

        best_path = None
        best_score = float("inf")
        for p in cands:
            s = forecast_score(p)
            if s < best_score:
                best_score = s
                best_path = p

        return best_path
