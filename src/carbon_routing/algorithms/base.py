from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple, List, TYPE_CHECKING

import networkx as nx

from ..ci.base import CIProvider
if TYPE_CHECKING:
    from ..device.models import RouterParams


@dataclass(frozen=True)
class AlgoContext:
    """
    Shared algorithm context (tunable knobs).
    Keep these explicit so each algorithm remains easy to explain and reproduce.
    """
    # Candidate path set size (path-aware setting exposes multiple paths; we sample K).
    k_paths: int = 8

    # For weighted multi-objective algorithms (if/when used).
    alpha: float = 0.7

    # For latency-bounded variants (if/when used): max_latency = stretch * baseline_latency
    stretch: float = 1.15

    # CIRo-style forecast lookahead window (hours). Used by CIRo-Core.
    forecast_window_hours: int = 4

    # Toggle exact paper-faithful logic where available.
    paper_faithful: bool = True

    # CIRo CIDT scoring assumptions.
    ciro_forecast_horizon_hours: int = 24
    ciro_demand_mbps: float = 100.0

    # Low-Carb BGP policy controls (paper-inspired transposition to BGP attrs).
    lcb_aggregate: str = "TCIM"          # TCIM | MCIM | HCIM
    lcb_transpose_attr: str = "local_pref"  # local_pref | weight | med | aspath
    lcb_base_pref: int = 10_000
    lcb_hcim_threshold: float | None = None  # optional route-filter threshold
    lcb_default_cim: float = 1_000.0

    # Optional shared objects for algorithms that need more context (e.g., CE).
    g: nx.Graph | None = None
    ci: CIProvider | None = None
    router_params: Dict[int, "RouterParams"] | None = None
    tm: Dict[Tuple[int, int], float] | None = None


@dataclass(frozen=True)
class AlgorithmResult:
    """
    Generic algorithm output used by non-RoutingAlgorithm helpers (e.g., CE).
    """
    name: str
    paths: Dict[Tuple[int, int], List[int]]


class RoutingAlgorithm(ABC):
    """
    Common interface for all routing/path-selection algorithms.

    Each algorithm selects ONE path for (src, dst) at a given hour.
    Algorithms may:
      - generate candidates internally (recommended for modularity), or
      - use shared candidate generation utilities.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in benchmark outputs."""
        raise NotImplementedError

    @abstractmethod
    def select_path(
        self,
        g: nx.Graph,
        ci: CIProvider,
        src: int,
        dst: int,
        hour: int,
        ctx: AlgoContext,
    ) -> Sequence[int]:
        """
        Select a path (sequence of ASNs) from src to dst at time index 'hour'.

        - g: AS-level topology graph with edge attributes (e.g., latency_ms)
        - ci: carbon-intensity provider (synthetic now, real API later)
        - hour: discrete time index (e.g., 0..23)
        - ctx: shared knobs (k_paths, stretch, forecast window, etc.)
        """
        raise NotImplementedError
