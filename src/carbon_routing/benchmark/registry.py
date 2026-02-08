# carbon_routing/benchmark/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import networkx as nx

from ..metrics.utilization import compute_link_utilization
from .all_runner import Pair, Paths, AlgoRunner


# Import your existing algorithm modules:
from ..algorithms.baseline_latency import run_baseline_latency
from ..algorithms.ciro_core import run_ciro_core
from ..algorithms.lowcarb_bgp import run_lowcarb_bgp
from ..algorithms.ospf_metrics import run_ospf, run_c, run_c_incd
from ..algorithms.ce import run_ce_wrapper


@dataclass
class BaselineRunner(AlgoRunner):
    name: str
    g: nx.Graph
    pairs: List[Pair]

    def reset(self) -> None:
        pass

    def run_hour(self, hour: int) -> Paths:
        return run_baseline_latency(self.g, self.pairs)


@dataclass
class CiroCoreRunner(AlgoRunner):
    name: str
    g: nx.Graph
    ci: Any
    router_params: Any
    pairs: List[Pair]
    horizon: int = 6  # if you used this earlier

    def reset(self) -> None:
        pass

    def run_hour(self, hour: int) -> Paths:
        return run_ciro_core(self.g, self.ci, self.router_params, self.pairs, hour=hour, horizon=self.horizon)


@dataclass
class LowCarbBGPRunner(AlgoRunner):
    name: str
    g: nx.Graph
    ci: Any
    router_params: Any
    pairs: List[Pair]
    k: int = 8
    alpha: float = 0.7
    latency_bound: float = 1.15

    def reset(self) -> None:
        pass

    def run_hour(self, hour: int) -> Paths:
        return run_lowcarb_bgp(
            self.g, self.ci, self.router_params, self.pairs,
            hour=hour, k=self.k, alpha=self.alpha, latency_bound=self.latency_bound
        )


@dataclass
class OspfRunner(AlgoRunner):
    name: str
    g: nx.Graph
    ci: Any
    router_params: Any
    pairs: List[Pair]

    def reset(self) -> None:
        pass

    def run_hour(self, hour: int) -> Paths:
        return run_ospf(self.g, self.ci, self.router_params, self.pairs, hour=hour)


@dataclass
class CRunner(AlgoRunner):
    name: str
    g: nx.Graph
    ci: Any
    router_params: Any
    pairs: List[Pair]

    def reset(self) -> None:
        pass

    def run_hour(self, hour: int) -> Paths:
        return run_c(self.g, self.ci, self.router_params, self.pairs, hour=hour)


@dataclass
class CIncDRunner(AlgoRunner):
    name: str
    g: nx.Graph
    ci: Any
    router_params: Any
    pairs: List[Pair]

    def reset(self) -> None:
        pass

    def run_hour(self, hour: int) -> Paths:
        return run_c_incd(self.g, self.ci, self.router_params, self.pairs, hour=hour)


@dataclass
class CERunner(AlgoRunner):
    name: str
    g: nx.Graph
    ci: Any
    router_params: Any
    pairs: List[Pair]
    gamma: float = 1.0

    prev_util_undir: Dict[Tuple[int, int], float] = None
    prev_paths: Optional[Paths] = None

    def reset(self) -> None:
        self.prev_util_undir = {}
        self.prev_paths = None

    def run_hour(self, hour: int) -> Paths:
        res = run_ce_wrapper(
            g=self.g,
            ci=self.ci,
            router_params=self.router_params,
            pairs=self.pairs,
            hour=hour,
            prev_util_undir=self.prev_util_undir,
            gamma=self.gamma,
        )
        paths = res["paths"] if isinstance(res, dict) else res.paths

        # update utilization for next hour (unit-demand proxy)
        dir_load: Dict[Tuple[int, int], float] = {}
        for p in paths.values():
            for a, b in zip(p[:-1], p[1:]):
                dir_load[(a, b)] = dir_load.get((a, b), 0.0) + 1.0
        self.prev_util_undir = compute_link_utilization(self.g, dir_load)

        return paths
