from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Optional, List

import networkx as nx
from ..ci.base import CIProvider

@dataclass(frozen=True)
class AlgoContext:
    """
    Shared algorithm context.
    Keep it small and explicit so algorithms are easy to explain.
    """
    k_paths: int = 8
    alpha: float = 0.7
    stretch: float = 1.15

class RoutingAlgorithm(ABC):
    """
    Common interface for all algorithms.
    Each algorithm is responsible for picking ONE path for (src,dst,hour).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

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
        ...
