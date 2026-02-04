from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Sequence

# pylint: disable=too-few-public-methods
# Base class for CI providers
# - Synthetic provider now
# - Real CI API provider later (ElectricityMaps/WattTime/etc.)


class CIProvider(ABC):
    """
    Pluggable interface:
    - Synthetic provider now
    - Real CI API provider later (ElectricityMaps/WattTime/etc.)
    """

    @abstractmethod
    def get_ci(self, asn: int, hour: int) -> float:
        """Return CI for AS `asn` at time index `hour` (0..H-1)."""

    @abstractmethod
    def get_series(self, asn: int) -> Sequence[float]:
        """Return full horizon series for an AS."""

    @abstractmethod
    def get_all_series(self) -> Dict[int, Sequence[float]]:
        """Return all AS time series."""
