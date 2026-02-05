from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, List
import numpy as np
from carbon_routing.config import CIConfig
from .base import CIProvider

# --- Synthetic CI Provider ---
# Creates synthetic carbon intensity data with realistic patterns.
# Uses per-AS region mapping to introduce correlation.
# Each region has a base CI level and daily pattern,
# with per-AS noise added.
# Configurable parameters include number of regions,
# base CI range, daily amplitude, and noise level.
# This is useful for testing and simulations.
# ------------------------------------------------------


@dataclass(frozen=True)
class SyntheticCIMapping:
    """
    Assign each AS to a 'region' to create correlation,
    so CI isn't i.i.d. per AS (more realistic).
    """
    as_to_region: Dict[int, int]

class SyntheticCIProvider(CIProvider):
    def __init__(self, n_as: int, cfg: CIConfig):
        self._n_as = n_as
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        self._mapping = self._make_region_mapping()
        self._series = self._generate_series()

    def _make_region_mapping(self) -> SyntheticCIMapping:
        # Deterministic-ish mapping with RNG for realism
        as_to_region = {}
        for asn in range(self._n_as):
            as_to_region[asn] = int(self._rng.integers(0, self._cfg.n_regions))
        return SyntheticCIMapping(as_to_region=as_to_region)

    def _generate_series(self) -> Dict[int, List[float]]:
        H = self._cfg.horizon_hours

        # Region baseline CI (gCO2/kWh)
        region_base = self._rng.uniform(self._cfg.base_min, self._cfg.base_max, size=self._cfg.n_regions)

        # Hourly daily pattern shared per region (sinusoidal)
        hours = np.arange(H)
        daily = np.sin(2 * np.pi * hours / 24.0)  # [-1..1]
        daily_factor = 1.0 + (self._cfg.daily_amplitude * daily)  # [1-amp .. 1+amp]

        series: Dict[int, List[float]] = {}
        for asn in range(self._n_as):
            region = self._mapping.as_to_region[asn]
            base = region_base[region]

            # AS-specific scale: +-10% around region base
            as_scale = float(self._rng.uniform(0.90, 1.10))

            # Multiplicative noise per hour
            noise = self._rng.normal(loc=0.0, scale=self._cfg.noise_sigma, size=H)
            noise_factor = np.clip(1.0 + noise, 0.6, 1.6)

            ci = base * as_scale * daily_factor * noise_factor
            # clamp to reasonable bounds
            ci = np.clip(ci, 0.0, 1200.0)

            series[asn] = [float(x) for x in ci.tolist()]

        return series

    # --- CIProvider interface ---

    def get_ci(self, asn: int, hour: int) -> float:
        if asn not in self._series:
            raise KeyError(f"Unknown AS {asn}")
        H = self._cfg.horizon_hours
        if hour < 0 or hour >= H:
            raise ValueError(f"hour must be in [0,{H-1}]")
        return self._series[asn][hour]

    def get_series(self, asn: int) -> Sequence[float]:
        if asn not in self._series:
            raise KeyError(f"Unknown AS {asn}")
        return self._series[asn]

    def get_all_series(self) -> Dict[int, Sequence[float]]:
        return self._series

    # Useful for debugging
    def get_region(self, asn: int) -> int:
        return self._mapping.as_to_region[asn]
