import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


# ----------------------------
# CI Provider Interface
# ----------------------------
class CIProvider:
    """Pluggable interface: Synthetic now, Real API later."""
    def get_ci(self, asn: int, t_hour: int) -> float:
        raise NotImplementedError

    def get_ci_series(self, asn: int) -> List[float]:
        raise NotImplementedError


@dataclass
class SyntheticCIConfig:
    hours: int = 24
    n_regions: int = 8
    # CI range (gCO2/kWh) typical-ish values; synthetic only.
    base_min: float = 50.0
    base_max: float = 700.0
    # Daily swing amplitude as fraction of base
    amp_frac_min: float = 0.05
    amp_frac_max: float = 0.35
    # Noise (gCO2/kWh)
    noise_std: float = 15.0
    # Per-AS bias around its region baseline (gCO2/kWh)
    as_bias_std: float = 25.0
    seed: int = 42


class SyntheticCIProvider(CIProvider):
    """
    Generates hourly CI profiles per region, then assigns each AS to a region
    and adds AS-level bias + noise. Designed to be replaced later by a real API provider.
    """
    def __init__(self, asns: List[int], config: SyntheticCIConfig):
        self.asns = asns
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # Assign each AS to a region (store for later)
        self.as_region: Dict[int, int] = {
            asn: int(self.rng.integers(0, self.cfg.n_regions)) for asn in self.asns
        }

        # Build region baselines (base + sinusoid)
        self.region_profiles: Dict[int, np.ndarray] = self._build_region_profiles()

        # Build AS-specific CI series (region profile + bias + noise)
        self.as_profiles: Dict[int, np.ndarray] = self._build_as_profiles()

    def _build_region_profiles(self) -> Dict[int, np.ndarray]:
        profiles: Dict[int, np.ndarray] = {}
        hours = np.arange(self.cfg.hours)

        for r in range(self.cfg.n_regions):
            base = float(self.rng.uniform(self.cfg.base_min, self.cfg.base_max))
            amp_frac = float(self.rng.uniform(self.cfg.amp_frac_min, self.cfg.amp_frac_max))
            amp = base * amp_frac

            # Random phase shift -> different regions peak at different hours
            phase = float(self.rng.uniform(0, 2 * np.pi))

            # Sinusoidal daily pattern
            profile = base + amp * np.sin(2 * np.pi * hours / self.cfg.hours + phase)

            # Keep CI non-negative
            profile = np.clip(profile, 1.0, None)
            profiles[r] = profile

        return profiles

    def _build_as_profiles(self) -> Dict[int, np.ndarray]:
        profiles: Dict[int, np.ndarray] = {}
        for asn in self.asns:
            region = self.as_region[asn]
            base_profile = self.region_profiles[region].copy()

            # AS-level bias (constant offset)
            bias = float(self.rng.normal(0.0, self.cfg.as_bias_std))

            # Hourly noise
            noise = self.rng.normal(0.0, self.cfg.noise_std, size=self.cfg.hours)

            as_profile = base_profile + bias + noise
            as_profile = np.clip(as_profile, 1.0, None)
            profiles[asn] = as_profile

        return profiles

    def get_ci(self, asn: int, t_hour: int) -> float:
        t = int(t_hour) % self.cfg.hours
        return float(self.as_profiles[asn][t])

    def get_ci_series(self, asn: int) -> List[float]:
        return [float(x) for x in self.as_profiles[asn].tolist()]


# ----------------------------
# Topology Generator
# ----------------------------
def generate_as_topology(n: int, m: int, seed: int) -> nx.Graph:
    """
    Generates a scale-free-ish AS graph (Barabási–Albert).
    n: number of nodes (ASes)
    m: edges to attach from a new node to existing nodes (controls avg degree)
    """
    if n <= m + 1:
        raise ValueError("n must be > m+1 for BA graph.")
    if m < 1:
        raise ValueError("m must be >= 1.")

    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

    # Ensure connected (BA should be connected for typical params, but verify)
    if not nx.is_connected(G):
        # Connect components (rare for BA but safe)
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            a = next(iter(components[i]))
            b = next(iter(components[i + 1]))
            G.add_edge(a, b)

    return G


def assign_link_latency(G: nx.Graph, seed: int) -> None:
    """
    Assign a synthetic latency to each edge (ms).
    For now: random 5–60ms. Later: derive from geography, distance, etc.
    """
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["latency_ms"] = float(rng.uniform(5.0, 60.0))


# ----------------------------
# Sanity checks / prints
# ----------------------------
def summarize_graph(G: nx.Graph) -> None:
    degrees = np.array([d for _, d in G.degree()])
    print("=== TOPOLOGY SUMMARY ===")
    print(f"N (AS nodes): {G.number_of_nodes()}")
    print(f"E (links):    {G.number_of_edges()}")
    print(f"Connected:    {nx.is_connected(G)}")
    print(f"Degree min/mean/max: {degrees.min()} / {degrees.mean():.2f} / {degrees.max()}")
    print("Top-5 nodes by degree:", sorted(G.degree, key=lambda x: x[1], reverse=True)[:5])
    print()


def summarize_ci(ci: SyntheticCIProvider, sample_asns: List[int]) -> None:
    print("=== CI SUMMARY (synthetic) ===")
    # Snapshot at a couple of times to see variation
    for t in [0, 12]:
        vals = np.array([ci.get_ci(asn, t) for asn in ci.asns])
        print(f"t={t:02d}: CI min/mean/max (gCO2/kWh): {vals.min():.1f} / {vals.mean():.1f} / {vals.max():.1f}")

    print("\nSample AS CI series (first 8 hours shown):")
    for asn in sample_asns:
        series = ci.get_ci_series(asn)
        region = ci.as_region[asn]
        preview = ", ".join(f"{x:.1f}" for x in series[:8])
        print(f"  AS{asn:04d} (region {region}): [{preview}, ...]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="Number of AS nodes (80–180 recommended for start)")
    ap.add_argument("--m", type=int, default=3, help="BA attachment parameter (controls avg degree)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--regions", type=int, default=8, help="Number of synthetic CI regions")
    args = ap.parse_args()

    # 1) Generate AS topology
    G = generate_as_topology(n=args.n, m=args.m, seed=args.seed)
    assign_link_latency(G, seed=args.seed + 1)

    # 2) Create synthetic CI provider (API-ready design)
    asns = list(G.nodes())
    ci_cfg = SyntheticCIConfig(n_regions=args.regions, seed=args.seed)
    ci = SyntheticCIProvider(asns=asns, config=ci_cfg)

    # 3) Print sanity checks
    summarize_graph(G)
    summarize_ci(ci, sample_asns=[0, args.n // 3, 2 * args.n // 3])

    # 4) One extra correctness check: latency attribute exists on all edges
    latencies = [G[u][v].get("latency_ms", None) for u, v in G.edges()]
    missing = sum(1 for x in latencies if x is None)
    print("=== EDGE ATTRIBUTE CHECK ===")
    print(f"Edges missing latency_ms: {missing} (expected 0)")
    print(f"Latency ms min/mean/max: {min(latencies):.2f} / {np.mean(latencies):.2f} / {max(latencies):.2f}")


if __name__ == "__main__":
    main()
