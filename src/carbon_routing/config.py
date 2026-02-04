from dataclasses import dataclass

@dataclass(frozen=True)
class TopologyConfig:
    n_as: int = 120 # number of autonomous systems
    m_links_per_new_node: int = 3  # Barabási–Albert parameter
    seed: int = 42 # random seed for topology generation

@dataclass(frozen=True)
class CIConfig:
    horizon_hours: int = 24
    seed: int = 42
    # Synthetic CI parameters (gCO2/kWh)
    base_min: float = 50.0 # minimum CI level
    base_max: float = 700.0 # maximum CI level
    daily_amplitude: float = 0.20   # +/- 20% swing
    noise_sigma: float = 0.05       # 5% multiplicative noise per hour
    n_regions: int = 8              # regional CI correlation groups

@dataclass(frozen=True)
class RunConfig:
    topology: TopologyConfig = TopologyConfig()
    ci: CIConfig = CIConfig()
