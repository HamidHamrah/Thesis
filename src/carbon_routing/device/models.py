from dataclasses import dataclass

@dataclass(frozen=True)
class RouterParams:
    # Static / semi-static router attributes used in Table 1
    ptyp_kw: float          # Typical power (kW)
    elabel_cost: int        # E-label mapped into [10..100] cost-scale :contentReference[oaicite:3]{index=3}
    incd_w_per_mbps: float  # λ (W/Mbps), typical range [1e-5..1e-1] :contentReference[oaicite:4]{index=4}

    # Needed for CE metric (idle + dynamic under utilization)
    pidle_kw: float         # idle power (kW)
    pmax_kw: float          # max power (kW)
