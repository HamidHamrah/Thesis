from __future__ import annotations
import random
from .models import RouterParams

def generate_router_params(n: int, seed: int = 1) -> dict[int, RouterParams]:
    rng = random.Random(seed)
    out: dict[int, RouterParams] = {}
    for asn in range(n):
        # Typical chassis routers: up to ~50 kW is plausible per paper narrative
        ptyp = rng.uniform(5.0, 60.0)

        # E-label quantized to [10..100] as in the paper mapping approach :contentReference[oaicite:5]{index=5}
        elabel_cost = rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        # IncD λ is typically in [0.00001; 0.1] W/Mbps :contentReference[oaicite:6]{index=6}
        incd = rng.uniform(1e-4, 5e-3)  # W/Mbps

        # Idle/max power (for CE). Keep consistent ordering.
        pidle = ptyp * rng.uniform(0.6, 0.95)
        pmax  = ptyp * rng.uniform(1.1, 1.8)

        out[asn] = RouterParams(
            ptyp_kw=ptyp,
            elabel_cost=elabel_cost,
            incd_w_per_mbps=incd,
            pidle_kw=pidle,
            pmax_kw=pmax,
        )
    return out
