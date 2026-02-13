from __future__ import annotations

from typing import Dict, Sequence, List

from ..ci.base import CIProvider
from ..device.models import RouterParams


def _safe_capacity_from_power(params: RouterParams) -> float:
    """
    Infer an effective throughput ceiling (Mbps) from power model:
      Pmax ~= Pidle + lambda * U_max
      U_max ~= (Pmax - Pidle) / lambda
    Units:
      P* in kW, lambda in W/Mbps, U in Mbps
    """
    lam = float(params.incd_w_per_mbps)
    if lam <= 0.0:
        return 1.0
    dynamic_kw = max(0.0, float(params.pmax_kw) - float(params.pidle_kw))
    u_max = (dynamic_kw * 1000.0) / lam
    return max(1.0, u_max)


def as_cidt_proxy(
    ci: CIProvider,
    router_params: Dict[int, RouterParams],
    asn: int,
    hour: int,
) -> float:
    """
    Paper-inspired per-AS CIDT proxy in gCO2 / Mbit.
    Includes:
    - marginal term (lambda)
    - amortized term (pidle spread across inferred max load)
    """
    p = router_params[asn]
    c = float(ci.get_ci(asn, hour))

    lam_w_per_mbps = float(p.incd_w_per_mbps)
    u_max_mbps = _safe_capacity_from_power(p)
    pidle_w_per_mbps = (float(p.pidle_kw) * 1000.0) / u_max_mbps

    # Convert W/Mbps -> kW/Mbps and multiply by CI (gCO2/kWh)
    return ((lam_w_per_mbps + pidle_w_per_mbps) / 1000.0) * c


def path_cidt_proxy(
    ci: CIProvider,
    router_params: Dict[int, RouterParams],
    path: Sequence[int],
    hour: int,
) -> float:
    """
    Additive inter-domain CIDT proxy (sum of per-AS hop CIDTs).
    """
    total = 0.0
    for asn in path:
        total += as_cidt_proxy(ci, router_params, asn, hour)
    return total


def forecast_path_cidt_proxy(
    ci: CIProvider,
    router_params: Dict[int, RouterParams],
    path: Sequence[int],
    start_hour: int,
    horizon_hours: int,
) -> List[float]:
    """
    Return forecast vector of CIDT proxy values for [start_hour, start_hour+horizon).
    Values beyond provider horizon are omitted.
    """
    out: List[float] = []
    H = max(1, int(horizon_hours))
    for h in range(start_hour, start_hour + H):
        try:
            out.append(path_cidt_proxy(ci, router_params, path, h))
        except Exception:
            break
    return out
