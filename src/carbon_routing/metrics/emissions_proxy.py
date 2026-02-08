# carbon_routing/metrics/emissions_proxy.py
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional


@dataclass(frozen=True)
class EmissionsParts:
    node_dynamic: float
    link_enable: float

    @property
    def total(self) -> float:
        return self.node_dynamic + self.link_enable


def _undirected_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def emissions_proxy_eq1(
    paths: Sequence[Sequence[int]],
    demands_mbps: Sequence[float],
    ci_by_node: Dict[int, float],
    incd_lambda_by_node: Dict[int, float],
    beta_per_enabled_link: float = 0.0,
    include_endpoints_in_xn: bool = True,
) -> EmissionsParts:
    """
    Paper-style proxy (Eq.1-like):
      C_tot = sum_n x_n * lambda_n * c_n  +  sum_l beta_l * (1 + c_u + c_v)

    - x_n is approximated as sum of demands over all paths that traverse node n.
    - beta term uses "enabled links" = links that appear in any selected path.
    """
    if len(paths) != len(demands_mbps):
        raise ValueError(f"paths({len(paths)}) and demands({len(demands_mbps)}) must match")

    x_n = defaultdict(float)
    enabled_links = set()

    for p, d in zip(paths, demands_mbps):
        if not p:
            continue

        nodes_iter = p if include_endpoints_in_xn else p[1:-1]
        for n in nodes_iter:
            x_n[n] += float(d)

        for u, v in zip(p[:-1], p[1:]):
            enabled_links.add(_undirected_edge(int(u), int(v)))

    node_dynamic = 0.0
    for n, xn in x_n.items():
        node_dynamic += xn * float(incd_lambda_by_node[n]) * float(ci_by_node[n])

    link_enable = 0.0
    if beta_per_enabled_link > 0.0:
        for u, v in enabled_links:
            link_enable += float(beta_per_enabled_link) * (1.0 + float(ci_by_node[u]) + float(ci_by_node[v]))

    return EmissionsParts(node_dynamic=node_dynamic, link_enable=link_enable)


def mean_emissions_proxy_for_paths(
    paths: Sequence[Sequence[int]],
    ci_by_node: Dict[int, float],
    demands_mbps: Sequence[float],
    incd_lambda_by_node: Dict[int, float],
    beta_per_enabled_link: float = 0.0,
) -> float:
    """
    Returns the TOTAL emissions proxy normalized by number of SD pairs (mean per pair).
    This matches how your other 'mean_*' metrics are reported.
    """
    parts = emissions_proxy_eq1(
        paths=paths,
        demands_mbps=demands_mbps,
        ci_by_node=ci_by_node,
        incd_lambda_by_node=incd_lambda_by_node,
        beta_per_enabled_link=beta_per_enabled_link,
    )
    return parts.total / max(len(paths), 1)
