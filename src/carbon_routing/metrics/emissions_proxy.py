from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import networkx as nx

@dataclass(frozen=True)
class EmissionsParams:
    # Keep defaults consistent with Step 10.2 / your paper mapping
    # If you already have params in another file, we will reuse them later.
    pass


def mean_emissions_proxy_for_paths(
    g: nx.Graph,
    paths: Sequence[Sequence[int]],
    ci_t: Dict[int, float],
    traffic: Sequence[float],
) -> float:
    """
    Paper-style emissions proxy for a batch of selected paths at a given time t.

    - `ci_t[node]` gives carbon intensity at time t for that node/AS
    - `traffic[i]` is the demand for paths[i] (any units; consistent scaling)
    """
    if not paths:
        return 0.0

    total = 0.0
    total_demand = 0.0

    for p, d in zip(paths, traffic):
        if p is None or len(p) < 2:
            continue
        total_demand += d

        # Very lightweight proxy: sum CI along nodes (or edges) weighted by demand.
        # IMPORTANT: If Step 10.2 uses a different exact proxy, we will align it.
        e = 0.0
        for node in p:
            e += ci_t[node]
        total += d * e

    return total / max(total_demand, 1e-9)
