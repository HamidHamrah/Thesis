from __future__ import annotations
import random
from typing import Dict, Tuple

def generate_synthetic_tm(
    n_nodes: int,
    n_demands: int = 300,
    demand_mbps_range: tuple[float, float] = (10.0, 200.0),
    seed: int = 1,
) -> Dict[Tuple[int, int], float]:
    """
    Synthetic traffic matrix: a dict (src,dst)->Mbps.
    Useful for experiments that require demand-weighted evaluation.
    """
    rng = random.Random(seed)
    tm: Dict[Tuple[int, int], float] = {}

    while len(tm) < n_demands:
        s = rng.randrange(0, n_nodes)
        d = rng.randrange(0, n_nodes)
        if s == d:
            continue
        mbps = rng.uniform(*demand_mbps_range)
        tm[(s, d)] = mbps

    return tm
