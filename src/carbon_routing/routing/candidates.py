from __future__ import annotations
from typing import List, Sequence, Callable, Iterable

import networkx as nx

def k_shortest_paths_latency(
    g: nx.Graph,
    src: int,
    dst: int,
    k: int = 10,
) -> List[List[int]]:
    """
    Generate up to k candidate simple paths ordered by total latency_ms.
    Uses NetworkX shortest_simple_paths (Yen-style).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    # shortest_simple_paths yields paths in increasing order of the 'weight'
    gen = nx.shortest_simple_paths(g, src, dst, weight="latency_ms")

    paths: List[List[int]] = []
    for p in gen:
        paths.append(list(p))
        if len(paths) >= k:
            break
    return paths

def path_latency_ms(g: nx.Graph, path: Sequence[int]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += float(g.edges[u, v].get("latency_ms", 0.0))
    return total
