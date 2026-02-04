from __future__ import annotations
import networkx as nx
# This module provides functions to compute statistics on network topology graphs.
# It uses NetworkX for graph representation and analysis.
# Functions include computing summary statistics and checking for path existence.
# The graphs are assumed to be undirected.
def topology_summary(g: nx.Graph) -> dict:
    n = g.number_of_nodes()
    e = g.number_of_edges()
    is_connected = nx.is_connected(g)
    comp_size = n
    if not is_connected:
        comp_size = len(max(nx.connected_components(g), key=len))

    degrees = [d for _, d in g.degree()]
    deg_min = min(degrees) if degrees else 0
    deg_max = max(degrees) if degrees else 0
    deg_avg = (sum(degrees) / len(degrees)) if degrees else 0.0

    return {
        "nodes": n,
        "edges": e,
        "connected": is_connected,
        "largest_component_size": comp_size,
        "degree_min": deg_min,
        "degree_avg": float(deg_avg),
        "degree_max": deg_max,
    }

def sample_path_exists(g: nx.Graph, src: int, dst: int) -> bool:
    try:
        nx.shortest_path(g, src, dst)
        return True
    except nx.NetworkXNoPath:
        return False
