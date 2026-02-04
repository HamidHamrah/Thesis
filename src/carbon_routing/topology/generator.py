from __future__ import annotations
import networkx as nx
from src.carbon_routing.config import TopologyConfig

def generate_as_topology(cfg: TopologyConfig) -> nx.Graph:
    """
    Generate a synthetic AS-level topology.
    We start with a scale-free-like graph using Barabási–Albert (BA),
    which is a reasonable first proxy for AS degree distribution.
    Nodes are labeled as integers [0..n_as-1].
    """
    if cfg.n_as < 2:
        raise ValueError("n_as must be >= 2") 
    if cfg.m_links_per_new_node < 1: 
        raise ValueError("m_links_per_new_node must be >= 1") # At least one link per new node
    if cfg.m_links_per_new_node >= cfg.n_as:
        raise ValueError("m_links_per_new_node must be < n_as") # Can't have more links than existing nodes

    g = nx.barabasi_albert_graph(
        n=cfg.n_as,
        m=cfg.m_links_per_new_node,
        seed=cfg.seed,
    )

    # Attach basic per-edge latency (ms) placeholder.
    # We'll improve this later; for now we need a stable "performance metric".
    # Keep it deterministic-ish by using node ids (no random yet).
    for u, v in g.edges():
        # Simple heuristic: slightly higher latency on edges involving low-degree nodes
        deg_u = g.degree(u)
        deg_v = g.degree(v)
        g.edges[u, v]["latency_ms"] = float(5 + (10 / max(1, min(deg_u, deg_v))))

    return g
