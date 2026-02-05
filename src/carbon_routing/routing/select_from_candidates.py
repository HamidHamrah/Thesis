from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional
import networkx as nx
from ..ci.base import CIProvider
from ..metrics.path_cost import compute_path_cost

@dataclass(frozen=True)
class CandidateSelectionResult:
    chosen_path: Sequence[int]
    chosen_cost: float
    reason: str

def select_min_carbon(
    g: nx.Graph,
    ci: CIProvider,
    candidates: List[Sequence[int]],
    hour: int,
) -> CandidateSelectionResult:
    best = None
    best_c = float("inf")
    for p in candidates:
        c = compute_path_cost(g, ci, p, hour).carbon
        if c < best_c:
            best_c = c
            best = p
    return CandidateSelectionResult(best, best_c, "min_carbon")

def select_weighted(
    g: nx.Graph,
    ci: CIProvider,
    candidates: List[Sequence[int]],
    hour: int,
    alpha: float,
) -> CandidateSelectionResult:
    best = None
    best_obj = float("inf")
    for p in candidates:
        pc = compute_path_cost(g, ci, p, hour)
        obj = alpha * pc.carbon + (1.0 - alpha) * pc.latency_ms
        if obj < best_obj:
            best_obj = obj
            best = p
    return CandidateSelectionResult(best, best_obj, f"weighted(alpha={alpha})")

def select_min_carbon_with_latency_bound(
    g: nx.Graph,
    ci: CIProvider,
    candidates: List[Sequence[int]],
    hour: int,
    max_latency_ms: float,
) -> CandidateSelectionResult:
    feasible = []
    for p in candidates:
        pc = compute_path_cost(g, ci, p, hour)
        if pc.latency_ms <= max_latency_ms:
            feasible.append((p, pc.carbon))

    if not feasible:
        # fallback: choose the minimum latency candidate
        best = min(candidates, key=lambda p: compute_path_cost(g, ci, p, hour).latency_ms)
        pc = compute_path_cost(g, ci, best, hour)
        return CandidateSelectionResult(best, pc.carbon, "fallback_min_latency_no_feasible")

    best_p, best_c = min(feasible, key=lambda x: x[1])
    return CandidateSelectionResult(best_p, best_c, f"min_carbon_under_latency<= {max_latency_ms:.2f}")
