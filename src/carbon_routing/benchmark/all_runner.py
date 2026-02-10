# carbon_routing/benchmark/all_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence, Optional, Protocol, Any

import networkx as nx

from ..metrics.path_cost import path_carbon, path_latency
from ..metrics.utilization import compute_link_utilization


Pair = Tuple[int, int]
Paths = Dict[Pair, List[int]]


@dataclass(frozen=True)
class AlgoHourStats:
    name: str
    mean_carbon: float
    mean_latency_ms: float
    mean_hops: float


@dataclass(frozen=True)
class AlgoDayStats:
    name: str
    mean_carbon_24h: float
    mean_latency_24h: float
    mean_hops_24h: float
    avg_reroute_rate_pct: float


@dataclass(frozen=True)
class Hour0Summary:
    name: str
    mean_carbon: float
    mean_latency_ms: float
    mean_hops: float
    mean_carbon_reduction_vs_base: float
    mean_latency_increase_vs_base: float
    pct_paths_changed_vs_base: float


@dataclass(frozen=True)
class DaySummary:
    name: str
    mean_carbon_24h: float
    mean_latency_24h: float
    mean_hops_24h: float
    avg_reroute_rate_pct: float
    carbon_reduction_vs_base_24h: float
    latency_increase_vs_base_24h: float


class AlgoRunner(Protocol):
    """
    Minimal interface:
    - reset() clears internal state for a new 24h run
    - run_hour(hour) returns chosen paths for all pairs at that hour
    """
    name: str

    def reset(self) -> None: ...
    def run_hour(self, hour: int) -> Paths: ...


def _paths_stats(g: nx.Graph, ci: Any, paths: Paths, hour: int) -> AlgoHourStats:
    carb = []
    lat = []
    hops = []
    for _sd, p in paths.items():
        carb.append(path_carbon(g, ci, p, hour=hour))
        lat.append(path_latency(g, p))
        hops.append(len(p) - 1)
    return AlgoHourStats(
        name="",
        mean_carbon=sum(carb) / len(carb),
        mean_latency_ms=sum(lat) / len(lat),
        mean_hops=sum(hops) / len(hops),
    )


def _reroute_rate(prev: Optional[Paths], cur: Paths) -> float:
    if prev is None:
        return 0.0
    changed = 0
    total = 0
    for k, p in cur.items():
        total += 1
        if prev.get(k) != p:
            changed += 1
    return 100.0 * changed / total if total else 0.0


def _dir_load_from_unit_pairs(paths: Paths) -> Dict[Tuple[int, int], float]:
    load: Dict[Tuple[int, int], float] = {}
    for p in paths.values():
        for a, b in zip(p[:-1], p[1:]):
            load[(a, b)] = load.get((a, b), 0.0) + 1.0
    return load


@dataclass
class BenchmarkResult:
    hour0: List[AlgoHourStats]
    day: List[AlgoDayStats]
    hour0_vs_base: List[dict]
    day_vs_base: List[dict]
    hour0_paths_by_name: Dict[str, Paths] = None
    day_paths_by_name: Dict[str, List[Paths]] = None
    hour0_summary: Dict[str, Hour0Summary] = None
    day_summary: Dict[str, DaySummary] = None


def run_all(
    g: nx.Graph,
    ci: Any,
    pairs: List[Pair],
    runners: List[AlgoRunner],
    hours: int = 24,
    baseline_name: Optional[str] = None,
) -> BenchmarkResult:
    # ---------- Hour 0 table ----------
    hour0_stats: List[AlgoHourStats] = []
    hour0_paths_by_name: Dict[str, Paths] = {}

    for r in runners:
        r.reset()
        paths0 = r.run_hour(0)
        s0 = _paths_stats(g, ci, paths0, hour=0)
        hour0_paths_by_name[r.name] = paths0
        hour0_stats.append(AlgoHourStats(r.name, s0.mean_carbon, s0.mean_latency_ms, s0.mean_hops))

    # baseline reference
    if baseline_name is not None:
        base0 = next(s for s in hour0_stats if s.name == baseline_name)
        base0_paths = hour0_paths_by_name[baseline_name]
    else:
        base0 = hour0_stats[0]
        base0_paths = hour0_paths_by_name[hour0_stats[0].name]


    hour0_vs_base = []
    hour0_summary_map: Dict[str, Hour0Summary] = {}
    for s in hour0_stats:
        paths = hour0_paths_by_name[s.name]
        changed = sum(1 for k in paths if paths[k] != base0_paths.get(k))
        pct_changed = 100.0 * changed / len(paths) if paths else 0.0

        mean_carbon_reduction_vs_base = 100.0 * (base0.mean_carbon - s.mean_carbon) / base0.mean_carbon
        mean_latency_increase_vs_base = 100.0 * (s.mean_latency_ms - base0.mean_latency_ms) / base0.mean_latency_ms

        hour0_vs_base.append({
            "name": s.name,
            "mean_carbon": s.mean_carbon,
            "mean_latency_ms": s.mean_latency_ms,
            "mean_hops": s.mean_hops,
            "mean_carbon_reduction_vs_base(%)": mean_carbon_reduction_vs_base,
            "mean_latency_increase_vs_base(%)": mean_latency_increase_vs_base,
            "pct_paths_changed_vs_base(%)": pct_changed,
        })

        hour0_summary_map[s.name] = Hour0Summary(
            name=s.name,
            mean_carbon=s.mean_carbon,
            mean_latency_ms=s.mean_latency_ms,
            mean_hops=s.mean_hops,
            mean_carbon_reduction_vs_base=mean_carbon_reduction_vs_base,
            mean_latency_increase_vs_base=mean_latency_increase_vs_base,
            pct_paths_changed_vs_base=pct_changed,
        )

    # ---------- 24h stats ----------
    day_stats: List[AlgoDayStats] = []
    day_vs_base: List[dict] = []
    day_paths_by_name: Dict[str, List[Paths]] = {}
    day_summary_map: Dict[str, DaySummary] = {}

    # First compute baseline day to compare
    if baseline_name is None:
        baseline_runner = runners[0]
    else:
        baseline_runner = next(r for r in runners if r.name == baseline_name)
    baseline_runner.reset()
    prev = None
    carb_series = []
    lat_series = []
    hops_series = []
    rr_series = []
    baseline_paths_by_hour = []

    for h in range(hours):
        cur = baseline_runner.run_hour(h)
        baseline_paths_by_hour.append(cur)
        st = _paths_stats(g, ci, cur, hour=h)
        carb_series.append(st.mean_carbon)
        lat_series.append(st.mean_latency_ms)
        hops_series.append(st.mean_hops)
        rr_series.append(_reroute_rate(prev, cur))
        prev = cur

    base_day = AlgoDayStats(
        name=baseline_runner.name,
        mean_carbon_24h=sum(carb_series)/hours,
        mean_latency_24h=sum(lat_series)/hours,
        mean_hops_24h=sum(hops_series)/hours,
        avg_reroute_rate_pct=sum(rr_series[1:])/(hours-1),
    )

    # Now all runners
    for r in runners:
        r.reset()
        prev = None
        carb_series = []
        lat_series = []
        hops_series = []
        rr_series = []
        paths_by_hour = []

        for h in range(hours):
            cur = r.run_hour(h)
            paths_by_hour.append(cur)
            st = _paths_stats(g, ci, cur, hour=h)
            carb_series.append(st.mean_carbon)
            lat_series.append(st.mean_latency_ms)
            hops_series.append(st.mean_hops)
            rr_series.append(_reroute_rate(prev, cur))
            prev = cur

        day_paths_by_name[r.name] = paths_by_hour

        d = AlgoDayStats(
            name=r.name,
            mean_carbon_24h=sum(carb_series)/hours,
            mean_latency_24h=sum(lat_series)/hours,
            mean_hops_24h=sum(hops_series)/hours,
            avg_reroute_rate_pct=sum(rr_series[1:])/(hours-1),
        )
        day_stats.append(d)

        carbon_reduction_vs_base_24h = 100.0 * (base_day.mean_carbon_24h - d.mean_carbon_24h) / base_day.mean_carbon_24h
        latency_increase_vs_base_24h = 100.0 * (d.mean_latency_24h - base_day.mean_latency_24h) / base_day.mean_latency_24h

        day_vs_base.append({
            "name": d.name,
            "mean_carbon_24h": d.mean_carbon_24h,
            "mean_latency_24h": d.mean_latency_24h,
            "mean_hops_24h": d.mean_hops_24h,
            "avg_reroute_rate_pct": d.avg_reroute_rate_pct,
            "carbon_reduction_vs_base_24h(%)": carbon_reduction_vs_base_24h,
            "latency_increase_vs_base_24h(%)": latency_increase_vs_base_24h,
        })

        day_summary_map[d.name] = DaySummary(
            name=d.name,
            mean_carbon_24h=d.mean_carbon_24h,
            mean_latency_24h=d.mean_latency_24h,
            mean_hops_24h=d.mean_hops_24h,
            avg_reroute_rate_pct=d.avg_reroute_rate_pct,
            carbon_reduction_vs_base_24h=carbon_reduction_vs_base_24h,
            latency_increase_vs_base_24h=latency_increase_vs_base_24h,
        )

    return BenchmarkResult(
        hour0=hour0_stats,
        day=day_stats,
        hour0_vs_base=hour0_vs_base,
        day_vs_base=day_vs_base,
        hour0_paths_by_name=hour0_paths_by_name,
        day_paths_by_name=day_paths_by_name,
        hour0_summary=hour0_summary_map,
        day_summary=day_summary_map,
    )
