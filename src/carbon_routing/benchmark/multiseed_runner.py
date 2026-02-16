from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import replace
from datetime import datetime
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from ..ci.synthetic import SyntheticCIProvider
from ..config import RunConfig
from ..device import generate_router_params
from ..metrics.emissions_proxy import mean_emissions_proxy_for_paths
from ..metrics.path_cost import path_carbon, path_latency
from ..topology.generator import generate_as_topology
from .all_runner import run_all
from .registry import (
    BaselineRunner,
    CiroCoreRunner,
    LowCarbBGPRunner,
    CarbonOptimalASPathRunner,
    OspfRunner,
    CRunner,
    CIncDRunner,
    CERunner,
)


Pair = Tuple[int, int]
PerSeedRow = Dict[str, float]
PerAlgoResults = Dict[str, List[PerSeedRow]]
SummaryRow = Dict[str, float]
SummaryTable = Dict[str, SummaryRow]
TimeSeriesSeedRow = Dict[str, float | int | str]
DaySeedRow = Dict[str, float | int | str]


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _std_pop(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = _mean(xs)
    return float(math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)))


def _sample_pairs(g: nx.Graph, n_pairs: int, seed: int) -> List[Pair]:
    rng = random.Random(seed)
    pairs: List[Pair] = []
    while len(pairs) < n_pairs:
        s = rng.randrange(0, g.number_of_nodes())
        d = rng.randrange(0, g.number_of_nodes())
        if s != d:
            pairs.append((s, d))
    return pairs


def _aggregate_paths(
    g: nx.Graph,
    ci: SyntheticCIProvider,
    day_paths: List[Dict[Pair, List[int]]],
) -> Dict[str, float]:
    total_carbon = 0.0
    total_latency = 0.0
    total_hops = 0.0
    n_paths = 0

    for hour, paths_by_pair in enumerate(day_paths):
        for p in paths_by_pair.values():
            total_carbon += path_carbon(g, ci, p, hour=hour)
            total_latency += path_latency(g, p)
            total_hops += float(max(0, len(p) - 1))
            n_paths += 1

    if n_paths == 0:
        return {
            "carbon_total": 0.0,
            "avg_carbon": 0.0,
            "avg_latency": 0.0,
            "avg_hops": 0.0,
        }

    return {
        "carbon_total": float(total_carbon),
        "avg_carbon": float(total_carbon / n_paths),
        "avg_latency": float(total_latency / n_paths),
        "avg_hops": float(total_hops / n_paths),
    }


def _mean_carbon_latency_hops_for_hour(
    g: nx.Graph,
    ci: SyntheticCIProvider,
    paths_by_pair: Dict[Pair, List[int]],
    hour: int,
) -> Tuple[float, float, float]:
    ci_h = {n: ci.get_ci(n, hour) for n in g.nodes}
    carb_sum = 0.0
    lat_sum = 0.0
    hop_sum = 0.0
    npaths = 0
    for path in paths_by_pair.values():
        if not path or len(path) < 2:
            continue
        carb_sum += sum(ci_h[n] for n in path)
        for u, v in zip(path[:-1], path[1:]):
            lat_sum += float(g.edges[u, v].get("latency_ms", 0.0))
        hop_sum += float(len(path) - 1)
        npaths += 1
    if npaths == 0:
        return 0.0, 0.0, 0.0
    return carb_sum / npaths, lat_sum / npaths, hop_sum / npaths


def _reroute_rate_pct(prev_paths_by_pair: Dict[Pair, List[int]] | None, cur_paths_by_pair: Dict[Pair, List[int]]) -> float:
    if prev_paths_by_pair is None:
        return 0.0
    keys = list(cur_paths_by_pair.keys())
    if not keys:
        return 0.0
    changed = 0
    for k in keys:
        if prev_paths_by_pair.get(k) != cur_paths_by_pair.get(k):
            changed += 1
    return 100.0 * changed / len(keys)


def _ordered_path_list(paths_by_pair: Dict[Pair, List[int]], pair_order: List[Pair]) -> List[List[int]]:
    return [paths_by_pair[p] for p in pair_order]


def _collect_seed_family_rows(
    seed: int,
    family: str,
    baseline_name: str,
    g: nx.Graph,
    ci: SyntheticCIProvider,
    router_params: Dict[int, Any],
    day_paths_by_name: Dict[str, List[Dict[Pair, List[int]]]],
    day_summary: Dict[str, Any],
) -> Tuple[List[TimeSeriesSeedRow], List[DaySeedRow]]:
    hours = len(next(iter(day_paths_by_name.values()))) if day_paths_by_name else 0
    pair_order = list(day_paths_by_name[baseline_name][0].keys()) if hours > 0 else []
    rng = random.Random(7)
    traffic = [rng.uniform(0.5, 1.5) for _ in pair_order]
    incd_lambda_by_node = {node: router_params[node].incd_w_per_mbps for node in router_params}

    baseline_emissions_by_hour: List[float] = []
    for h in range(hours):
        ci_h = {n: ci.get_ci(n, h) for n in g.nodes}
        base_h = day_paths_by_name[baseline_name][h]
        base_list = _ordered_path_list(base_h, pair_order)
        baseline_emissions_by_hour.append(
            mean_emissions_proxy_for_paths(base_list, ci_h, traffic, incd_lambda_by_node)
        )

    ts_rows: List[TimeSeriesSeedRow] = []
    emission_sum_by_algo: Dict[str, float] = {name: 0.0 for name in day_paths_by_name.keys()}
    for algo_name, paths_by_hour in day_paths_by_name.items():
        prev = None
        for h in range(hours):
            paths_h = paths_by_hour[h]
            base_h = day_paths_by_name[baseline_name][h]

            mean_c, mean_l, mean_hops = _mean_carbon_latency_hops_for_hour(g, ci, paths_h, h)
            base_c, base_l, _ = _mean_carbon_latency_hops_for_hour(g, ci, base_h, h)
            dC = 100.0 * (base_c - mean_c) / max(base_c, 1e-9)
            dL = 100.0 * (mean_l - base_l) / max(base_l, 1e-9)
            rr = _reroute_rate_pct(prev, paths_h)

            ci_h = {n: ci.get_ci(n, h) for n in g.nodes}
            algo_list = _ordered_path_list(paths_h, pair_order)
            e_algo = mean_emissions_proxy_for_paths(algo_list, ci_h, traffic, incd_lambda_by_node)
            e_base = baseline_emissions_by_hour[h]
            dE = 100.0 * (e_base - e_algo) / max(e_base, 1e-9)

            ts_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "hour": h,
                    "algo": algo_name,
                    "carbon": mean_c,
                    "latency_ms": mean_l,
                    "hops": mean_hops,
                    "reroute_pct": rr,
                    "deltaC_pct": dC,
                    "deltaL_pct": dL,
                    "emissions_proxy": e_algo,
                    "deltaE_pct": dE,
                }
            )
            emission_sum_by_algo[algo_name] += e_algo
            prev = paths_h

    day_rows: List[DaySeedRow] = []
    base_e24 = (sum(baseline_emissions_by_hour) / hours) if hours > 0 else 0.0
    for algo_name, summary in day_summary.items():
        algo_e24 = (emission_sum_by_algo[algo_name] / hours) if hours > 0 else 0.0
        dE24 = 100.0 * (base_e24 - algo_e24) / max(base_e24, 1e-9)
        day_rows.append(
            {
                "seed": seed,
                "family": family,
                "algo": algo_name,
                "deltaC_pct": float(summary.carbon_reduction_vs_base_24h),
                "deltaE_pct": dE24,
            }
        )
    return ts_rows, day_rows


def run_multiseed_evaluation(
    base_cfg: RunConfig,
    r: int = 20,
    base_seed: int = 42,
    hours: int = 24,
    n_pairs: int = 200,
) -> Tuple[List[int], PerAlgoResults, List[TimeSeriesSeedRow], List[DaySeedRow]]:
    """
    Run the benchmark for R independent seeds while keeping per-seed conditions identical
    across all algorithms.
    """
    if r < 1:
        raise ValueError("r must be >= 1")

    seeds = [base_seed + i for i in range(r)]
    results: PerAlgoResults = defaultdict(list)
    timeseries_seed_rows: List[TimeSeriesSeedRow] = []
    day_seed_rows: List[DaySeedRow] = []

    for seed in seeds:
        # Global seeds for reproducibility hygiene.
        random.seed(seed)
        np.random.seed(seed)

        cfg = RunConfig(
            topology=replace(base_cfg.topology, seed=seed),
            ci=replace(base_cfg.ci, seed=seed),
        )

        g = generate_as_topology(cfg.topology)
        ci = SyntheticCIProvider(n_as=cfg.topology.n_as, cfg=cfg.ci)
        router_params = generate_router_params(cfg.topology.n_as, seed=seed)
        pairs = _sample_pairs(g, n_pairs=n_pairs, seed=seed)

        runners_a = [
            BaselineRunner(name="baseline_latency", g=g, pairs=pairs),
            CiroCoreRunner(name="ciro_core", g=g, ci=ci, router_params=router_params, pairs=pairs),
            LowCarbBGPRunner(name="lowcarb_bgp", g=g, ci=ci, router_params=router_params, pairs=pairs),
            CarbonOptimalASPathRunner(
                name="carbon_optimal_as_path", g=g, ci=ci, router_params=router_params, pairs=pairs
            ),
        ]
        runners_b = [
            OspfRunner(name="OSPF", g=g, ci=ci, router_params=router_params, pairs=pairs),
            CRunner(name="C", g=g, ci=ci, router_params=router_params, pairs=pairs),
            CIncDRunner(name="C+IncD", g=g, ci=ci, router_params=router_params, pairs=pairs),
            CERunner(name="CE", g=g, ci=ci, router_params=router_params, pairs=pairs, gamma=1.0),
        ]

        res_a = run_all(
            g=g,
            ci=ci,
            pairs=pairs,
            runners=runners_a,
            hours=hours,
            baseline_name="baseline_latency",
        )
        res_b = run_all(
            g=g,
            ci=ci,
            pairs=pairs,
            runners=runners_b,
            hours=hours,
            baseline_name="OSPF",
        )

        for algo_name, day_paths in res_a.day_paths_by_name.items():
            agg = _aggregate_paths(g, ci, day_paths)
            agg["switch_rate"] = float(res_a.day_summary[algo_name].avg_reroute_rate_pct)
            results[algo_name].append(agg)

        for algo_name, day_paths in res_b.day_paths_by_name.items():
            agg = _aggregate_paths(g, ci, day_paths)
            agg["switch_rate"] = float(res_b.day_summary[algo_name].avg_reroute_rate_pct)
            results[algo_name].append(agg)

        ts_a, day_a = _collect_seed_family_rows(
            seed=seed,
            family="interdomain",
            baseline_name="baseline_latency",
            g=g,
            ci=ci,
            router_params=router_params,
            day_paths_by_name=res_a.day_paths_by_name,
            day_summary=res_a.day_summary,
        )
        ts_b, day_b = _collect_seed_family_rows(
            seed=seed,
            family="ospf_metric",
            baseline_name="OSPF",
            g=g,
            ci=ci,
            router_params=router_params,
            day_paths_by_name=res_b.day_paths_by_name,
            day_summary=res_b.day_summary,
        )
        timeseries_seed_rows.extend(ts_a)
        timeseries_seed_rows.extend(ts_b)
        day_seed_rows.extend(day_a)
        day_seed_rows.extend(day_b)

    return seeds, dict(results), timeseries_seed_rows, day_seed_rows


def summarize_multiseed_results(results: PerAlgoResults) -> SummaryTable:
    out: SummaryTable = {}
    for algo_name, rows in results.items():
        carbon_total = [r["carbon_total"] for r in rows]
        avg_carbon = [r["avg_carbon"] for r in rows]
        avg_latency = [r["avg_latency"] for r in rows]
        avg_hops = [r["avg_hops"] for r in rows]
        switch_rate = [r["switch_rate"] for r in rows]

        out[algo_name] = {
            "carbon_total_mean": _mean(carbon_total),
            "carbon_total_std": _std_pop(carbon_total),
            "avg_carbon_mean": _mean(avg_carbon),
            "avg_carbon_std": _std_pop(avg_carbon),
            "avg_latency_mean": _mean(avg_latency),
            "avg_latency_std": _std_pop(avg_latency),
            "avg_hops_mean": _mean(avg_hops),
            "avg_hops_std": _std_pop(avg_hops),
            "switch_rate_mean": _mean(switch_rate),
            "switch_rate_std": _std_pop(switch_rate),
        }
    return out


def print_multiseed_summary(seeds: List[int], summary: SummaryTable) -> None:
    print("\n=== Multi-seed reproducibility ===")
    print(f"Seeds used ({len(seeds)}): {seeds}")

    print("\nAlgorithm | Carbon (mean ± std) | Latency (mean ± std) | Hops (mean ± std) | Switch (mean ± std)")
    print("-" * 112)
    for algo_name in sorted(summary.keys()):
        s = summary[algo_name]
        print(
            f"{algo_name:22s} | "
            f"{s['avg_carbon_mean']:.2f} ± {s['avg_carbon_std']:.2f} | "
            f"{s['avg_latency_mean']:.2f} ± {s['avg_latency_std']:.2f} | "
            f"{s['avg_hops_mean']:.2f} ± {s['avg_hops_std']:.2f} | "
            f"{s['switch_rate_mean']:.2f} ± {s['switch_rate_std']:.2f}"
        )

    print("\nTotal carbon emissions (mean ± std across seeds)")
    print("-" * 56)
    for algo_name in sorted(summary.keys()):
        s = summary[algo_name]
        print(f"{algo_name:22s} | {s['carbon_total_mean']:.2f} ± {s['carbon_total_std']:.2f}")


def _write_csv(path: Path, rows: List[Dict[str, float | int | str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _savefig(fig: "plt.Figure", outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def _ordered_algorithms(names: List[str]) -> List[str]:
    preferred = [
        "baseline_latency",
        "ciro_core",
        "lowcarb_bgp",
        "carbon_optimal_as_path",
        "OSPF",
        "C",
        "C+IncD",
        "CE",
    ]
    in_preferred = [n for n in preferred if n in names]
    extras = sorted([n for n in names if n not in set(preferred)])
    return in_preferred + extras


def _plot_metric_bar_with_error(
    summary: SummaryTable,
    metric_mean_key: str,
    metric_std_key: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    algos = _ordered_algorithms(list(summary.keys()))
    means = [summary[a][metric_mean_key] for a in algos]
    stds = [summary[a][metric_std_key] for a in algos]

    fig = plt.figure(figsize=(11, 5.5))
    ax = fig.add_subplot(111)
    x = list(range(len(algos)))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, outpath)


def _aggregate_group_mean_std(
    rows: List[Dict[str, float | int | str]],
    group_keys: List[str],
    metric_keys: List[str],
) -> List[Dict[str, float | int | str]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        if key not in grouped:
            grouped[key] = {m: [] for m in metric_keys}
        for m in metric_keys:
            grouped[key][m].append(float(row[m]))

    out: List[Dict[str, float | int | str]] = []
    for key, vals in grouped.items():
        row_out: Dict[str, float | int | str] = {}
        for i, k in enumerate(group_keys):
            row_out[k] = key[i]
        for m in metric_keys:
            row_out[f"{m}_mean"] = _mean(vals[m])
            row_out[f"{m}_std"] = _std_pop(vals[m])
        out.append(row_out)
    return out


def _plot_timeseries_mean_std(
    df: "pd.DataFrame",
    family_name: str,
    metric: str,
    ylabel: str,
    title: str,
    outpath: Path,
    algos_order: List[str],
) -> None:
    d = df[df["family"] == family_name].copy()
    d = d.sort_values(["algo", "hour"])
    available = set(d["algo"].unique())
    algos = [a for a in algos_order if a in available]

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    for algo in algos:
        dd = d[d["algo"] == algo].sort_values("hour")
        x = dd["hour"].to_numpy()
        y = dd[f"{metric}_mean"].to_numpy()
        s = dd[f"{metric}_std"].to_numpy()
        ax.plot(x, y, label=algo)
        ax.fill_between(x, y - s, y + s, alpha=0.2)

    ax.set_xlabel("Hour")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _savefig(fig, outpath)


def _plot_day_bar_mean_std(
    df: "pd.DataFrame",
    family_name: str,
    metric: str,
    ylabel: str,
    title: str,
    outpath: Path,
    algos_order: List[str],
) -> None:
    d = df[df["family"] == family_name].copy()
    available = set(d["algo"].unique())
    algos = [a for a in algos_order if a in available]
    d = d.set_index("algo").reindex(algos).reset_index()

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    x = list(range(len(algos)))
    ax.bar(x, d[f"{metric}_mean"], yerr=d[f"{metric}_std"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, outpath)


def export_multiseed_artifacts(
    seeds: List[int],
    per_seed_results: PerAlgoResults,
    summary: SummaryTable,
    timeseries_seed_rows: List[TimeSeriesSeedRow] | None = None,
    day_seed_rows: List[DaySeedRow] | None = None,
    outdir: Path | None = None,
) -> Path:
    if outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("results") / f"multiseed_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save exact seed list for reproducibility.
    with (outdir / "seeds.txt").open("w") as f:
        f.write(",".join(str(s) for s in seeds) + "\n")

    # Per-seed flattened table: one row per (algorithm, seed).
    per_seed_rows: List[Dict[str, float | int | str]] = []
    for algo_name in sorted(per_seed_results.keys()):
        rows = per_seed_results[algo_name]
        for i, r in enumerate(rows):
            per_seed_rows.append(
                {
                    "algorithm": algo_name,
                    "seed": seeds[i],
                    "carbon_total": r["carbon_total"],
                    "avg_carbon": r["avg_carbon"],
                    "avg_latency": r["avg_latency"],
                    "avg_hops": r["avg_hops"],
                    "switch_rate": r["switch_rate"],
                }
            )
    _write_csv(
        outdir / "multiseed_per_seed.csv",
        per_seed_rows,
        ["algorithm", "seed", "carbon_total", "avg_carbon", "avg_latency", "avg_hops", "switch_rate"],
    )

    # Summary table with mean/std.
    summary_rows: List[Dict[str, float | int | str]] = []
    for algo_name in sorted(summary.keys()):
        s = summary[algo_name]
        summary_rows.append(
            {
                "algorithm": algo_name,
                "carbon_total_mean": s["carbon_total_mean"],
                "carbon_total_std": s["carbon_total_std"],
                "avg_carbon_mean": s["avg_carbon_mean"],
                "avg_carbon_std": s["avg_carbon_std"],
                "avg_latency_mean": s["avg_latency_mean"],
                "avg_latency_std": s["avg_latency_std"],
                "avg_hops_mean": s["avg_hops_mean"],
                "avg_hops_std": s["avg_hops_std"],
                "switch_rate_mean": s["switch_rate_mean"],
                "switch_rate_std": s["switch_rate_std"],
            }
        )
    _write_csv(
        outdir / "multiseed_summary.csv",
        summary_rows,
        [
            "algorithm",
            "carbon_total_mean",
            "carbon_total_std",
            "avg_carbon_mean",
            "avg_carbon_std",
            "avg_latency_mean",
            "avg_latency_std",
            "avg_hops_mean",
            "avg_hops_std",
            "switch_rate_mean",
            "switch_rate_std",
        ],
    )

    # Plot mean +/- std per metric.
    _plot_metric_bar_with_error(
        summary=summary,
        metric_mean_key="avg_carbon_mean",
        metric_std_key="avg_carbon_std",
        ylabel="Average Path Carbon",
        title="Multi-seed Average Path Carbon (mean +/- std)",
        outpath=plots_dir / "avg_carbon_mean_std",
    )
    _plot_metric_bar_with_error(
        summary=summary,
        metric_mean_key="avg_latency_mean",
        metric_std_key="avg_latency_std",
        ylabel="Average Latency (ms)",
        title="Multi-seed Average Latency (mean +/- std)",
        outpath=plots_dir / "avg_latency_mean_std",
    )
    _plot_metric_bar_with_error(
        summary=summary,
        metric_mean_key="avg_hops_mean",
        metric_std_key="avg_hops_std",
        ylabel="Average Hop Count",
        title="Multi-seed Average Hop Count (mean +/- std)",
        outpath=plots_dir / "avg_hops_mean_std",
    )
    _plot_metric_bar_with_error(
        summary=summary,
        metric_mean_key="switch_rate_mean",
        metric_std_key="switch_rate_std",
        ylabel="Switch Rate (%)",
        title="Multi-seed Path Switching Rate (mean +/- std)",
        outpath=plots_dir / "switch_rate_mean_std",
    )
    _plot_metric_bar_with_error(
        summary=summary,
        metric_mean_key="carbon_total_mean",
        metric_std_key="carbon_total_std",
        ylabel="Total Carbon Emissions",
        title="Multi-seed Total Carbon Emissions (mean +/- std)",
        outpath=plots_dir / "carbon_total_mean_std",
    )

    if timeseries_seed_rows:
        ts_agg_rows = _aggregate_group_mean_std(
            rows=timeseries_seed_rows,
            group_keys=["family", "hour", "algo"],
            metric_keys=[
                "carbon",
                "latency_ms",
                "hops",
                "reroute_pct",
                "deltaC_pct",
                "deltaL_pct",
                "emissions_proxy",
                "deltaE_pct",
            ],
        )
        _write_csv(
            outdir / "multiseed_timeseries_per_seed.csv",
            timeseries_seed_rows,
            [
                "seed",
                "family",
                "hour",
                "algo",
                "carbon",
                "latency_ms",
                "hops",
                "reroute_pct",
                "deltaC_pct",
                "deltaL_pct",
                "emissions_proxy",
                "deltaE_pct",
            ],
        )
        _write_csv(
            outdir / "multiseed_timeseries_mean_std.csv",
            ts_agg_rows,
            [
                "family",
                "hour",
                "algo",
                "carbon_mean",
                "carbon_std",
                "latency_ms_mean",
                "latency_ms_std",
                "hops_mean",
                "hops_std",
                "reroute_pct_mean",
                "reroute_pct_std",
                "deltaC_pct_mean",
                "deltaC_pct_std",
                "deltaL_pct_mean",
                "deltaL_pct_std",
                "emissions_proxy_mean",
                "emissions_proxy_std",
                "deltaE_pct_mean",
                "deltaE_pct_std",
            ],
        )
        ts_df = pd.DataFrame(ts_agg_rows)
        interdomain_order = ["baseline_latency", "ciro_core", "lowcarb_bgp", "carbon_optimal_as_path"]
        ospf_order = ["OSPF", "C", "C+IncD", "CE"]

        _plot_timeseries_mean_std(
            ts_df, "interdomain", "carbon", "Mean carbon (arb units)",
            "Inter-domain: Carbon vs time (mean +/- std)",
            plots_dir / "interdomain_carbon_vs_time", interdomain_order
        )
        _plot_timeseries_mean_std(
            ts_df, "interdomain", "latency_ms", "Mean latency (ms)",
            "Inter-domain: Latency vs time (mean +/- std)",
            plots_dir / "interdomain_latency_vs_time", interdomain_order
        )
        _plot_timeseries_mean_std(
            ts_df, "interdomain", "reroute_pct", "Reroute rate (%)",
            "Inter-domain: Reroute rate vs time (mean +/- std)",
            plots_dir / "interdomain_reroute_vs_time", interdomain_order
        )
        _plot_timeseries_mean_std(
            ts_df, "interdomain", "emissions_proxy", "Emissions proxy (paper metric)",
            "Inter-domain: Emissions proxy vs time (mean +/- std)",
            plots_dir / "interdomain_emissions_proxy_vs_time", interdomain_order
        )
        _plot_timeseries_mean_std(
            ts_df, "interdomain", "deltaE_pct", "DeltaE vs baseline (%)",
            "Inter-domain: Emissions reduction (DeltaE) vs time (mean +/- std)",
            plots_dir / "interdomain_deltaE_vs_time", interdomain_order
        )

        _plot_timeseries_mean_std(
            ts_df, "ospf_metric", "carbon", "Mean carbon (arb units)",
            "OSPF-family: Carbon vs time (mean +/- std)",
            plots_dir / "ospf_family_carbon_vs_time", ospf_order
        )
        _plot_timeseries_mean_std(
            ts_df, "ospf_metric", "latency_ms", "Mean latency (ms)",
            "OSPF-family: Latency vs time (mean +/- std)",
            plots_dir / "ospf_family_latency_vs_time", ospf_order
        )
        _plot_timeseries_mean_std(
            ts_df, "ospf_metric", "reroute_pct", "Reroute rate (%)",
            "OSPF-family: Reroute rate vs time (mean +/- std)",
            plots_dir / "ospf_family_reroute_vs_time", ospf_order
        )
        _plot_timeseries_mean_std(
            ts_df, "ospf_metric", "emissions_proxy", "Emissions proxy (paper metric)",
            "OSPF-family: Emissions proxy vs time (mean +/- std)",
            plots_dir / "ospf_family_emissions_proxy_vs_time", ospf_order
        )
        _plot_timeseries_mean_std(
            ts_df, "ospf_metric", "deltaE_pct", "DeltaE vs OSPF (%)",
            "OSPF-family: Emissions reduction (DeltaE) vs time (mean +/- std)",
            plots_dir / "ospf_family_deltaE_vs_time", ospf_order
        )

    if day_seed_rows:
        day_agg_rows = _aggregate_group_mean_std(
            rows=day_seed_rows,
            group_keys=["family", "algo"],
            metric_keys=["deltaC_pct", "deltaE_pct"],
        )
        _write_csv(
            outdir / "multiseed_day_per_seed.csv",
            day_seed_rows,
            ["seed", "family", "algo", "deltaC_pct", "deltaE_pct"],
        )
        _write_csv(
            outdir / "multiseed_day_mean_std.csv",
            day_agg_rows,
            [
                "family",
                "algo",
                "deltaC_pct_mean",
                "deltaC_pct_std",
                "deltaE_pct_mean",
                "deltaE_pct_std",
            ],
        )
        day_df = pd.DataFrame(day_agg_rows)
        interdomain_order = ["baseline_latency", "ciro_core", "lowcarb_bgp", "carbon_optimal_as_path"]
        ospf_order = ["OSPF", "C", "C+IncD", "CE"]

        _plot_day_bar_mean_std(
            day_df, "interdomain", "deltaC_pct", "DeltaC over 24h (%)",
            "Inter-domain: 24h carbon reduction vs baseline (mean +/- std)",
            plots_dir / "interdomain_bar_deltaC_24h", interdomain_order
        )
        _plot_day_bar_mean_std(
            day_df, "interdomain", "deltaE_pct", "DeltaE over 24h (%)",
            "Inter-domain: 24h emissions reduction vs baseline (mean +/- std)",
            plots_dir / "interdomain_bar_deltaE_24h", interdomain_order
        )
        _plot_day_bar_mean_std(
            day_df, "ospf_metric", "deltaC_pct", "DeltaC over 24h (%)",
            "OSPF-family: 24h carbon reduction vs OSPF (mean +/- std)",
            plots_dir / "ospf_family_bar_deltaC_24h", ospf_order
        )
        _plot_day_bar_mean_std(
            day_df, "ospf_metric", "deltaE_pct", "DeltaE over 24h (%)",
            "OSPF-family: 24h emissions reduction vs OSPF (mean +/- std)",
            plots_dir / "ospf_family_bar_deltaE_24h", ospf_order
        )

    return outdir
