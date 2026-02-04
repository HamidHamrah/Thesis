from __future__ import annotations
import random
import sys
sys.path.insert(0, '../../')  # Add project root to path
from src.carbon_routing.config import RunConfig
from src.topology.generator import generate_as_topology
from src.topology.stats import topology_summary, sample_path_exists
from src.ci.synthetic import SyntheticCIProvider

# --- Step 1: Topology and Synthetic CI Generation ---
# Generates an AS topology and synthetic carbon intensity data.
# Prints summaries and sanity checks.
# ------------------------------------------------------    




def run_step1(cfg: RunConfig) -> None:
    # 1) Generate topology
    g = generate_as_topology(cfg.topology)
    summary = topology_summary(g)

    print("=== Step 1: Topology summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if not summary["connected"]:
        print("WARNING: Graph is not fully connected; using largest component for later steps.")
        # For now we just warn; later we can auto-extract GCC.

    # 2) Generate CI time series (synthetic)
    ci = SyntheticCIProvider(n_as=cfg.topology.n_as, cfg=cfg.ci)

    # Print CI sanity checks
    t0 = 0
    t1 = 1 if cfg.ci.horizon_hours > 1 else 0

    all_ci_t0 = [ci.get_ci(asn, t0) for asn in range(cfg.topology.n_as)]
    all_ci_t1 = [ci.get_ci(asn, t1) for asn in range(cfg.topology.n_as)]

    def stats(xs):
        return min(xs), sum(xs) / len(xs), max(xs)

    mn0, av0, mx0 = stats(all_ci_t0)
    mn1, av1, mx1 = stats(all_ci_t1)

    print("\n=== Step 1: Synthetic CI sanity check ===")
    print(f"CI[t={t0}] min/mean/max: {mn0:.2f} / {av0:.2f} / {mx0:.2f} (gCO2/kWh)")
    print(f"CI[t={t1}] min/mean/max: {mn1:.2f} / {av1:.2f} / {mx1:.2f} (gCO2/kWh)")

    # 3) Spot-check a few ASes
    sample_as = [0, cfg.topology.n_as // 2, cfg.topology.n_as - 1]
    print("\n=== Step 1: Sample AS CI series (first 6 hours) ===")
    for asn in sample_as:
        region = ci.get_region(asn)
        series6 = list(ci.get_series(asn))[:6]
        print(f"AS{asn:04d} (region={region}) CI[0..5] = {[round(x,2) for x in series6]}")

    # 4) Connectivity/path existence spot check
    print("\n=== Step 1: Sample path existence checks ===")
    rng = random.Random(cfg.topology.seed)
    pairs_checked = 0
    while pairs_checked < 3:
        src = rng.randrange(0, cfg.topology.n_as)
        dst = rng.randrange(0, cfg.topology.n_as)
        if src == dst:
            continue
        ok = sample_path_exists(g, src, dst)
        print(f"path exists AS{src:04d} -> AS{dst:04d}: {ok}")
        pairs_checked += 1

def main():
    cfg = RunConfig()  # defaults: n_as=120
    run_step1(cfg)

if __name__ == "__main__":
    main()
