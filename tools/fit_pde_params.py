#!/usr/bin/env python3
"""Fit PDE diffusion coefficient D from Ki67-vs-distance validation curve.

Uses the global Ki67-vs-vessel-distance profile (produced by
``stages/multiplex_layers --validate-ki67-distance``) as a proxy for oxygen
availability: Ki67 should track oxygenation (both decrease with distance from
vessels), so D is chosen to maximise Pearson correlation between the PDE-
predicted mean O2 per distance bin and the observed mean Ki67 per distance bin.

Outputs the optimal D and prints the suggested CLI flag.

Usage
-----
    python tools/fit_pde_params.py \\
        --processed processed_crc33_crop/ \\
        --metadata-csv data/markers.csv \\
        --n-patches 10

"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.stats

from stages.multiplex_layers_lib.masks import binarize_otsu
from stages.multiplex_layers_lib.pde import (
    build_consumption_map,
    build_vessel_source_map,
    compute_metabolic_demand_map,
    solve_steady_state_diffusion,
)
from stages.multiplex_layers_lib.channels import (
    get_channel_index,
    get_first_matching_channel_index,
)
from utils.normalize import percentile_norm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


# ── Patch loading ─────────────────────────────────────────────────────────────

def _load_patch(
    mx_path: pathlib.Path,
    cd31_idx: int,
    ki67_idx: int,
    cd68_idx: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (vessel_mask, ki67_norm, demand_map, immune_map)."""
    arr = np.load(mx_path)
    cd31_raw = arr[cd31_idx].astype(np.float32)
    ki67_raw = arr[ki67_idx].astype(np.float32)

    vessel_mask = binarize_otsu(percentile_norm(cd31_raw))
    ki67_norm = percentile_norm(ki67_raw)
    demand_map = compute_metabolic_demand_map(ki67_raw)
    immune_map = (
        percentile_norm(arr[cd68_idx].astype(np.float32))
        if cd68_idx is not None else None
    )
    return vessel_mask, ki67_norm, demand_map, immune_map


# ── O2 profile simulation ─────────────────────────────────────────────────────

def _simulate_o2_profile(
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]],
    D: float,
    k_base: float,
    demand_weight: float,
    immune_weight: float,
    mpp: float,
    bin_um: float,
    max_iters: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PDE on all patches; return (distance_um_bins, mean_o2_per_bin)."""
    bin_px = bin_um / mpp
    n_bins = int(np.ceil(500.0 / bin_um))  # up to 500 µm

    o2_sum = np.zeros(n_bins, dtype=np.float64)
    o2_count = np.zeros(n_bins, dtype=np.int64)

    for vessel_mask, _, demand_map, immune_map in patches:
        source = build_vessel_source_map(vessel_mask)
        k_map = build_consumption_map(
            demand_map,
            base_rate=k_base,
            demand_weight=demand_weight,
            immune_map=immune_map,
            immune_weight=immune_weight,
        )
        u = solve_steady_state_diffusion(
            source, k_map, diffusion=D, max_iters=max_iters, tol=tol
        )
        dist_px = scipy.ndimage.distance_transform_edt(~vessel_mask)
        bin_idx = np.clip((dist_px / bin_px).astype(int), 0, n_bins - 1).ravel()
        u_flat = u.ravel()
        o2_sum += np.bincount(bin_idx, weights=u_flat, minlength=n_bins)
        o2_count += np.bincount(bin_idx, minlength=n_bins)

    valid = o2_count > 0
    mean_o2 = np.where(valid, o2_sum / np.maximum(o2_count, 1), np.nan)
    distance_um = (np.arange(n_bins) + 0.5) * bin_um
    return distance_um, mean_o2


# ── Correlation objective ─────────────────────────────────────────────────────

def _correlation(
    log_D: float,
    patches: list,
    ki67_df: pd.DataFrame,
    k_base: float,
    demand_weight: float,
    immune_weight: float,
    mpp: float,
    bin_um: float,
    max_iters: int,
    tol: float,
) -> float:
    """Return negative Pearson r between predicted O2 and observed Ki67 profile."""
    D = float(np.exp(log_D))
    dist_um, mean_o2 = _simulate_o2_profile(
        patches, D, k_base, demand_weight, immune_weight, mpp, bin_um, max_iters, tol
    )

    # Align bins with ki67_df distance_um
    ki67_interp = np.interp(dist_um, ki67_df["distance_um"], ki67_df["ki67_mean"],
                            left=np.nan, right=np.nan)

    valid = np.isfinite(mean_o2) & np.isfinite(ki67_interp)
    if valid.sum() < 3:
        return 0.0
    r, _ = scipy.stats.pearsonr(mean_o2[valid], ki67_interp[valid])
    return float(r)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit PDE diffusion coefficient D from global Ki67-vs-distance curve. "
            "Searches log-space D values; picks the one whose predicted O2 profile "
            "best correlates with observed Ki67 per distance bin."
        )
    )
    parser.add_argument("--processed", required=True,
                        help="Processed directory (contains multiplex/, vasculature_mask/, "
                             "validation/ki67_vs_distance.csv, index.json).")
    parser.add_argument("--metadata-csv", required=True,
                        help="Channel metadata CSV.")
    parser.add_argument("--channels", nargs="+", default=None,
                        help="Channel names in .npy order. Default: from index.json.")
    parser.add_argument("--n-patches", type=int, default=20,
                        help="Number of patches to sample for fitting (default: 20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k-base", type=float, default=0.1,
                        help="Base consumption rate to use during search (default: 0.1).")
    parser.add_argument("--demand-weight", type=float, default=0.3,
                        help="Ki67 demand weight (default: 0.3).")
    parser.add_argument("--immune-weight", type=float, default=0.1,
                        help="CD68 immune weight (default: 0.1).")
    parser.add_argument("--bin-um", type=float, default=10.0,
                        help="Distance bin width in µm (default: 10).")
    parser.add_argument("--d-min", type=float, default=1.0,
                        help="Minimum D to search (default: 1).")
    parser.add_argument("--d-max", type=float, default=1e6,
                        help="Maximum D to search (default: 1e6).")
    parser.add_argument("--n-steps", type=int, default=25,
                        help="Number of log-spaced D values to evaluate (default: 25).")
    parser.add_argument("--max-iters", type=int, default=1000,
                        help="PDE solver iterations per candidate D (default: 1000).")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="PDE convergence tolerance (default: 1e-3).")
    args = parser.parse_args()

    processed_dir = pathlib.Path(args.processed)
    val_csv = processed_dir / "validation" / "ki67_vs_distance.csv"
    index_path = processed_dir / "index.json"
    mx_dir = processed_dir / "multiplex"

    if not val_csv.exists():
        raise FileNotFoundError(
            f"Ki67 validation CSV not found: {val_csv}\n"
            "Run: python -m stages.multiplex_layers ... --validate-ki67-distance"
        )

    # Load index and mpp
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)
    mpp: float = index.get("mx_mpp") or index.get("he_mpp") or 1.0
    channels = args.channels or index.get("channels", [])
    log.info("mpp=%.4f, %d channels", mpp, len(channels))

    # Physical calibration reference
    L_o2_px = 160.0 / mpp
    L_glc_px = 450.0 / mpp
    D_phys_o2 = L_o2_px ** 2 * args.k_base
    D_phys_glc = L_glc_px ** 2 * args.k_base
    log.info("Physical calibration: D_O2=%.1f  D_glucose=%.1f  (k_base=%.4f, mpp=%.4f)",
             D_phys_o2, D_phys_glc, args.k_base, mpp)

    # Resolve channel indices
    cd31_idx = get_channel_index(channels, "CD31")
    ki67_idx = get_channel_index(channels, "Ki67")
    cd68_idx = get_first_matching_channel_index(channels, ["CD68"])

    # Load Ki67 validation curve
    ki67_df = pd.read_csv(val_csv).dropna(subset=["ki67_mean"])
    ki67_df = ki67_df[ki67_df["pixel_count"] > 50]  # skip low-count bins
    if ki67_df.empty:
        raise RuntimeError("Ki67 validation CSV is empty or all bins have < 50 pixels.")
    log.info("Ki67 curve: %d bins, distance %.1f–%.1f µm",
             len(ki67_df), ki67_df["distance_um"].min(), ki67_df["distance_um"].max())

    # Sample patches
    mx_files = sorted(mx_dir.glob("*.npy"))
    rng = random.Random(args.seed)
    selected = rng.sample(mx_files, min(args.n_patches, len(mx_files)))
    log.info("Loading %d patches …", len(selected))
    patches = [_load_patch(p, cd31_idx, ki67_idx, cd68_idx) for p in selected]

    # Skip patches with empty vessel masks
    patches = [(vm, ki, dm, im) for vm, ki, dm, im in patches if np.any(vm)]
    log.info("  %d patches with valid vessel masks.", len(patches))
    if not patches:
        raise RuntimeError("No patches with non-empty vessel masks.")

    # Grid search over D in log space
    D_candidates = np.logspace(np.log10(args.d_min), np.log10(args.d_max), args.n_steps)
    log.info("Searching %d D values from %.1f to %.1e …", args.n_steps, args.d_min, args.d_max)

    results = []
    for D in D_candidates:
        r = _correlation(
            np.log(D), patches, ki67_df,
            args.k_base, args.demand_weight, args.immune_weight,
            mpp, args.bin_um, args.max_iters, args.tol,
        )
        results.append((D, r))
        log.info("  D=%10.1f  r=%.4f", D, r)

    results.sort(key=lambda x: -x[1])
    D_opt, r_opt = results[0]

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Physical calibration (Krogh 160 µm):  D = {D_phys_o2:.1f}")
    print(f"  Physical calibration (Krogh 450 µm):  D = {D_phys_glc:.1f}")
    print(f"  Ki67-fitted D (oxygen):                D = {D_opt:.1f}  (r = {r_opt:.4f})")
    print()
    print("Suggested CLI (physically calibrated, recommended):")
    print(f"  --oxygen-model pde --oxygen-krogh-um 160")
    print(f"  --glucose-model pde --glucose-krogh-um 450")
    print(f"  --oxygen-consumption-base {args.k_base}")
    print(f"  --pde-max-iters 2000")
    print()
    print("Suggested CLI (Ki67-fitted, data-driven):")
    print(f"  --oxygen-model pde --oxygen-pde-diffusion {D_opt:.1f}")
    print(f"  --oxygen-consumption-base {args.k_base}")
    print(f"  --pde-max-iters 2000")
    print("=" * 60)

    # Save CSV of the search
    out_csv = processed_dir / "validation" / "pde_d_fit.csv"
    pd.DataFrame(results, columns=["D", "pearson_r"]).to_csv(out_csv, index=False)
    log.info("Search results saved to %s", out_csv)


if __name__ == "__main__":
    main()
