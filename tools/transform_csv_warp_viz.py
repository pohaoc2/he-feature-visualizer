#!/usr/bin/env python3
"""
transform_csv_warp_viz.py

Transform CSV cell locations with `warp_matrix` from index.json and generate
a 3-panel visualization:
  1) Original locations (MX space),
  2) Rotation/deformation-only transformed locations in MX space,
  3) Overlap (both in MX space).

Transform convention matches debug tools:
  - index.json `warp_matrix` maps H&E px -> MX px
  - CSV Xt/Yt are in um and are converted to MX px via `/ csv_mpp`
  - inverse(warp_matrix) gives MX -> H&E affine; for MX-space visualization we
    apply only its linear part (rotation/scale/shear), centered at point-cloud
    centroid (translation removed).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

# Ensure matplotlib has a writable config/cache directory in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position


def _load_index_payload(index_json: Path) -> dict:
    with index_json.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_warp_matrix(payload: dict, index_json: Path) -> np.ndarray:
    if "warp_matrix" not in payload:
        raise ValueError(f"'warp_matrix' missing in {index_json}")
    m_full = np.asarray(payload["warp_matrix"], dtype=np.float64)
    if m_full.shape != (2, 3):
        raise ValueError(f"warp_matrix must be shape (2, 3), got {m_full.shape}")
    return m_full


def _resolve_scale_he_to_mx(payload: dict) -> float:
    if payload.get("scale_he_to_mx") is not None:
        return float(payload["scale_he_to_mx"])
    he_mpp = payload.get("he_mpp")
    mx_mpp = payload.get("mx_mpp")
    if he_mpp and mx_mpp:
        return float(he_mpp) / float(mx_mpp)
    return 1.0


def _invert_affine_2x3(m: np.ndarray) -> np.ndarray:
    m3 = np.eye(3, dtype=np.float64)
    m3[:2] = m
    m3_inv = np.linalg.inv(m3)
    return m3_inv[:2]


def _transform_points(points_xy: np.ndarray, m: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([points_xy.astype(np.float64), ones], axis=1)
    return (m.astype(np.float64) @ hom.T).T


def _transform_points_linear_centered(
    points_xy: np.ndarray, m: np.ndarray
) -> np.ndarray:
    """Apply only linear part of 2x3 affine, centered on point-cloud centroid."""
    linear = m[:, :2].astype(np.float64)
    center = points_xy.mean(axis=0, keepdims=True).astype(np.float64)
    shifted = points_xy.astype(np.float64) - center
    transformed = (linear @ shifted.T).T + center
    return transformed


def _sample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _scatter_panel(
    ax: plt.Axes,
    points_xy: np.ndarray,
    color: str,
    title: str,
    size: float,
    alpha: float,
) -> None:
    ax.scatter(
        points_xy[:, 0],
        points_xy[:, 1],
        s=size,
        c=color,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    # Image coordinate convention: origin at top-left.
    ax.invert_yaxis()


def _build_figure(
    points_mx: np.ndarray,
    points_rotdef_mx: np.ndarray,
    out_png: Path,
    csv_mpp: float,
    max_plot_points: int,
) -> None:
    mx_plot = _sample_points(points_mx, max_plot_points, seed=0)
    rotdef_plot = _sample_points(points_rotdef_mx, max_plot_points, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    _scatter_panel(
        axes[0],
        mx_plot,
        color="#1f77b4",
        title=f"A. Original locations (MX px)\nfrom Xt/Yt (um) divided by mpp={csv_mpp}",
        size=0.8,
        alpha=0.35,
    )

    _scatter_panel(
        axes[1],
        rotdef_plot,
        color="#ff7f0e",
        title="B. Rot/Deform-only transformed locations (MX px)\nInverse(warp_matrix) linear part, centered",
        size=0.8,
        alpha=0.35,
    )

    axes[2].scatter(
        mx_plot[:, 0],
        mx_plot[:, 1],
        s=0.8,
        c="#1f77b4",
        alpha=0.25,
        linewidths=0,
        label="Original (MX px)",
        rasterized=True,
    )
    axes[2].scatter(
        rotdef_plot[:, 0],
        rotdef_plot[:, 1],
        s=0.8,
        c="#ff7f0e",
        alpha=0.25,
        linewidths=0,
        label="Rot/Deform-only transformed (MX px)",
        rasterized=True,
    )
    axes[2].set_title("C. Overlap (MX frame)")
    axes[2].set_xlabel("X (MX px)")
    axes[2].set_ylabel("Y (MX px)")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True, alpha=0.2, linewidth=0.5)
    axes[2].invert_yaxis()
    axes[2].legend(loc="upper right", markerscale=6, frameon=True)

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform CSV cell locations using warp_matrix and visualize results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", required=True, help="Path to index.json")
    parser.add_argument("--csv", required=True, help="Path to CSV with Xt, Yt (um)")
    parser.add_argument(
        "--csv-mpp",
        type=float,
        default=0.65,
        help="CSV coordinate um/px. Xt/Yt are divided by this to convert to MX px.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/WD-76845-097.transformed.csv",
        help="Output CSV path with added transformed coordinate columns.",
    )
    parser.add_argument(
        "--out-png",
        default="data/WD-76845-097.transform_viz.png",
        help="Output 3-panel visualization PNG path.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=200000,
        help="Maximum number of points to plot (sampling only for visualization).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    index_path = Path(args.index)
    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    payload = _load_index_payload(index_path)
    m_full = _extract_warp_matrix(payload, index_path)
    scale_he_to_mx = _resolve_scale_he_to_mx(payload)
    m_inv = _invert_affine_2x3(m_full)

    df = pd.read_csv(csv_path)
    if "Xt" not in df.columns or "Yt" not in df.columns:
        raise ValueError("CSV must contain 'Xt' and 'Yt' columns.")

    points_um = df[["Xt", "Yt"]].to_numpy(dtype=np.float64)
    points_mx = points_um / float(args.csv_mpp)
    points_he = _transform_points(points_mx, m_inv)
    # Normalize inverse affine from HE px scale back into MX px scale so Panel B/C
    # remain in MX coordinate units.
    m_inv_mx_scale = m_inv.copy()
    m_inv_mx_scale[:, :2] *= scale_he_to_mx
    points_rotdef_mx = _transform_points_linear_centered(points_mx, m_inv_mx_scale)

    out_df = df.copy()
    out_df["Xt_mx_px"] = points_mx[:, 0]
    out_df["Yt_mx_px"] = points_mx[:, 1]
    out_df["Xt_rotdef_mx_px"] = points_rotdef_mx[:, 0]
    out_df["Yt_rotdef_mx_px"] = points_rotdef_mx[:, 1]
    out_df["Xt_he_px"] = points_he[:, 0]
    out_df["Yt_he_px"] = points_he[:, 1]
    out_df.to_csv(out_csv, index=False)

    _build_figure(
        points_mx=points_mx,
        points_rotdef_mx=points_rotdef_mx,
        out_png=out_png,
        csv_mpp=float(args.csv_mpp),
        max_plot_points=int(args.max_plot_points),
    )

    print(f"Input CSV cells: {len(df):,}")
    print(f"Wrote transformed CSV: {out_csv}")
    print(f"Wrote visualization : {out_png}")
    print(
        "MX range: "
        f"x[{points_mx[:, 0].min():.1f}, {points_mx[:, 0].max():.1f}] "
        f"y[{points_mx[:, 1].min():.1f}, {points_mx[:, 1].max():.1f}]"
    )
    print(
        "Rot/Deform-only MX range: "
        f"x[{points_rotdef_mx[:, 0].min():.1f}, {points_rotdef_mx[:, 0].max():.1f}] "
        f"y[{points_rotdef_mx[:, 1].min():.1f}, {points_rotdef_mx[:, 1].max():.1f}]"
    )
    print(f"Scale normalization (HE px -> MX px): {scale_he_to_mx:.6f}")


if __name__ == "__main__":
    main(_parse_args())
