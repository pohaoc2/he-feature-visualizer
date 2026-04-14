"""Generate violin plots and markdown summaries for CRC33 cell characterization."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-cache"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from utils.colormaps import CELL_TYPE_COLORS
from utils.marker_aliases import resolve_first_present_column


HE_MPP = 0.325  # µm/px for CRC33 H&E (from OME-XML PhysicalSizeX)

CELL_TYPES = ["cancer", "immune", "healthy"]
COLORS = {
    ct: "#{:02X}{:02X}{:02X}".format(*CELL_TYPE_COLORS[ct][:3])
    for ct in CELL_TYPES
}
# Diagnostic subset: validates cell type assignments + proliferation state
MARKERS = [
    "Hoechst",    # nuclear baseline (all types)
    "Pan-CK",     # cancer (epithelial)
    "E-cadherin", # cancer (epithelial)
    "CD45",       # immune (pan-leukocyte)
    "CD3e",       # immune (T cell)
    "CD4",        # immune (helper T)
    "CD8a",       # immune (cytotoxic T)
    "CD20",       # immune (B cell)
    "CD68",       # immune (macrophage)
    "Ki67",       # proliferative state
]


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and merge cell assignments with cached shape features."""
    assignments_path = data_dir / "cell_assignments.csv"
    shape_path = data_dir / "cell_shape_features.csv"

    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing: {assignments_path}")
    if not shape_path.exists():
        raise FileNotFoundError(
            f"Missing: {shape_path}. Run compute_shape_features.py first."
        )

    assignments = pd.read_csv(assignments_path)
    shapes = pd.read_csv(shape_path)
    merge_keys = [
        key for key in ("CellID", "PatchID") if key in assignments.columns and key in shapes.columns
    ]
    if not merge_keys:
        raise KeyError("No shared merge keys between assignments and shape features.")

    shape_cols = merge_keys + [
        col for col in ("area_px", "perimeter_px", "circularity") if col in shapes.columns
    ]
    return assignments.merge(shapes.loc[:, shape_cols], on=merge_keys, how="left")


def zscore_markers(df: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """Clip raw values at 99th percentile, then z-score each marker independently."""
    normalized = df.copy()
    for marker in markers:
        if marker not in normalized.columns:
            print(f"WARNING: marker '{marker}' not found in data, skipping.")
            continue
        series = normalized[marker].astype(float)
        upper = float(series.quantile(0.99))
        series = series.clip(upper=upper)
        std = float(series.std())
        if math.isnan(std) or std == 0.0:
            normalized[marker] = 0.0
            continue
        normalized[marker] = (series - float(series.mean())) / std
    return normalized


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Generate cell morphology and marker violin plots."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--save-dir", type=Path, default=None)
    return parser.parse_args()


def _style_violin_parts(parts: dict, cell_types: list[str]) -> None:
    """Apply project colors to violin plot bodies and summary lines."""
    for body, cell_type in zip(parts["bodies"], cell_types):
        body.set_facecolor(COLORS[cell_type])
        body.set_edgecolor(COLORS[cell_type])
        body.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        artist = parts.get(key)
        if artist is not None:
            artist.set_color("black")
            artist.set_linewidth(0.8)


def _violin_values(df: pd.DataFrame, column: str, clip_pct: float = 99.0) -> list[np.ndarray]:
    """Collect one clipped numeric array per canonical cell type.

    Clips at the global clip_pct percentile so violin x-axis reflects
    the bulk distribution rather than extreme outliers.
    """
    all_vals = df[column].dropna().astype(float)
    upper = float(np.percentile(all_vals, clip_pct)) if len(all_vals) > 0 else np.inf
    values: list[np.ndarray] = []
    for cell_type in CELL_TYPES:
        arr = df.loc[df["cell_type"] == cell_type, column].dropna().astype(float).to_numpy()
        clipped = arr[arr <= upper]
        # fall back to full array if clipping removes all data (e.g. tiny test sets)
        values.append(clipped if len(clipped) >= 2 else arr)
    return values


def plot_violin_area(df: pd.DataFrame, save_path: Path) -> None:
    """Save a per-cell-type violin plot for cell area in µm²."""
    df = df.copy()
    df["area_um2"] = df["Area_cellvit_px"] * (HE_MPP ** 2)
    data = _violin_values(df, "area_um2")
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data, positions=range(len(CELL_TYPES)), showmedians=True)
    _style_violin_parts(parts, CELL_TYPES)
    ax.set_xticks(range(len(CELL_TYPES)))
    ax.set_xticklabels(CELL_TYPES)
    ax.set_ylabel("Cell area (µm²)")
    ax.set_title("Cell Area by Type")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_violin_circularity(df: pd.DataFrame, save_path: Path) -> None:
    """Save a per-cell-type violin plot for circularity."""
    data = _violin_values(df, "circularity")
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data, positions=range(len(CELL_TYPES)), showmedians=True)
    _style_violin_parts(parts, CELL_TYPES)
    ax.set_xticks(range(len(CELL_TYPES)))
    ax.set_xticklabels(CELL_TYPES)
    ax.set_ylabel("Circularity (0-1)")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Cell Circularity by Type")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_violin_markers(df: pd.DataFrame, save_path: Path) -> None:
    """Save grouped violins of z-scored marker intensities by cell type."""
    resolved_pairs: list[tuple[str, str]] = []
    for marker in MARKERS:
        column = resolve_first_present_column(df.columns, marker)
        if column is None:
            print(f"WARNING: marker '{marker}' not found in data, skipping.")
            continue
        resolved_pairs.append((marker, column))

    resolved_columns = [column for _, column in resolved_pairs]
    df_z = zscore_markers(df.loc[:, ["cell_type", *resolved_columns]], resolved_columns)

    if not resolved_pairs:
        raise ValueError("No marker columns available to plot.")

    split_idx = math.ceil(len(resolved_pairs) / 2)
    row_groups = [resolved_pairs[:split_idx], resolved_pairs[split_idx:]]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    offsets = [-0.28, 0.0, 0.28]
    for ax, group in zip(axes, row_groups):
        if not group:
            ax.axis("off")
            continue

        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for marker_idx, (label_name, column_name) in enumerate(group):
            center = float(marker_idx)
            for offset, cell_type in zip(offsets, CELL_TYPES):
                values = (
                    df_z.loc[df_z["cell_type"] == cell_type, column_name]
                    .dropna()
                    .astype(float)
                    .to_numpy()
                )
                if len(values) == 0:
                    continue
                parts = ax.violinplot(
                    [values],
                    positions=[center + offset],
                    widths=0.24,
                    showmedians=True,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(COLORS[cell_type])
                    body.set_edgecolor(COLORS[cell_type])
                    body.set_alpha(0.6)
                for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                    artist = parts.get(key)
                    if artist is not None:
                        artist.set_color("black")
                        artist.set_linewidth(0.5)

            tick_positions.append(center)
            tick_labels.append(label_name)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Z-score intensity")
        ax.axhline(0.0, color="#666666", linewidth=0.7, linestyle="--")
        ax.set_xlim(-0.7, max(tick_positions) + 0.7 if tick_positions else 0.7)

    legend_handles = [Patch(facecolor=COLORS[cell_type], label=cell_type) for cell_type in CELL_TYPES]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle("Multiplex Marker Intensities by Cell Type", fontsize=13)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def _ordered_states(df: pd.DataFrame) -> list[str]:
    """Return preferred cell-state display order with extras appended."""
    preferred = ["quiescent", "proliferative", "dead"]
    seen = [state for state in preferred if state in set(df["cell_state"].dropna().astype(str))]
    extras = sorted(
        state
        for state in df["cell_state"].dropna().astype(str).unique().tolist()
        if state not in preferred
    )
    return [*seen, *extras]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a simple markdown table."""
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]
    for row in rows:
        lines.append(f"| {' | '.join(row)} |")
    return lines


def _key_observations(df: pd.DataFrame) -> list[str]:
    """Summarize a few high-signal observations from the merged table."""
    observations: list[str] = []

    cancer = df.loc[df["cell_type"] == "cancer"]
    if len(cancer) > 0 and "cell_state" in cancer.columns:
        prolif_rate = 100.0 * float((cancer["cell_state"] == "proliferative").mean())
        observations.append(f"Cancer proliferation rate: {prolif_rate:.1f}% of cancer cells.")

    available_markers = [marker for marker in MARKERS if resolve_first_present_column(df.columns, marker)]
    if available_markers:
        means = zscore_markers(df.loc[:, ["cell_type", *available_markers]], available_markers)
        for cell_type in CELL_TYPES:
            subset = means.loc[means["cell_type"] == cell_type, available_markers]
            if subset.empty:
                continue
            top_marker = subset.mean().sort_values(ascending=False).index[0]
            observations.append(f"{cell_type.capitalize()} cells show highest relative signal in {top_marker}.")
            break

    return observations


def write_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """Write markdown summary tables and a brief observations section."""
    n_cells = int(len(df))
    n_patches = int(df["PatchID"].nunique())
    states = _ordered_states(df)

    type_rows = []
    for cell_type in CELL_TYPES:
        count = int((df["cell_type"] == cell_type).sum())
        pct = (100.0 * count / n_cells) if n_cells else 0.0
        type_rows.append([cell_type, f"{count:,}", f"{pct:.1f}%"])

    state_rows = []
    for state in states:
        count = int((df["cell_state"] == state).sum())
        pct = (100.0 * count / n_cells) if n_cells else 0.0
        state_rows.append([state, f"{count:,}", f"{pct:.1f}%"])

    count_headers = ["cell_type", *states, "Total"]
    count_rows = []
    for cell_type in CELL_TYPES:
        subset = df.loc[df["cell_type"] == cell_type]
        row_counts = [int((subset["cell_state"] == state).sum()) for state in states]
        count_rows.append([cell_type, *(f"{value:,}" for value in row_counts), f"{len(subset):,}"])
    total_counts = [int((df["cell_state"] == state).sum()) for state in states]
    count_rows.append(["**Total**", *(f"{value:,}" for value in total_counts), f"{n_cells:,}"])

    pct_headers = ["cell_type", *(f"{state} %" for state in states)]
    pct_rows = []
    for cell_type in CELL_TYPES:
        subset = df.loc[df["cell_type"] == cell_type]
        denom = max(len(subset), 1)
        pct_rows.append(
            [
                cell_type,
                *(
                    f"{100.0 * float((subset['cell_state'] == state).sum()) / denom:.1f}%"
                    for state in states
                ),
            ]
        )

    lines = [
        f"# CRC33 Cell Summary (n = {n_cells:,} cells, {n_patches:,} patches)",
        "",
        "## Cell Type Distribution",
        *_markdown_table(["Type", "N", "%"], type_rows),
        "",
        "## Cell State Distribution",
        *_markdown_table(["State", "N", "%"], state_rows),
        "",
        "## Cell Type x State - Counts",
        *_markdown_table(count_headers, count_rows),
        "",
        "## Cell Type x State - Row %",
        *_markdown_table(pct_headers, pct_rows),
    ]

    observations = _key_observations(df)
    if observations:
        lines.extend(["", "## Key Observations", ""])
        lines.extend([f"- {item}" for item in observations])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) if args.save_dir is not None else data_dir / "cell_vis"
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)
    plot_violin_area(df, save_dir / "violin_area.png")
    plot_violin_circularity(df, save_dir / "violin_circularity.png")
    plot_violin_markers(df, save_dir / "violin_markers.png")
    write_summary_md(df, data_dir / "cell_summary.md")
    print("Done.")


if __name__ == "__main__":
    main()
