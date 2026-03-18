#!/usr/bin/env python3
"""CellViT vs CODEX cell type comparison figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path("/tmp/matplotlib")
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.cell_assignment_reports import load_cell_assignments
from utils.normalize import percentile_norm

CELL_TYPES: tuple[str, str, str] = ("cancer", "immune", "healthy")

TYPE_COLORS = {
    "cancer": "#DC3232",
    "immune": "#3264DC",
    "healthy": "#32B432",
}

MARKER_COLORS = {
    "cancer": "#DC3232",
    "immune": "#3264DC",
    "healthy": "#32B432",
}

CROP_HALF = 28  # 56×56 px crop


# ── Data helpers ──────────────────────────────────────────────────────────────


def _filter_assignable(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows where cell_type is not one of the three canonical classes."""
    return df[df["cell_type"].isin(CELL_TYPES)].copy()


def _compute_codex_margin(df: pd.DataFrame) -> pd.Series:
    """Return per-row CODEX model margin: p_model_winner - max(p_model_others).

    This reflects CODEX classifier confidence only; CellViT contributes to the
    final fused cell_type via probabilistic fusion, not to this margin.
    Caller must ensure df contains only rows where cell_type ∈ {cancer, immune, healthy}.
    """
    margins = []
    for _, row in df.iterrows():
        ct = str(row["cell_type"])
        winner = float(row.get(f"p_model_{ct}", 0.0))
        others = [
            float(row.get(f"p_model_{other}", 0.0))
            for other in CELL_TYPES
            if other != ct
        ]
        margins.append(winner - max(others))
    return pd.Series(margins, index=df.index, dtype=float)


def _select_examples(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """Select one cell per bucket (agree, medium, disagree) with de-duplication.

    df must already have a 'codex_margin' column.
    Returns dict with keys 'agree', 'medium', 'disagree' (value is None if bucket empty).
    """
    used: set[object] = set()
    results: dict[str, pd.Series | None] = {}

    # agree: is_mismatch=False, highest codex_margin
    non_mismatch = df[~df["is_mismatch"]].sort_values("codex_margin", ascending=False)
    results["agree"] = None
    for idx, row in non_mismatch.iterrows():
        if idx not in used:
            used.add(idx)
            results["agree"] = row
            break

    # medium: is_mismatch=False, lowest codex_margin (least confident agreement), not already used
    # Reverse the descending-sorted slice rather than re-sorting ascending.
    results["medium"] = None
    for idx, row in non_mismatch.iloc[::-1].iterrows():
        if idx not in used:
            used.add(idx)
            results["medium"] = row
            break

    # disagree: is_mismatch=True, highest codex_margin
    results["disagree"] = None
    for idx, row in (
        df[df["is_mismatch"]].sort_values("codex_margin", ascending=False).iterrows()
    ):
        if idx not in used:
            used.add(idx)
            results["disagree"] = row
            break

    return results


# ── Rendering helpers ──────────────────────────────────────────────────────────


def _load_he_crop(
    processed_dir: Path,
    patch_id: str,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, float, float, float, float] | None:
    """Load H&E patch and return (crop, cx_in_crop, cy_in_crop, x0_patch, y0_patch).

    The cell centroid is always placed at the exact centre of the crop.
    Regions outside the patch boundary are filled with white (255).
    x0_patch / y0_patch are the crop origin in patch coordinates, used to
    transform CellViT contour points into crop space.
    Returns None if patch image not found.
    """
    he_path = processed_dir / "he" / f"{patch_id}.png"
    if not he_path.exists():
        return None
    img = np.array(Image.open(he_path).convert("RGB"))

    # Pad the full patch by CROP_HALF on every side so any centroid can be
    # centred without clamping.
    img_padded = np.pad(
        img,
        ((CROP_HALF, CROP_HALF), (CROP_HALF, CROP_HALF), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    # Centroid position in padded-image coordinates
    cx_pad = int(round(cx)) + CROP_HALF
    cy_pad = int(round(cy)) + CROP_HALF

    x0p = cx_pad - CROP_HALF
    y0p = cy_pad - CROP_HALF
    crop = img_padded[y0p : y0p + 2 * CROP_HALF, x0p : x0p + 2 * CROP_HALF]

    # Centroid in crop coords (sub-pixel accurate)
    cx_crop = cx - (x0p - CROP_HALF)
    cy_crop = cy - (y0p - CROP_HALF)

    # Crop origin in original patch coordinates (may be negative near edges)
    x0_patch = float(x0p - CROP_HALF)
    y0_patch = float(y0p - CROP_HALF)

    return crop, cx_crop, cy_crop, x0_patch, y0_patch


BUCKET_LABELS = {"agree": "Agree", "medium": "Medium", "disagree": "Disagree"}


def _load_cell_contour(
    processed_dir: Path,
    patch_id: str,
    cell_index: int,
    x0_crop: float,
    y0_crop: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (xs, ys) contour coordinates in crop space from CellViT JSON.

    Contour points are stored as [[x, y], ...] in local patch coordinates.
    x0_crop / y0_crop are the crop origin in patch coordinates.
    Returns None if the JSON or cell is not found.
    """
    json_path = processed_dir / "cellvit" / f"{patch_id}.json"
    if not json_path.exists():
        return None
    with json_path.open() as f:
        data = json.load(f)
    cells = data.get("cells", [])
    if cell_index < 0 or cell_index >= len(cells):
        return None
    contour = cells[cell_index].get("contour", [])
    if not contour:
        return None
    pts = np.array(contour, dtype=float)  # shape (N, 2): [[x, y], ...]
    xs = pts[:, 0] - x0_crop
    ys = pts[:, 1] - y0_crop
    return xs, ys


def _plot_marker_bar(
    ax: plt.Axes,
    row: pd.Series,
    norm_vals: dict[str, np.ndarray],
    markers: dict[str, str],
    row_index: int,
) -> None:
    """Draw three horizontal marker bars plus CellViT/CODEX/Margin text block.

    markers: dict with keys 'cancer_marker', 'immune_marker', 'healthy_marker'.
    norm_vals: dict[marker_name -> float32 array over all cells], indexed by row_index.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bar_y_positions = [0.88, 0.65, 0.42]
    bar_h = 0.15
    role_keys = ["cancer_marker", "immune_marker", "healthy_marker"]
    colors = [
        MARKER_COLORS["cancer"],
        MARKER_COLORS["immune"],
        MARKER_COLORS["healthy"],
    ]

    for bar_y, role, color in zip(bar_y_positions, role_keys, colors):
        marker_name = markers.get(role, "")
        label = marker_name or role

        if marker_name in norm_vals:
            val = float(norm_vals[marker_name][row_index])
            ax.barh(bar_y, val, height=bar_h, color=color, align="center", left=0)
            ax.text(min(val + 0.02, 0.97), bar_y, f"{val:.2f}", va="center", fontsize=6)
        else:
            ax.text(
                0.5, bar_y, "n/a", va="center", ha="center", fontsize=7, color="#888888"
            )

        ax.text(-0.02, bar_y, label, va="center", ha="right", fontsize=6)

    # x-axis label between bottom bar and text block
    ax.text(
        0.5,
        0.29,
        "norm. intensity (1–99th pct.)",
        va="center",
        ha="center",
        fontsize=5,
        color="#555555",
        transform=ax.transAxes,
    )

    # text block — placed at the very bottom of the axes below the bars
    final = str(row.get("cell_type", "?"))
    cvit = str(row.get("cellvit_mapped_type", "?"))
    codex = str(row.get("type_codex", "?"))
    margin = float(row.get("codex_margin", float("nan")))
    # "CODEX conf." = p_winner − max(p_others): how decisively CODEX chose this class
    text = (
        f"Final:   {final}\nCellViT: {cvit}\nCODEX:   {codex}\nCODEX conf: {margin:.2f}"
    )
    ax.text(
        0.5,
        0.01,
        text,
        va="bottom",
        ha="center",
        fontsize=6,
        family="monospace",
        transform=ax.transAxes,
    )


def _plot_confusion_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """3×3 confusion matrix: CellViT rows × CODEX (type_codex) cols.

    Cells show raw count (bold) + row-normalised % below.
    """
    classes = list(CELL_TYPES)
    codex_col = "type_codex"
    counts = np.zeros((3, 3), dtype=int)
    for i, rv in enumerate(classes):
        for j, cv in enumerate(classes):
            counts[i, j] = int(
                ((df["cellvit_mapped_type"] == rv) & (df[codex_col] == cv)).sum()
            )

    row_totals = counts.sum(axis=1, keepdims=True)
    row_norm = np.where(row_totals > 0, counts / row_totals, 0.0)

    im = ax.imshow(counts, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i - 0.12,
                str(counts[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if counts[i, j] > counts.max() * 0.6 else "black",
            )
            ax.text(
                j,
                i + 0.22,
                f"{row_norm[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if counts[i, j] > counts.max() * 0.6 else "black",
            )

    ax.set_xticks(range(3))
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_yticks(range(3))
    ax.set_yticklabels(classes, fontsize=13)
    ax.set_xlabel("CODEX label", fontsize=13)
    ax.set_ylabel("CellViT label", fontsize=13)
    ax.set_title("CellViT vs CODEX Confusion", fontsize=15)


def _plot_summary_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    markers: dict[str, str],
) -> None:
    """Text summary panel: total cells, mismatch rate, per-class counts, markers."""
    ax.axis("off")
    total = len(df)
    n_mismatch = int(df["is_mismatch"].sum())
    rate = n_mismatch / total if total else 0.0

    lines = [
        "Model: CODEX",
        f"Total cells: {total:,}",
        f"Mismatch rate: {rate:.1%}",
        "",
    ]
    for ct in CELL_TYPES:
        sub = df[df["cell_type"] == ct]
        non_mm = sub[~sub["is_mismatch"]]
        n_mm = int(sub["is_mismatch"].sum())
        # Split non-mismatch into agree (high margin ≥0.5) vs medium (low margin <0.5)
        if "codex_margin" in non_mm.columns:
            n_ag = int((non_mm["codex_margin"] >= 0.5).sum())
            n_med = int((non_mm["codex_margin"] < 0.5).sum())
        else:
            n_ag = len(non_mm)
            n_med = 0
        lines.append(f"{ct}: {n_ag} agree / {n_med} medium / {n_mm} disagree")

    lines += [
        "",
        f"Cancer marker:  {markers.get('cancer_marker', 'Pan-CK')} -> cancer",
        f"Immune marker:  {markers.get('immune_marker', 'CD45')} -> immune",
        f"Healthy marker: {markers.get('healthy_marker', 'SMA')} -> healthy",
        "",
        "CODEX conf = p_winner - p_2nd_best",
        "  (CODEX confidence only; CellViT",
        "   contributes to Final via fusion)",
    ]

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        linespacing=1.6,
    )


def _plot_triptych(
    fig: plt.Figure,
    subplot_spec,
    df_class: pd.DataFrame,
    class_name: str,
    panel_label: str,
    processed_dir: Path,
    norm_vals: dict[str, np.ndarray],
    markers: dict[str, str],
    all_indices: pd.Index,
) -> None:
    """Render one class triptych (header + 3 H&E crops + 3 marker bars).

    all_indices: index of the full filtered assignments_df, used to map
    selected row's integer position back to norm_vals array index.
    """
    inner = subplot_spec.subgridspec(
        3,
        3,
        height_ratios=[0.15, 1.1, 0.85],
        hspace=0.06,
        wspace=0.12,
    )

    # Header row (spans all 3 cols)
    ax_header = fig.add_subplot(inner[0, :])
    ax_header.axis("off")
    ax_header.text(
        0.02,
        0.5,
        f"{panel_label}   {class_name.capitalize()}",
        transform=ax_header.transAxes,
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TYPE_COLORS.get(class_name, "#333333"),
    )

    # Select examples
    examples = _select_examples(df_class)
    spine_color = TYPE_COLORS.get(class_name, "#333333")

    for col, bucket in enumerate(("agree", "medium", "disagree")):
        row_data = examples.get(bucket)

        # H&E crop (inner row 1)
        ax_he = fig.add_subplot(inner[1, col])
        ax_he.set_title(BUCKET_LABELS[bucket], fontsize=9, pad=2)
        for spine in ax_he.spines.values():
            spine.set_edgecolor(spine_color)
            spine.set_linewidth(1.8)
        ax_he.set_xticks([])
        ax_he.set_yticks([])

        if row_data is not None:
            result = _load_he_crop(
                processed_dir,
                str(row_data["patch_id"]),
                float(row_data["centroid_x_local"]),
                float(row_data["centroid_y_local"]),
            )
            if result is not None:
                crop, cx_crop, cy_crop, x0_patch, y0_patch = result
                ax_he.imshow(crop, interpolation="nearest")
                # Draw real CellViT contour polygon (optional — requires CellIndex column)
                cell_index_raw = row_data.get("CellIndex")
                if cell_index_raw is not None:
                    try:
                        cell_index = int(cell_index_raw)
                        contour = _load_cell_contour(
                            processed_dir,
                            str(row_data["patch_id"]),
                            cell_index,
                            x0_patch,
                            y0_patch,
                        )
                        if contour is not None:
                            xs, ys = contour
                            ax_he.plot(
                                np.append(xs, xs[0]),
                                np.append(ys, ys[0]),
                                color=spine_color,
                                linewidth=1.5,
                            )
                    except (ValueError, TypeError):
                        pass
            else:
                ax_he.set_facecolor("#dddddd")
                ax_he.text(
                    0.5,
                    0.5,
                    "image\nnot found",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#555555",
                    transform=ax_he.transAxes,
                )
        else:
            ax_he.set_facecolor("#eeeeee")
            ax_he.text(
                0.5,
                0.5,
                "no example\navailable",
                ha="center",
                va="center",
                fontsize=7,
                color="#777777",
                transform=ax_he.transAxes,
            )

        # Marker bar (inner row 2)
        ax_bar = fig.add_subplot(inner[2, col])
        if row_data is not None:
            # Map DataFrame index label to positional index in all_indices
            row_pos = all_indices.get_loc(row_data.name)
            _plot_marker_bar(ax_bar, row_data, norm_vals, markers, row_index=row_pos)
        else:
            ax_bar.axis("off")
            ax_bar.text(
                0.5,
                0.5,
                "—",
                ha="center",
                va="center",
                fontsize=10,
                color="#aaaaaa",
                transform=ax_bar.transAxes,
            )


def build_report_figure(
    assignments_df: pd.DataFrame,
    processed_dir: Path,
    *,
    markers: dict[str, str] | None = None,
) -> plt.Figure:
    """Build the 16×16 CellViT vs CODEX comparison figure."""
    if markers is None:
        markers = {
            "cancer_marker": "Pan-CK",
            "immune_marker": "CD45",
            "healthy_marker": "SMA",
        }

    # Pre-filter to canonical classes only
    df = _filter_assignable(assignments_df).copy()
    df["codex_margin"] = _compute_codex_margin(df)

    # Global percentile normalization for all canonical markers (once, across all rows)
    norm_vals: dict[str, np.ndarray] = {}
    for col_name in markers.values():
        if col_name in df.columns:
            norm_vals[col_name] = percentile_norm(
                df[col_name].values.astype(np.float32)
            )
        # if absent from CSV, omit — _plot_marker_bar renders n/a

    # Outer gridspec: 4 rows × 1 col
    # Do NOT pass hspace — constrained_layout=True manages spacing automatically.
    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.0, 1.0, 1.0])

    # Row 0: top row (confusion + summary)
    top_inner = outer[0].subgridspec(1, 2, width_ratios=[1.2, 0.8])
    ax_cm = fig.add_subplot(top_inner[0, 0])
    ax_summary = fig.add_subplot(top_inner[0, 1])

    _plot_confusion_heatmap(ax_cm, df)
    _plot_summary_panel(ax_summary, df, markers)

    # Rows 1-3: per-class triptychs (B=cancer, C=immune, D=healthy)
    panel_labels = ["B", "C", "D"]
    for row_idx, class_name in enumerate(CELL_TYPES):
        df_class = df[df["cell_type"] == class_name].copy()
        _plot_triptych(
            fig,
            outer[row_idx + 1],
            df_class,
            class_name,
            panel_labels[row_idx],
            processed_dir,
            norm_vals,
            markers,
            df.index,
        )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CellViT vs CODEX cell type comparison figure."
    )
    parser.add_argument(
        "--processed", required=True, help="Processed directory (for he/ patches)."
    )
    parser.add_argument(
        "--assignments-csv",
        default=None,
        help="Path to cell_assignments.csv. Default: <processed>/cell_assignments.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output path prefix. Default: <processed>/codex_comparison",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (default: pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    parser.add_argument("--cancer-marker", default="Pan-CK")
    parser.add_argument("--immune-marker", default="CD45")
    parser.add_argument("--healthy-marker", default="SMA")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    assignments_path = (
        Path(args.assignments_csv)
        if args.assignments_csv
        else processed_dir / "cell_assignments.csv"
    )
    out_prefix = (
        Path(args.out_prefix) if args.out_prefix else processed_dir / "codex_comparison"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments CSV: {assignments_path}")

    markers = {
        "cancer_marker": args.cancer_marker,
        "immune_marker": args.immune_marker,
        "healthy_marker": args.healthy_marker,
    }

    assignments_df = load_cell_assignments(assignments_path)
    fig = build_report_figure(assignments_df, processed_dir, markers=markers)

    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()] or ["png"]
    for fmt in formats:
        out_path = out_prefix.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
