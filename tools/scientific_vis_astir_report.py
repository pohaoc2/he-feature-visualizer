#!/usr/bin/env python3
"""Sample-level model evidence and assignment comparison report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path("/tmp/matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.cell_assignment_reports import (
    collapse_display_lines,
    load_cell_assignments,
    model_display_name,
    model_label_column,
    select_representative_cells,
)

DEFAULT_SCI_VIS_ROOT = Path(
    "/home/pohaoc2/.claude/plugins/marketplaces/"
    "claude-scientific-skills/scientific-skills/scientific-visualization"
)

CELL_TYPES: tuple[str, str, str] = ("cancer", "immune", "healthy")
TYPE_COLORS = {
    "cancer": "#dc3232",
    "immune": "#3264dc",
    "healthy": "#32b432",
}

EXAMPLE_KINDS: tuple[str, str, str] = ("match", "ambiguous", "disagreement")


def _scientific_style(sci_vis_root: Path):
    scripts_dir = sci_vis_root / "scripts"
    if not scripts_dir.exists():
        return None, f"scientific-vis scripts not found at: {scripts_dir}"

    sys.path.insert(0, str(scripts_dir))
    try:
        from figure_export import save_publication_figure
        from style_presets import apply_publication_style, set_color_palette

        apply_publication_style("default")
        set_color_palette("okabe_ito")
        return save_publication_figure, None
    except Exception as exc:  # pragma: no cover
        return None, f"failed to load scientific-vis scripts: {exc}"


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def _wrapped_lines(text: str, width: int) -> list[str]:
    wrapped = textwrap.fill(text, width=width)
    return wrapped.splitlines() if wrapped else [""]


def _draw_wrapped_block(
    ax: plt.Axes,
    x: float,
    y: float,
    lines: list[str],
    *,
    width: int,
    fontsize: float = 8,
    line_step: float = 0.06,
    paragraph_gap: float = 0.03,
    fontweight: str | None = None,
) -> float:
    current_y = y
    for raw_line in lines:
        wrapped = _wrapped_lines(raw_line, width) if raw_line else [""]
        for line in wrapped:
            ax.text(
                x,
                current_y,
                line,
                va="top",
                fontsize=fontsize,
                fontweight=fontweight,
                transform=ax.transAxes,
            )
            current_y -= line_step
        current_y -= paragraph_gap
    return current_y


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _plot_class_counts(
    ax: plt.Axes,
    assignments_df: pd.DataFrame,
    classifier_used: str,
) -> None:
    labels = ["CellViT", model_display_name(classifier_used), "Final"]
    columns = ["cellvit_mapped_type", "type_astir", "cell_type"]
    x = np.arange(len(CELL_TYPES))
    width = 0.22
    for idx, (label, column) in enumerate(zip(labels, columns, strict=False)):
        counts = (
            assignments_df[column]
            .astype(str)
            .value_counts()
            .reindex(CELL_TYPES, fill_value=0)
            .to_numpy(dtype=float)
        )
        ax.bar(x + (idx - 1) * width, counts, width=width, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_TYPES)
    ax.set_ylabel("Cells")
    ax.set_title("Assignment counts")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(frameon=False, fontsize=7)


def _plot_confusion_heatmap(
    ax: plt.Axes,
    assignments_df: pd.DataFrame,
    classifier_used: str,
) -> None:
    model_name = model_display_name(classifier_used)
    matrix = pd.crosstab(
        assignments_df["cellvit_mapped_type"].astype(str),
        assignments_df["type_astir"].astype(str),
    ).reindex(index=CELL_TYPES, columns=CELL_TYPES, fill_value=0)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues")
    ax.set_xticks(np.arange(len(CELL_TYPES)))
    ax.set_xticklabels(CELL_TYPES, rotation=30, ha="right", fontsize=13)
    ax.set_yticks(np.arange(len(CELL_TYPES)))
    ax.set_yticklabels(CELL_TYPES, fontsize=13)
    ax.set_title(f"CellViT vs {model_name}", fontsize=15, pad=10)
    ax.set_xlabel(model_name, fontsize=13)
    ax.set_ylabel("CellViT", fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix.iat[row, col]), ha="center", va="center", fontsize=14, fontweight="bold")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_probability_distributions(
    ax: plt.Axes,
    assignments_df: pd.DataFrame,
    classifier_used: str,
) -> None:
    data = []
    labels = []
    for cell_type in CELL_TYPES:
        column = f"p_model_{cell_type}"
        subset = assignments_df.loc[
            assignments_df["cell_type"].astype(str) == cell_type,
            column,
        ].astype(float)
        if subset.empty:
            continue
        data.append(subset.to_numpy(dtype=float))
        labels.append(cell_type)
    if not data:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center")
        ax.axis("off")
        return
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylim(0.0, 1.0)
    model_name = model_display_name(classifier_used)
    title = f"{model_name} probability distributions"
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.tick_params(axis="both", labelsize=8)


def _plot_model_subtype_counts(
    ax: plt.Axes,
    assignments_df: pd.DataFrame,
    classifier_used: str,
) -> None:
    model_name = model_display_name(classifier_used)
    column = model_label_column(assignments_df, classifier_used, prefer_fine=True)
    counts = assignments_df[column].astype(str).value_counts()
    if counts.empty:
        ax.text(0.5, 0.5, "No model subtype data", ha="center", va="center")
        ax.axis("off")
        return

    labels = counts.index.astype(str).tolist()
    values = counts.to_numpy(dtype=float)
    colors = [TYPE_COLORS.get(label, plt.get_cmap("tab20")(idx % 20)) for idx, label in enumerate(labels)]
    ax.bar(np.arange(len(labels)), values, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cells")
    ax.tick_params(axis="both", labelsize=8)
    title = f"{model_name} subtype distribution" if column == "type_astir_fine" else f"{model_name} class distribution"
    ax.set_title(title)


def _load_he_crop(processed_dir: Path | None, row: pd.Series, crop_size: int = 24) -> np.ndarray | None:
    if processed_dir is None:
        return None
    patch_path = processed_dir / "he" / f"{row['patch_id']}.png"
    if not patch_path.exists():
        return None
    image = np.array(Image.open(patch_path).convert("RGB"))
    cx = int(round(float(row.get("centroid_x_local", image.shape[1] / 2))))
    cy = int(round(float(row.get("centroid_y_local", image.shape[0] / 2))))
    half = crop_size // 2
    x0 = max(cx - half, 0)
    x1 = min(cx + half, image.shape[1])
    y0 = max(cy - half, 0)
    y1 = min(cy + half, image.shape[0])
    return image[y0:y1, x0:x1]


def _placeholder_crop(size: int = 40) -> np.ndarray:
    return np.full((size, size, 3), 235, dtype=np.uint8)


def _plot_summary_panel(
    ax: plt.Axes,
    assignments_df: pd.DataFrame,
    classifier_used: str,
) -> None:
    ax.axis("off")
    model_name = model_display_name(classifier_used)
    model_col = model_label_column(assignments_df, classifier_used, prefer_fine=True)
    mismatch_rate = float(assignments_df["is_mismatch"].mean()) if not assignments_df.empty else 0.0
    marker_columns = [m for m in ("Pan-CK", "CD45", "SMA") if m in assignments_df.columns]
    lines = [
        f"Model: {model_name}",
        f"Cells: {len(assignments_df)}",
        f"Mismatch rate: {mismatch_rate:.1%}",
        f"Markers: {', '.join(marker_columns) if marker_columns else 'n/a'}",
    ]
    y = _draw_wrapped_block(
        ax,
        0.0,
        0.98,
        lines,
        width=30,
        fontsize=8,
        line_step=0.075,
        paragraph_gap=0.02,
    )
    collapse_lines = collapse_display_lines(model_col)
    if collapse_lines:
        y = _draw_wrapped_block(
            ax,
            0.0,
            y - 0.02,
            ["Fine -> final collapse:"],
            width=30,
            fontsize=8,
            line_step=0.075,
            paragraph_gap=0.0,
        )
        y = _draw_wrapped_block(
            ax,
            0.04,
            y,
            collapse_lines,
            width=28,
            fontsize=8,
            line_step=0.075,
            paragraph_gap=0.0,
        )
    if classifier_used == "rule_fallback":
        _draw_wrapped_block(
            ax,
            0.0,
            y - 0.02,
            ["Astir unavailable; probability panels show rule fallback outputs."],
            width=30,
            fontsize=8,
            line_step=0.075,
            paragraph_gap=0.0,
        )


def _plot_example_triptych(
    fig: plt.Figure,
    subplot_spec,
    selected: pd.DataFrame,
    class_name: str,
    processed_dir: Path | None,
    classifier_used: str,
    panel_label: str,
) -> None:
    class_examples = selected[selected["cell_type"].astype(str) == class_name].copy()
    model_name = model_display_name(classifier_used)
    model_col = model_label_column(selected, classifier_used, prefer_fine=True)

    inner = subplot_spec.subgridspec(
        3,
        3,
        height_ratios=[0.18, 1.05, 0.95],
        hspace=0.08,
        wspace=0.14,
    )
    header_ax = fig.add_subplot(inner[0, :])
    header_ax.axis("off")
    header_ax.set_title(f"{class_name.capitalize()} examples", fontsize=12, pad=2)
    _panel_label(header_ax, panel_label)

    for idx, kind in enumerate(EXAMPLE_KINDS):
        image_ax = fig.add_subplot(inner[1, idx])
        text_ax = fig.add_subplot(inner[2, idx])

        rows = class_examples[class_examples["example_kind"].astype(str) == kind]
        row = rows.iloc[0] if not rows.empty else None

        image_ax.set_title(kind.capitalize(), fontsize=9, pad=2)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color(TYPE_COLORS.get(class_name, "#666666"))

        text_ax.axis("off")
        if row is None:
            image_ax.imshow(_placeholder_crop())
            _draw_wrapped_block(
                text_ax,
                0.0,
                1.0,
                ["No example available"],
                width=18,
                fontsize=7.5,
                line_step=0.16,
                paragraph_gap=0.0,
            )
            continue

        preview = _load_he_crop(processed_dir, row, crop_size=40)
        image_ax.imshow(preview if preview is not None and preview.size > 0 else _placeholder_crop())
        block = [
            f"CellViT: {row.get('cellvit_mapped_type', 'other')}",
            f"{model_name}: {row.get(model_col, row.get('type_astir', 'other'))}",
            f"Final: {row.get('cell_type', class_name)}",
            f"Margin: {float(row.get('final_margin', 0.0)):.2f}",
            (
                f"P=({float(row.get('p_final_cancer', 0.0)):.2f}, "
                f"{float(row.get('p_final_immune', 0.0)):.2f}, "
                f"{float(row.get('p_final_healthy', 0.0)):.2f})"
            ),
        ]
        _draw_wrapped_block(
            text_ax,
            0.0,
            1.0,
            block,
            width=18,
            fontsize=7.3,
            line_step=0.16,
            paragraph_gap=0.01,
        )


def build_report_figure(
    assignments_df: pd.DataFrame,
    summary: dict,
    processed_dir: Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Build the sample-level comparison report and return representative cells."""
    classifier_used = str(summary.get("classifier_used", "unknown"))
    selected = select_representative_cells(assignments_df)

    # Layout: tall top row (confusion + summary) followed by one row per cell class.
    fig = plt.figure(figsize=(16.0, 16.0), constrained_layout=True)
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.5, 1.0, 1.0, 1.0],
    )

    # --- Row 0: confusion matrix (large, left) + summary (right) ---
    top_row = outer[0].subgridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.25)
    confusion_ax = fig.add_subplot(top_row[0, 0])
    _plot_confusion_heatmap(confusion_ax, assignments_df, classifier_used)
    _panel_label(confusion_ax, "A")

    summary_ax = fig.add_subplot(top_row[0, 1])
    _plot_summary_panel(summary_ax, assignments_df, classifier_used)

    # --- Rows 1-3: per-class triptychs (match / ambiguous / disagreement) ---
    _plot_example_triptych(fig, outer[1], selected, "cancer", processed_dir, classifier_used, "B")
    _plot_example_triptych(fig, outer[2], selected, "immune", processed_dir, classifier_used, "C")
    _plot_example_triptych(fig, outer[3], selected, "healthy", processed_dir, classifier_used, "D")

    fig.suptitle(
        (
            f"Model evidence report (mode={classifier_used}, "
            f"n={len(assignments_df)}, mismatch={float(assignments_df['is_mismatch'].mean()):.1%})"
        ),
        fontsize=12,
        y=1.0,
    )
    return fig, selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sample-level model evidence and assignment comparison report."
    )
    parser.add_argument("--processed", required=True, help="Processed directory.")
    parser.add_argument(
        "--assignments-csv",
        default=None,
        help="Path to Stage 3 per-cell assignments CSV. Default: <processed>/cell_assignments.csv",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Path to Stage 3 summary JSON. Default: <processed>/cell_summary.json",
    )
    parser.add_argument(
        "--scientific-vis-root",
        default=str(DEFAULT_SCI_VIS_ROOT),
        help="Path to scientific-visualization skill directory.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output path prefix without extension. Default: <processed>/astir_report.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated formats (e.g. pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    assignments_path = (
        Path(args.assignments_csv)
        if args.assignments_csv
        else processed_dir / "cell_assignments.csv"
    )
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else processed_dir / "cell_summary.json"
    )
    out_prefix = (
        Path(args.out_prefix)
        if args.out_prefix
        else processed_dir / "astir_report"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    assignments_df = load_cell_assignments(assignments_path)
    summary = _load_summary(summary_path)
    fig, _ = build_report_figure(assignments_df, summary, processed_dir)

    save_publication_figure, style_warning = _scientific_style(Path(args.scientific_vis_root))
    if style_warning:
        print(f"Warning: {style_warning}")

    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]
    if not formats:
        formats = ["png"]

    if save_publication_figure is not None:
        save_publication_figure(fig, out_prefix, formats=formats, dpi=int(args.dpi))
    else:
        for fmt in formats:
            out_path = out_prefix.with_suffix(f".{fmt}")
            fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
            print(f"Saved: {out_path}")

    plt.close(fig)
    print(f"Done. mode={summary.get('classifier_used', 'unknown')}, out={out_prefix}")


if __name__ == "__main__":
    main()
