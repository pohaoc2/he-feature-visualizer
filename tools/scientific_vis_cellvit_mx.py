#!/usr/bin/env python3
"""Publication-style patch report comparing CellViT, model, and final labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep matplotlib cache writable in sandboxed environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path("/tmp/matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.cell_assignment_reports import (
    choose_marker_for_patch,
    load_cell_assignments,
    map_cellvit_type,
    model_display_name,
    model_label_column,
)
from utils.marker_aliases import canonicalize_marker_name, normalize_marker_name
from utils.normalize import percentile_norm

DEFAULT_SCI_VIS_ROOT = Path(
    "/home/pohaoc2/.claude/plugins/marketplaces/"
    "claude-scientific-skills/scientific-skills/scientific-visualization"
)

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "cancer": (220, 50, 50, 200),
    "immune": (50, 100, 220, 200),
    "healthy": (50, 180, 50, 200),
    "other": (150, 150, 150, 120),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (0, 255, 0, 200),
    "quiescent": (100, 149, 237, 200),
    "dead": (139, 0, 139, 200),
    "other": (80, 80, 80, 150),
}

MODEL_FINE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "epithelial": (220, 50, 50, 200),
    "cd4_t": (70, 120, 220, 200),
    "cd8_t": (30, 80, 200, 200),
    "treg": (145, 90, 205, 200),
    "b_cell": (70, 180, 235, 200),
    "macrophage": (255, 150, 40, 200),
    "endothelial": (40, 170, 140, 200),
    "sma_stromal": (80, 180, 80, 200),
    "other": (150, 150, 150, 120),
}


def _load_patch_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        cells = data.get("cells", [])
        if isinstance(cells, list):
            return cells
    if isinstance(data, list):
        return data
    return []


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _composite_rgba_on_rgb(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    if overlay_rgba.ndim != 3 or overlay_rgba.shape[-1] != 4:
        return base_rgb
    ov = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    out = (alpha * ov + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    return out


def _draw_cellvit_contours(he_rgb: np.ndarray, cells: list[dict]) -> np.ndarray:
    out = he_rgb.copy()
    for cell in cells:
        contour = cell.get("contour", [])
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    return out


def _resolve_mx_channel(
    index_json: Path,
    mx_channel: int,
    mx_marker: str | None,
) -> tuple[int, str]:
    resolved_idx = int(mx_channel)
    resolved_name = f"ch{resolved_idx}"

    if not index_json.exists():
        return resolved_idx, resolved_name

    with index_json.open(encoding="utf-8") as fh:
        index_data = json.load(fh)
    channels = [str(x) for x in index_data.get("channels", [])]
    if not channels:
        return resolved_idx, resolved_name

    if mx_marker:
        target_norm = normalize_marker_name(mx_marker)
        target_canon = canonicalize_marker_name(mx_marker)
        for i, name in enumerate(channels):
            name_norm = normalize_marker_name(name)
            name_canon = canonicalize_marker_name(name)
            if (
                name_norm == target_norm
                or name_canon == target_canon
                or target_norm in name_norm
            ):
                resolved_idx = i
                resolved_name = name
                break
        else:
            raise ValueError(
                f"Marker '{mx_marker}' not found in index channels: {channels}"
            )
    else:
        if resolved_idx < 0 or resolved_idx >= len(channels):
            raise ValueError(
                f"--mx-channel {resolved_idx} out of range [0, {len(channels) - 1}]"
            )
        resolved_name = channels[resolved_idx]

    return resolved_idx, resolved_name


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
    except Exception as exc:  # pragma: no cover - defensive fallback
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


def _match_assignments_to_cells(
    cells: list[dict],
    assignments_patch: pd.DataFrame,
    max_centroid_distance: float = 6.0,
) -> list[tuple[dict, dict | None]]:
    rows = assignments_patch.to_dict(orient="records")
    unused = set(range(len(rows)))
    pairs: list[tuple[dict, dict | None]] = []

    for cell in cells:
        centroid = np.asarray(cell.get("centroid", [0.0, 0.0]), dtype=float)
        best_idx: int | None = None
        best_dist = float("inf")
        for idx in unused:
            row = rows[idx]
            dist = float(
                np.hypot(
                    centroid[0] - float(row.get("centroid_x_local", 0.0)),
                    centroid[1] - float(row.get("centroid_y_local", 0.0)),
                )
            )
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None or best_dist > max_centroid_distance:
            pairs.append((cell, None))
            continue

        unused.remove(best_idx)
        pairs.append((cell, rows[best_idx]))

    return pairs


def _render_overlay(
    cell_pairs: list[tuple[dict, dict | None]],
    patch_shape: tuple[int, int],
    label_getter,
    color_map: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    canvas = np.zeros((patch_shape[0], patch_shape[1], 4), dtype=np.uint8)
    for cell, row in cell_pairs:
        contour = cell.get("contour", [])
        if len(contour) < 3:
            continue
        label = str(label_getter(cell, row))
        rgba = color_map.get(label, color_map.get("other", (150, 150, 150, 120)))
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], rgba)
    return canvas


def _build_evidence_panel(
    ax: plt.Axes,
    assignments_patch: pd.DataFrame,
    mx_name: str,
    classifier_used: str,
) -> None:
    ax.axis("off")
    if assignments_patch.empty:
        ax.text(0.0, 1.0, "No matched cells", va="top")
        return

    mismatch_rate = float(assignments_patch["is_mismatch"].mean())
    confidence_counts = (
        assignments_patch["cell_type_confidence"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )
    model_name = model_display_name(classifier_used)
    model_col = model_label_column(assignments_patch, classifier_used, prefer_fine=True)
    lines = [
        f"Mode: {classifier_used}",
        f"Model: {model_name}",
        f"Cells: {len(assignments_patch)}",
        f"Mismatch: {mismatch_rate:.1%}",
        (
            "Confidence: "
            f"H={confidence_counts.get('high', 0)} "
            f"M={confidence_counts.get('medium', 0)} "
            f"L={confidence_counts.get('low', 0)}"
        ),
        f"MX marker: {mx_name}",
        "",
        "Counts",
    ]
    y = 0.98
    for line in lines:
        ax.text(0.0, y, line, va="top", fontsize=8, transform=ax.transAxes)
        y -= 0.08 if line else 0.04

    count_specs = [
        ("CellViT", "cellvit_mapped_type", ["cancer", "immune", "healthy"], CELL_TYPE_COLORS),
        (
            model_name,
            model_col,
            (
                assignments_patch[model_col].astype(str).value_counts().head(4).index.tolist()
                if model_col == "type_astir_fine"
                else ["cancer", "immune", "healthy"]
            ),
            MODEL_FINE_COLORS if model_col == "type_astir_fine" else CELL_TYPE_COLORS,
        ),
        ("Final", "cell_type", ["cancer", "immune", "healthy"], CELL_TYPE_COLORS),
    ]
    for label, column, classes, color_map in count_specs:
        counts = (
            assignments_patch[column]
            .astype(str)
            .value_counts()
            .reindex(classes, fill_value=0)
        )
        ax.text(0.0, y, f"{label}:", va="top", fontsize=8, fontweight="bold", transform=ax.transAxes)
        y -= 0.07
        for cls in classes:
            ax.text(
                0.04,
                y,
                f"{cls}={int(counts[cls])}",
                va="top",
                fontsize=8,
                color=np.array(color_map.get(cls, color_map["other"])[:3], dtype=float) / 255.0,
                transform=ax.transAxes,
            )
            y -= 0.06


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-style figure for one patch showing "
            "H&E, MX marker, CellViT contours, CellViT mapped type, "
            "model type, final type, and state."
        )
    )
    parser.add_argument("--processed", required=True, help="Processed directory.")
    parser.add_argument(
        "--patch",
        required=True,
        help="Patch key as x0_y0, e.g. 256_256.",
    )
    parser.add_argument(
        "--assignments-csv",
        default=None,
        help="Path to Stage 3 per-cell assignments CSV. Default: <processed>/cell_assignments.csv",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to Stage 3 summary JSON. Default: <processed>/cell_summary.json",
    )
    parser.add_argument(
        "--mx-channel",
        type=int,
        default=0,
        help="MX channel index if --mx-marker is not used.",
    )
    parser.add_argument(
        "--mx-marker",
        default=None,
        help="Preferred marker name (resolved from index.json channels).",
    )
    parser.add_argument(
        "--scientific-vis-root",
        default=str(DEFAULT_SCI_VIS_ROOT),
        help="Path to scientific-visualization skill directory.",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output path prefix without extension. Default: <processed>/scientific_vis_<patch>.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated formats (e.g. pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    patch = str(args.patch)
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
        else processed_dir / f"scientific_vis_{patch}"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    he_path = processed_dir / "he" / f"{patch}.png"
    mx_path = processed_dir / "multiplex" / f"{patch}.npy"
    cellvit_path = processed_dir / "cellvit" / f"{patch}.json"
    index_path = processed_dir / "index.json"

    required = [he_path, mx_path, cellvit_path, index_path, assignments_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    summary = _load_summary(summary_path)
    classifier_used = str(summary.get("classifier_used", "unknown"))
    model_name = model_display_name(classifier_used)

    assignments_all = load_cell_assignments(assignments_path)
    assignments_patch = assignments_all[assignments_all["patch_id"].astype(str) == patch].copy()
    if assignments_patch.empty:
        raise ValueError(f"No assignment rows found for patch '{patch}' in {assignments_path}")
    model_col = model_label_column(assignments_patch, classifier_used, prefer_fine=True)
    model_colors = MODEL_FINE_COLORS if model_col == "type_astir_fine" else CELL_TYPE_COLORS
    model_title = f"{model_name} fine type" if model_col == "type_astir_fine" else f"{model_name} top class"

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    channels = [str(x) for x in index_data.get("channels", [])]
    marker_name = args.mx_marker
    if marker_name is None and channels:
        marker_candidates = [name for name in channels if name in assignments_patch.columns]
        marker_name = choose_marker_for_patch(assignments_patch, marker_candidates)
    mx_idx, mx_name = _resolve_mx_channel(index_path, args.mx_channel, marker_name)

    he_rgb = np.array(Image.open(he_path).convert("RGB"))
    mx_arr = np.load(mx_path)
    if mx_arr.ndim != 3:
        raise ValueError(f"Expected multiplex patch shape (C,H,W), got {mx_arr.shape}")
    if mx_idx < 0 or mx_idx >= mx_arr.shape[0]:
        raise ValueError(
            f"Resolved MX channel index {mx_idx} out of range [0, {mx_arr.shape[0] - 1}]"
        )
    mx_img = percentile_norm(mx_arr[mx_idx].astype(np.float32))

    cells = _load_patch_json(cellvit_path)
    cell_pairs = _match_assignments_to_cells(cells, assignments_patch)
    patch_shape = he_rgb.shape[:2]

    contour_img = _draw_cellvit_contours(he_rgb, cells)
    cellvit_overlay = _render_overlay(
        cell_pairs,
        patch_shape,
        lambda cell, row: (
            row.get("cellvit_mapped_type")
            if row is not None
            else map_cellvit_type(cell.get("type_cellvit", 0), None)
        ),
        CELL_TYPE_COLORS,
    )
    model_overlay = _render_overlay(
        cell_pairs,
        patch_shape,
        lambda _cell, row: row.get(model_col, "other") if row is not None else "other",
        model_colors,
    )
    final_overlay = _render_overlay(
        cell_pairs,
        patch_shape,
        lambda _cell, row: row.get("cell_type", "other") if row is not None else "other",
        CELL_TYPE_COLORS,
    )
    state_overlay = _render_overlay(
        cell_pairs,
        patch_shape,
        lambda _cell, row: row.get("cell_state", "other") if row is not None else "other",
        CELL_STATE_COLORS,
    )

    cellvit_on_he = _composite_rgba_on_rgb(he_rgb, cellvit_overlay)
    model_on_he = _composite_rgba_on_rgb(he_rgb, model_overlay)
    final_on_he = _composite_rgba_on_rgb(he_rgb, final_overlay)
    state_on_he = _composite_rgba_on_rgb(he_rgb, state_overlay)

    save_publication_figure, style_warning = _scientific_style(Path(args.scientific_vis_root))
    if style_warning:
        print(f"Warning: {style_warning}")

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.2), constrained_layout=True)
    ax = axes.ravel()

    ax[0].imshow(he_rgb)
    ax[0].set_title("H&E")
    ax[0].axis("off")
    _panel_label(ax[0], "A")

    ax[1].imshow(mx_img, cmap="viridis")
    ax[1].set_title(f"MX marker: {mx_name}")
    ax[1].axis("off")
    _panel_label(ax[1], "B")

    ax[2].imshow(contour_img)
    ax[2].set_title(f"CellViT contours (n={len(cells)})")
    ax[2].axis("off")
    _panel_label(ax[2], "C")

    ax[3].imshow(cellvit_on_he)
    ax[3].set_title("CellViT mapped type")
    ax[3].axis("off")
    _panel_label(ax[3], "D")

    ax[4].imshow(model_on_he)
    ax[4].set_title(model_title)
    ax[4].axis("off")
    _panel_label(ax[4], "E")

    ax[5].imshow(final_on_he)
    ax[5].set_title("Final fused type")
    ax[5].axis("off")
    _panel_label(ax[5], "F")

    ax[6].imshow(state_on_he)
    ax[6].set_title("Cell state")
    ax[6].axis("off")
    _panel_label(ax[6], "G")

    _build_evidence_panel(ax[7], assignments_patch, mx_name, classifier_used)
    _panel_label(ax[7], "H")

    mismatch_rate = float(assignments_patch["is_mismatch"].mean())
    fig.suptitle(
        (
            f"Patch {patch}: CellViT vs {model_name} vs Final "
            f"(n={len(assignments_patch)}, mismatch={mismatch_rate:.1%}, mode={classifier_used})"
        ),
        fontsize=10,
        y=1.02,
    )

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
    print(f"Done. patch={patch}, mx_channel={mx_idx}, mx_marker={mx_name}")


if __name__ == "__main__":
    main()
