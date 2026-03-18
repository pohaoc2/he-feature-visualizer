#!/usr/bin/env python3
"""Random patch grid: H&E | Hoechst | CellViT mask | Final type | CD31 — one row per patch."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path("/tmp/matplotlib")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from PIL import Image

from utils.marker_aliases import canonicalize_marker_name, normalize_marker_name
from utils.normalize import percentile_norm

# ── Color maps ────────────────────────────────────────────────────────────────

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "cancer": (220, 50, 50, 200),
    "immune": (50, 100, 220, 200),
    "healthy": (50, 180, 50, 200),
    "other": (150, 150, 150, 120),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (230, 50, 180, 200),  # magenta
    "quiescent": (240, 140, 30, 200),  # amber
    "dead": (110, 40, 160, 200),  # purple
    "other": (160, 160, 160, 120),
}

# Hoechst 33342 fluorescence look: black background → electric blue → blue-white peak
HOECHST_CMAP = LinearSegmentedColormap.from_list(
    "hoechst33342",
    [
        (0.00, (0.00, 0.00, 0.00)),  # black background
        (0.35, (0.04, 0.10, 0.55)),  # deep blue
        (0.65, (0.10, 0.35, 0.90)),  # electric blue
        (0.85, (0.35, 0.65, 1.00)),  # bright blue-cyan
        (1.00, (0.80, 0.93, 1.00)),  # near-white peak
    ],
)

COL_TITLES = [
    "H&E",
    "Hoechst",
    "Cell mask (CellViT)",
    "Cell type (CellViT + CODEX)",
    "Cell state (CODEX)",
    "Vasculature",
    "Oxygen (O₂)",
    "Glucose",
]

ASSIGNMENT_REQUIRED_COLUMNS: tuple[str, str, str] = (
    "patch_id",
    "centroid_x_local",
    "centroid_y_local",
)
ASSIGNMENT_OPTIONAL_COLUMNS: tuple[str, str] = ("cell_type", "cell_state")
ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    *ASSIGNMENT_REQUIRED_COLUMNS,
    *ASSIGNMENT_OPTIONAL_COLUMNS,
)

# ── Small rendering helpers ───────────────────────────────────────────────────


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


def _composite_rgba_on_rgb(
    base_rgb: np.ndarray, overlay_rgba: np.ndarray
) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    if overlay_rgba.ndim != 3 or overlay_rgba.shape[-1] != 4:
        return base_rgb
    ov = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    return (alpha * ov + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)


def _resolve_marker_soft(
    index_json: Path,
    mx_arr: np.ndarray,
    marker: str,
) -> tuple[np.ndarray, str] | tuple[None, str]:
    """Return (normed float32 image, resolved_name) or (None, marker) if absent."""
    try:
        with index_json.open(encoding="utf-8") as fh:
            channels = [str(x) for x in json.load(fh).get("channels", [])]
        target_norm = normalize_marker_name(marker)
        target_canon = canonicalize_marker_name(marker)
        for i, name in enumerate(channels):
            if (
                normalize_marker_name(name) == target_norm
                or canonicalize_marker_name(name) == target_canon
                or target_norm in normalize_marker_name(name)
            ):
                if 0 <= i < mx_arr.shape[0]:
                    return percentile_norm(mx_arr[i].astype(np.float32)), name
                break
    except Exception:  # noqa: BLE001
        pass
    return None, marker


def _draw_cellvit_mask(shape: tuple[int, int], cells: list[dict]) -> np.ndarray:
    """Binary cell mask: white cells on black background."""
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for cell in cells:
        contour = cell.get("contour", [])
        if len(contour) < 3:
            continue
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], (255, 255, 255))
    return canvas


def _draw_final_type(
    he_rgb: np.ndarray,
    cells: list[dict],
    assignments_patch: pd.DataFrame,
    max_dist: float = 6.0,
) -> np.ndarray:
    """Fill CellViT contours with final-type colours, matched by centroid proximity."""
    rows = assignments_patch.to_dict(orient="records")
    unused = set(range(len(rows)))
    canvas = np.zeros((*he_rgb.shape[:2], 4), dtype=np.uint8)

    for cell in cells:
        centroid = np.asarray(cell.get("centroid", [0.0, 0.0]), dtype=float)
        best_idx, best_dist = None, float("inf")
        for idx in unused:
            r = rows[idx]
            d = float(
                np.hypot(
                    centroid[0] - float(r.get("centroid_x_local", 0.0)),
                    centroid[1] - float(r.get("centroid_y_local", 0.0)),
                )
            )
            if d < best_dist:
                best_dist, best_idx = d, idx

        label = "other"
        if best_idx is not None and best_dist <= max_dist:
            unused.remove(best_idx)
            label = str(rows[best_idx].get("cell_type", "other"))

        contour = cell.get("contour", [])
        if len(contour) < 3:
            continue
        rgba = CELL_TYPE_COLORS.get(label, CELL_TYPE_COLORS["other"])
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], rgba)

    return _composite_rgba_on_rgb(he_rgb, canvas)


def _draw_cell_state(
    he_rgb: np.ndarray,
    cells: list[dict],
    assignments_patch: pd.DataFrame,
    max_dist: float = 6.0,
) -> np.ndarray:
    """Fill CellViT contours with cell-state colours, matched by centroid proximity."""
    rows = assignments_patch.to_dict(orient="records")
    unused = set(range(len(rows)))
    canvas = np.zeros((*he_rgb.shape[:2], 4), dtype=np.uint8)

    for cell in cells:
        centroid = np.asarray(cell.get("centroid", [0.0, 0.0]), dtype=float)
        best_idx, best_dist = None, float("inf")
        for idx in unused:
            r = rows[idx]
            d = float(
                np.hypot(
                    centroid[0] - float(r.get("centroid_x_local", 0.0)),
                    centroid[1] - float(r.get("centroid_y_local", 0.0)),
                )
            )
            if d < best_dist:
                best_dist, best_idx = d, idx

        label = "other"
        if best_idx is not None and best_dist <= max_dist:
            unused.remove(best_idx)
            label = str(rows[best_idx].get("cell_state", "other"))

        contour = cell.get("contour", [])
        if len(contour) < 3:
            continue
        rgba = CELL_STATE_COLORS.get(label, CELL_STATE_COLORS["other"])
        pts = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], rgba)

    return _composite_rgba_on_rgb(he_rgb, canvas)


def _show_or_placeholder(
    ax: plt.Axes,
    img: np.ndarray | None,
    shape: tuple[int, int],
    cmap: str,
    label: str,
) -> None:
    if img is not None:
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.full((*shape, 3), 220, dtype=np.uint8))
        ax.text(
            0.5,
            0.5,
            f"{label}\nnot in panel",
            ha="center",
            va="center",
            fontsize=7,
            color="#555555",
            transform=ax.transAxes,
        )


def _add_type_legend(
    ax: plt.Axes,
    color_map: dict[str, tuple[int, int, int, int]],
    title: str = "Type",
) -> None:
    handles = [
        Patch(facecolor=np.array(v[:3], dtype=float) / 255.0, edgecolor="none", label=k)
        for k, v in color_map.items()
        if k != "other"
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        frameon=True,
        framealpha=0.85,
        fontsize=6,
        title=title,
        title_fontsize=6,
    )


# ── Patch discovery ───────────────────────────────────────────────────────────


def _available_patches(
    processed_dir: Path,
    assignments: pd.DataFrame | set[str] | None,
) -> list[str]:
    """Return patch IDs that have H&E + CellViT JSON (assignments optional)."""
    he_dir = processed_dir / "he"
    cellvit_dir = processed_dir / "cellvit"
    assigned_patches: set[str] | None = None
    if assignments is not None:
        if isinstance(assignments, pd.DataFrame):
            assigned_patches = set(assignments["patch_id"].astype(str).unique())
        else:
            assigned_patches = {str(pid) for pid in assignments}

    patches = []
    for png in sorted(he_dir.glob("*.png")):
        pid = png.stem
        if not (cellvit_dir / f"{pid}.json").exists():
            continue
        if assigned_patches is not None and pid not in assigned_patches:
            continue
        patches.append(pid)
    return patches


def _empty_assignments_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ASSIGNMENT_COLUMNS))


def _load_assignment_patch_ids(csv_path: Path, *, chunksize: int = 250_000) -> set[str]:
    """Stream only patch_id values from CSV and return unique IDs."""
    patch_ids: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=["patch_id"], chunksize=chunksize):
        patch_ids.update(chunk["patch_id"].astype(str).unique())
    return patch_ids


def _load_assignments_for_patches(
    csv_path: Path,
    patch_ids: set[str],
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load only the rows needed for selected patches and required display columns."""
    if not patch_ids:
        return _empty_assignments_df()

    header = pd.read_csv(csv_path, nrows=0)
    available = set(header.columns)
    missing_required = [
        col for col in ASSIGNMENT_REQUIRED_COLUMNS if col not in available
    ]
    if missing_required:
        missing = ", ".join(missing_required)
        raise RuntimeError(f"Assignments CSV is missing required columns: {missing}")

    usecols = [
        *ASSIGNMENT_REQUIRED_COLUMNS,
        *[c for c in ASSIGNMENT_OPTIONAL_COLUMNS if c in available],
    ]

    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        patch_col = chunk["patch_id"].astype(str)
        keep = patch_col.isin(patch_ids)
        if not keep.any():
            continue
        filtered = chunk.loc[keep].copy()
        filtered["patch_id"] = patch_col.loc[keep]
        rows.append(filtered)

    if not rows:
        return _empty_assignments_df()

    result = pd.concat(rows, ignore_index=True)
    for col in ASSIGNMENT_OPTIONAL_COLUMNS:
        if col not in result.columns:
            result[col] = "other"
        else:
            result[col] = result[col].fillna("other").astype(str)
    for col in ("centroid_x_local", "centroid_y_local"):
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    return result.loc[:, list(ASSIGNMENT_COLUMNS)].copy()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publication-style random patch grid: "
            "H&E | Hoechst | CellViT mask | Final type | CD31 — one row per patch."
        )
    )
    parser.add_argument("--processed", required=True, help="Processed directory.")
    parser.add_argument(
        "--random",
        dest="n_patches",
        type=int,
        required=True,
        help="Number of patches to sample randomly.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--assignments-csv",
        default=None,
        help="Path to cell_assignments.csv. Default: <processed>/cell_assignments.csv",
    )
    parser.add_argument(
        "--vasc-cd31",
        default="CD31",
        help="CD31 marker name in index.json (default: CD31).",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output path prefix. Default: <processed>/patch_grid.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (default: pdf,png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    assignments_path = (
        Path(args.assignments_csv)
        if args.assignments_csv
        else processed_dir / "cell_assignments.csv"
    )
    out_prefix = (
        Path(args.out_prefix) if args.out_prefix else processed_dir / "patch_grid"
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    index_path = processed_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.json: {index_path}")

    assigned_patch_ids: set[str] | None = None
    if assignments_path.exists():
        assigned_patch_ids = _load_assignment_patch_ids(assignments_path)
    else:
        print(
            f"[warn] assignments CSV not found ({assignments_path}); cell type/state cols will be blank."
        )

    patches = _available_patches(processed_dir, assigned_patch_ids)
    if not patches:
        raise RuntimeError(
            "No patches found with all required files (he, cellvit, assignments)."
        )

    n = min(args.n_patches, len(patches))
    rng = random.Random(args.seed)
    selected = rng.sample(patches, n)

    if assignments_path.exists():
        assignments_df = _load_assignments_for_patches(assignments_path, set(selected))
    else:
        assignments_df = None

    # Figure: n rows × 8 cols + extra width for two colorbars
    col_w, row_h = 2.5, 2.5
    n_cols = len(COL_TITLES)
    fig, axes = plt.subplots(
        n, n_cols, figsize=(n_cols * col_w + 1.2, n * row_h), constrained_layout=True
    )
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    # Column titles on first row
    for col, title in enumerate(COL_TITLES):
        axes[0, col].set_title(title, fontsize=9, pad=3)

    for row_idx, patch_id in enumerate(selected):
        he_path = processed_dir / "he" / f"{patch_id}.png"
        mx_path = processed_dir / "multiplex" / f"{patch_id}.npy"
        cellvit_path = processed_dir / "cellvit" / f"{patch_id}.json"

        he_rgb = np.array(Image.open(he_path).convert("RGB"))
        patch_shape = he_rgb.shape[:2]
        mx_arr = np.load(mx_path) if mx_path.exists() else None

        cells = _load_patch_json(cellvit_path)
        asgn = (
            assignments_df[assignments_df["patch_id"] == patch_id].copy()
            if assignments_df is not None
            else pd.DataFrame()
        )

        # Hoechst = channel 0 of the MX array
        hoechst_img = (
            percentile_norm(mx_arr[0].astype(np.float32))
            if mx_arr is not None
            else None
        )

        # Vasculature / oxygen / glucose PNGs from Stage 4 output
        vasc_png = processed_dir / "vasculature" / f"{patch_id}.png"
        oxygen_png = processed_dir / "oxygen" / f"{patch_id}.png"
        glucose_png = processed_dir / "glucose" / f"{patch_id}.png"
        vasc_rgba = (
            np.array(Image.open(vasc_png).convert("RGBA"))
            if vasc_png.exists()
            else None
        )
        oxygen_rgba = (
            np.array(Image.open(oxygen_png).convert("RGBA"))
            if oxygen_png.exists()
            else None
        )
        glucose_rgba = (
            np.array(Image.open(glucose_png).convert("RGBA"))
            if glucose_png.exists()
            else None
        )

        # Composite vasculature overlay onto H&E for display
        vasc_on_he = (
            _composite_rgba_on_rgb(he_rgb, vasc_rgba)
            if vasc_rgba is not None
            else he_rgb.copy()
        )

        mask_img = _draw_cellvit_mask(patch_shape, cells)
        final_on_he = (
            _draw_final_type(he_rgb, cells, asgn) if not asgn.empty else he_rgb.copy()
        )
        state_on_he = (
            _draw_cell_state(he_rgb, cells, asgn) if not asgn.empty else he_rgb.copy()
        )

        ax_row = axes[row_idx]

        # C1: H&E
        ax_row[0].imshow(he_rgb)
        ax_row[0].set_ylabel(patch_id, fontsize=7, labelpad=3)

        # C2: Hoechst (channel 0)
        _show_or_placeholder(
            ax_row[1], hoechst_img, patch_shape, HOECHST_CMAP, "Hoechst"
        )

        # C3: CellViT binary mask
        ax_row[2].imshow(mask_img, cmap="gray")

        # C4: Final fused type
        if assignments_df is not None:
            ax_row[3].imshow(final_on_he)
            if row_idx == n - 1:
                _add_type_legend(ax_row[3], CELL_TYPE_COLORS)
        else:
            ax_row[3].imshow(np.full((*patch_shape, 3), 220, dtype=np.uint8))
            ax_row[3].text(
                0.5,
                0.5,
                "No assignments",
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
                transform=ax_row[3].transAxes,
            )

        # C5: Cell state
        if assignments_df is not None:
            ax_row[4].imshow(state_on_he)
            if row_idx == n - 1:
                _add_type_legend(ax_row[4], CELL_STATE_COLORS, title="State")
        else:
            ax_row[4].imshow(np.full((*patch_shape, 3), 220, dtype=np.uint8))
            ax_row[4].text(
                0.5,
                0.5,
                "No assignments",
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
                transform=ax_row[4].transAxes,
            )

        # C6: Vasculature overlay on H&E
        if vasc_rgba is not None:
            ax_row[5].imshow(vasc_on_he)
        else:
            ax_row[5].imshow(np.full((*patch_shape, 3), 220, dtype=np.uint8))
            ax_row[5].text(
                0.5,
                0.5,
                "Vasculature\nnot found",
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
                transform=ax_row[5].transAxes,
            )

        # C7: Oxygen map (RGBA colormap image)
        if oxygen_rgba is not None:
            ax_row[6].imshow(oxygen_rgba)
        else:
            ax_row[6].imshow(np.full((*patch_shape, 3), 220, dtype=np.uint8))
            ax_row[6].text(
                0.5,
                0.5,
                "Oxygen\nnot found",
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
                transform=ax_row[6].transAxes,
            )

        # C8: Glucose map (RGBA colormap image)
        if glucose_rgba is not None:
            ax_row[7].imshow(glucose_rgba)
        else:
            ax_row[7].imshow(np.full((*patch_shape, 3), 220, dtype=np.uint8))
            ax_row[7].text(
                0.5,
                0.5,
                "Glucose\nnot found",
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
                transform=ax_row[7].transAxes,
            )

        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # Colorbars for oxygen (col 6) and glucose (col 7)
    sm_o2 = ScalarMappable(cmap="RdYlBu", norm=Normalize(vmin=0, vmax=1))
    sm_o2.set_array([])
    cb_o2 = fig.colorbar(sm_o2, ax=axes[:, 6], shrink=0.6, pad=0.03, aspect=30)
    cb_o2.set_label("O₂ proxy", fontsize=7, labelpad=4)
    cb_o2.set_ticks([0, 0.5, 1])
    cb_o2.set_ticklabels(["hypoxic", "0.5", "oxygenated"], fontsize=6)

    sm_glc = ScalarMappable(cmap="hot", norm=Normalize(vmin=0, vmax=1))
    sm_glc.set_array([])
    cb_glc = fig.colorbar(sm_glc, ax=axes[:, 7], shrink=0.6, pad=0.03, aspect=30)
    cb_glc.set_label("Glucose proxy", fontsize=7, labelpad=4)
    cb_glc.set_ticks([0, 0.5, 1])
    cb_glc.set_ticklabels(["depleted", "0.5", "high"], fontsize=6)

    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()] or ["png"]
    for fmt in formats:
        out_path = out_prefix.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)
    print(f"Done. patches={selected}")


if __name__ == "__main__":
    main()
