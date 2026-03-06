"""
assign_cells.py — Stage 3 of the histopathology pipeline.

Matches CellViT-segmented cells (from per-patch JSON files) to the nearest
row in CRC02.csv using a KD-tree on image pixel coordinates. Assigns each cell
a type (tumor/immune/stromal/vasculature/other) and state
(proliferating/emt/other) from the matched row's marker intensities.
Rasterizes the cell contours as filled RGBA PNG images saved to
processed/cell_types/ and processed/cell_states/.
"""

import argparse
import json
import logging
import pathlib
import warnings
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import scipy.spatial
from PIL import Image

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "tumor": (220, 50, 50, 200),  # red
    "immune": (50, 100, 220, 200),  # blue
    "stromal": (50, 180, 50, 200),  # green
    "other": (150, 150, 150, 150),  # gray
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferating": (0, 255, 0, 200),  # green   — Ki67/PCNA high
    "emt": (255, 165, 0, 200),  # orange  — Vimentin high + E-cad low
    "apoptotic": (139, 0, 139, 200),  # purple  — CellViT Dead (type 4)
    "quiescent": (100, 149, 237, 200),  # blue    — Keratin+ resting tumor
    "healthy": (144, 238, 144, 200),  # lime    — E-cad high, non-tumor, non-dividing
    "other": (80, 80, 80, 150),  # dark gray (visible on black background)
}

# ---------------------------------------------------------------------------
# Marker lists
# ---------------------------------------------------------------------------

TYPE_MARKERS = ["Keratin", "CD45", "aSMA", "CD31"]
STATE_MARKERS = ["Ki67", "PCNA", "Vimentin", "Ecadherin"]

# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def build_csv_index(df: pd.DataFrame, x_col: str, y_col: str) -> scipy.spatial.KDTree:
    """Build a KDTree from the cell coordinate columns of a DataFrame (global pixel space)."""
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    return scipy.spatial.KDTree(coords)


def assign_type(row: pd.Series, thresholds: dict[str, float]) -> str:
    """Assign cell type from marker intensities. Priority:
      tumor   if Keratin >= thresholds['Keratin']
      immune  if CD45   >= thresholds['CD45']
      stromal if aSMA   >= thresholds['aSMA']  (CD31+ endothelial cells also fall here)
      other   otherwise
    Note: vasculature is a tissue-level feature tracked via the CD31 channel in
    multiplex_layers.py, not a cell type — endothelial cells are classified as stromal.
    Missing markers treated as 0. Never raises."""
    try:
        keratin = float(
            row.get("Keratin", 0)
            if hasattr(row, "get")
            else getattr(row, "Keratin", 0) if hasattr(row, "Keratin") else 0
        )
        cd45 = float(
            row.get("CD45", 0)
            if hasattr(row, "get")
            else getattr(row, "CD45", 0) if hasattr(row, "CD45") else 0
        )
        asma = float(
            row.get("aSMA", 0)
            if hasattr(row, "get")
            else getattr(row, "aSMA", 0) if hasattr(row, "aSMA") else 0
        )
        cd31 = float(
            row.get("CD31", 0)
            if hasattr(row, "get")
            else getattr(row, "CD31", 0) if hasattr(row, "CD31") else 0
        )
    except Exception:
        return "other"

    try:
        if keratin >= thresholds.get("Keratin", float("inf")):
            return "tumor"
        if cd45 >= thresholds.get("CD45", float("inf")):
            return "immune"
        if asma >= thresholds.get("aSMA", float("inf")) or cd31 >= thresholds.get(
            "CD31", float("inf")
        ):
            return "stromal"
    except Exception:
        pass

    return "other"


def assign_state(
    row: pd.Series, thresholds: dict[str, float], type_cellvit: int = 0
) -> str:
    """Assign cell state from marker intensities and CellViT morphology type.

    Priority order:
      apoptotic     if type_cellvit == 4 (CellViT 'Dead' class — H&E morphology)
      proliferating if Ki67 >= thresholds['Ki67'] OR PCNA >= thresholds['PCNA']
      emt           if Vimentin >= thresholds['Vimentin'] AND Ecadherin < thresholds['Ecadherin']
      quiescent     if Keratin >= thresholds['Keratin'] AND Ecadherin >= thresholds['Ecadherin_high']
                    (resting tumor cell — Keratin+ but not dividing, epithelial junctions intact)
      healthy       if Ecadherin >= thresholds['Ecadherin_high'] AND Keratin < thresholds['Keratin']
                    (normal epithelial — E-cad high, not tumor marker, not dividing)
      other         otherwise
    Missing markers treated as 0. Never raises."""

    # 1. Apoptotic: CellViT morphology (Dead cell, type ID 4)
    if type_cellvit == 4:
        return "apoptotic"

    def _get(key):
        try:
            return float(
                row.get(key, 0) if hasattr(row, "get") else getattr(row, key, 0)
            )
        except Exception:
            return 0.0

    try:
        ki67 = _get("Ki67")
        pcna = _get("PCNA")
        vimentin = _get("Vimentin")
        ecad = _get("Ecadherin")
        keratin = _get("Keratin")
    except Exception:
        return "other"

    try:
        # 2. Proliferating: active division markers
        if ki67 >= thresholds.get("Ki67", float("inf")) or pcna >= thresholds.get(
            "PCNA", float("inf")
        ):
            return "proliferating"

        # 3. EMT: migratory — Vimentin high + E-cadherin lost
        if vimentin >= thresholds.get(
            "Vimentin", float("inf")
        ) and ecad < thresholds.get("Ecadherin", float("-inf")):
            return "emt"

        # 4. Quiescent tumor: Keratin+ resting cell with intact epithelial junctions
        if keratin >= thresholds.get(
            "Keratin", float("inf")
        ) and ecad >= thresholds.get("Ecadherin_high", float("inf")):
            return "quiescent"

        # 5. Healthy epithelial: E-cad high, no tumor marker, not dividing
        if ecad >= thresholds.get(
            "Ecadherin_high", float("inf")
        ) and keratin < thresholds.get("Keratin", float("inf")):
            return "healthy"

    except Exception:
        pass

    return "other"


def match_cells(
    cells: list[dict],
    kdtree,
    df: pd.DataFrame,
    thresholds: dict[str, float],
    x0: int,
    y0: int,
    max_dist: float = 15.0,
    coord_scale: float = 1.0,
) -> list[dict]:
    """Match each cell to the nearest CSV row within max_dist pixels (CSV coordinate space).

    Local centroid [lx, ly] is converted to global H&E coordinates, then scaled by
    coord_scale to match the CSV coordinate space:
        gx_csv = (x0 + lx) * coord_scale
        gy_csv = (y0 + ly) * coord_scale

    For CRC02: H&E is 0.325 µm/px, CSV coordinates are in multiplex space (0.650 µm/px),
    so coord_scale = 0.325 / 0.650 = 0.5.

    If distance <= max_dist (in CSV pixel units), assigns type+state from matched row.
    If no match, assigns cell_type='other', cell_state='other'.
    Modifies and returns the cells list in-place."""
    for cell in cells:
        try:
            centroid = cell.get("centroid", [0, 0])
            lx = float(centroid[0])
            ly = float(centroid[1])
            gx = (x0 + lx) * coord_scale
            gy = (y0 + ly) * coord_scale

            dist, idx = kdtree.query([gx, gy])
            if dist <= max_dist:
                matched_row = df.iloc[idx]
                type_cellvit = int(cell.get("type_cellvit", 0))
                cell["cell_type"] = assign_type(matched_row, thresholds)
                cell["cell_state"] = assign_state(matched_row, thresholds, type_cellvit)
            else:
                cell["cell_type"] = "other"
                cell["cell_state"] = "other"
        except Exception:
            cell["cell_type"] = "other"
            cell["cell_state"] = "other"

    return cells


def rasterize_cells(
    cells: list[dict],
    patch_size: int,
    color_key: str,
    color_map: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw filled cell contours into a (patch_size, patch_size, 4) uint8 RGBA array.
    For each cell: look up color from color_map[cell[color_key]].
    Draw filled polygon from cell['contour'] (list of [x,y] pairs) using cv2.fillPoly.
    Returns zeros (fully transparent) where no cell.
    Skip cells whose contour has fewer than 3 points."""
    canvas = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)

    for cell in cells:
        contour = cell.get("contour", [])
        if len(contour) < 3:
            continue

        label = cell.get(color_key, "other")
        rgba = color_map.get(label, color_map.get("other", (150, 150, 150, 150)))

        pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], rgba)

    return canvas


def compute_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-marker thresholds from the full CSV.

    Cell TYPE markers (exclusive classification — only top cells qualify):
      Keratin, CD45, aSMA, CD31: 95th percentile

    Cell STATE markers (broader, biologically common signals):
      Ki67, PCNA, Vimentin: 75th percentile
        → ~25% of cells are proliferating/migratory, matching CRC biology
      Ecadherin low (EMT):  25th percentile (clearly lost E-cad)
      Ecadherin high (healthy/quiescent): 50th percentile (median, intact junctions)
    """
    thresholds: dict[str, float] = {}

    for marker in TYPE_MARKERS:
        if marker in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thresholds[marker] = float(
                    np.nanpercentile(df[marker].to_numpy(dtype=float), 95)
                )
        else:
            logging.warning(
                "Marker '%s' not found in CSV; threshold set to inf.", marker
            )
            thresholds[marker] = float("inf")

    for marker in ["Ki67", "PCNA", "Vimentin"]:
        if marker in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thresholds[marker] = float(
                    np.nanpercentile(df[marker].to_numpy(dtype=float), 75)
                )
        else:
            logging.warning(
                "Marker '%s' not found in CSV; threshold set to inf.", marker
            )
            thresholds[marker] = float("inf")

    # Ecadherin: 25th pct = low (EMT — clearly lost); 50th pct = high (intact junctions)
    if "Ecadherin" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = df["Ecadherin"].to_numpy(dtype=float)
            thresholds["Ecadherin"] = float(np.nanpercentile(vals, 25))
            thresholds["Ecadherin_high"] = float(np.nanpercentile(vals, 50))
    else:
        logging.warning(
            "Marker 'Ecadherin' not found in CSV; thresholds set to extremes."
        )
        thresholds["Ecadherin"] = float("-inf")
        thresholds["Ecadherin_high"] = float("inf")

    return thresholds


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _resolve_coord_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return (x_col, y_col): prefer Xt/Yt, fallback to X/Y."""
    if "Xt" in df.columns and "Yt" in df.columns:
        return "Xt", "Yt"
    if "X" in df.columns and "Y" in df.columns:
        return "X", "Y"
    raise ValueError(
        "CSV must contain coordinate columns 'Xt'/'Yt' or 'X'/'Y'. "
        f"Found columns: {list(df.columns)}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Stage 3: assign cell types/states and rasterize contours."
    )
    parser.add_argument(
        "--cellvit-dir",
        required=True,
        help="Directory of {i}_{j}.json files from CellViT.",
    )
    parser.add_argument(
        "--features-csv", required=True, help="CRC02.csv with Xt, Yt, marker columns."
    )
    parser.add_argument(
        "--index", required=True, help="processed/index.json (patch grid)."
    )
    parser.add_argument(
        "--out", default="processed/", help="Output directory (default: processed/)."
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=15.0,
        help="Max nearest-neighbor distance in CSV pixel units (default: 15.0).",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to H&E global coordinates before KDTree query. "
        "Set to he_mpp/csv_mpp when H&E and CSV use different pixel spacings "
        "(e.g. 0.5 when H&E is 0.325 µm/px and CSV is in 20x/0.650 µm/px space).",
    )
    args = parser.parse_args()

    cellvit_dir = pathlib.Path(args.cellvit_dir)
    features_csv = pathlib.Path(args.features_csv)
    index_path = pathlib.Path(args.index)
    out_dir = pathlib.Path(args.out)
    max_dist = args.max_dist
    coord_scale = args.coord_scale
    log.info("Coord scale: %.4f (H&E px → CSV px)", coord_scale)

    # Output subdirectories
    types_dir = out_dir / "cell_types"
    states_dir = out_dir / "cell_states"
    types_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load CSV, compute thresholds, build KDTree
    # ------------------------------------------------------------------
    log.info("Loading features CSV: %s", features_csv)
    df = pd.read_csv(features_csv)
    log.info("  %d rows loaded.", len(df))

    x_col, y_col = _resolve_coord_cols(df)
    log.info("  Using coordinate columns: (%s, %s)", x_col, y_col)

    log.info("Computing per-marker thresholds …")
    thresholds = compute_thresholds(df)
    for marker, val in thresholds.items():
        log.info("  %-12s → %.4f", marker, val)

    log.info("Building KDTree …")
    kdtree = build_csv_index(df, x_col, y_col)
    log.info("  KDTree built on %d points.", kdtree.n)

    # ------------------------------------------------------------------
    # 2. Load index.json
    # ------------------------------------------------------------------
    log.info("Loading patch index: %s", index_path)
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    # ------------------------------------------------------------------
    # 3–6. Iterate patches
    # ------------------------------------------------------------------
    patch_size = index.get("patch_size", 256)
    processed = 0
    skipped = 0
    total_cells = 0

    per_patch_summary: dict[str, dict] = {}
    global_type_counts: Counter = Counter()
    global_state_counts: Counter = Counter()

    for patch_meta in patches:
        x0 = patch_meta["x0"]
        y0 = patch_meta["y0"]
        patch_id = f"{x0}_{y0}"

        json_path = cellvit_dir / f"{patch_id}.json"
        if not json_path.exists():
            log.warning("Missing CellViT file: %s — skipping.", json_path)
            skipped += 1
            continue

        with json_path.open() as fh:
            cells: list[dict] = json.load(fh).get("cells", [])

        # 4. Match + assign
        cells = match_cells(
            cells, kdtree, df, thresholds, x0, y0, max_dist, coord_scale
        )
        total_cells += len(cells)

        # 5. Rasterize
        type_img = rasterize_cells(cells, patch_size, "cell_type", CELL_TYPE_COLORS)
        state_img = rasterize_cells(cells, patch_size, "cell_state", CELL_STATE_COLORS)

        # 6. Save PNGs
        Image.fromarray(type_img, mode="RGBA").save(types_dir / f"{patch_id}.png")
        Image.fromarray(state_img, mode="RGBA").save(states_dir / f"{patch_id}.png")

        # 7. Accumulate summary counts
        type_counts = Counter(c.get("cell_type", "other") for c in cells)
        state_counts = Counter(c.get("cell_state", "other") for c in cells)
        per_patch_summary[patch_id] = {
            "n_cells": len(cells),
            "x0": x0,
            "y0": y0,
            "cell_types": dict(type_counts),
            "cell_states": dict(state_counts),
        }
        global_type_counts += type_counts
        global_state_counts += state_counts

        processed += 1
        if processed % 50 == 0:
            log.info(
                "  Progress: %d patches processed, %d skipped …", processed, skipped
            )

    # ------------------------------------------------------------------
    # 8. Save cell_summary.json
    # ------------------------------------------------------------------
    summary = {
        "n_patches": processed,
        "n_cells": total_cells,
        "coord_scale": coord_scale,
        "cell_types": dict(global_type_counts),
        "cell_states": dict(global_state_counts),
        "per_patch": per_patch_summary,
    }
    summary_path = out_dir / "cell_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary written to %s", summary_path)

    # ------------------------------------------------------------------
    # 9. Log global summary
    # ------------------------------------------------------------------
    log.info("Done.")
    log.info("  Patches processed : %d", processed)
    log.info("  Patches skipped   : %d", skipped)
    log.info("  Total cells       : %d", total_cells)
    log.info("  Cell types  (global): %s", dict(global_type_counts))
    log.info("  Cell states (global): %s", dict(global_state_counts))
    log.info("  Cell type PNGs    → %s", types_dir)
    log.info("  Cell state PNGs   → %s", states_dir)


if __name__ == "__main__":
    main()
