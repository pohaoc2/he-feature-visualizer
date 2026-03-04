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

import cv2
import numpy as np
import pandas as pd
import scipy.spatial
from PIL import Image

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "tumor":       (220,  50,  50, 200),
    "immune":      ( 50, 100, 220, 200),
    "stromal":     ( 50, 180,  50, 200),
    "vasculature": (255, 140,   0, 200),
    "other":       (150, 150, 150, 150),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferating": (  0, 255,   0, 200),
    "emt":           (255, 165,   0, 200),
    "other":         (  0,   0,   0,   0),
}

# ---------------------------------------------------------------------------
# Marker lists
# ---------------------------------------------------------------------------

TYPE_MARKERS  = ["Keratin", "CD45", "aSMA", "CD31"]
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
      vasculature if CD31  >= thresholds['CD31']
      tumor       if Keratin >= thresholds['Keratin']
      immune      if CD45  >= thresholds['CD45']
      stromal     if aSMA  >= thresholds['aSMA']
      other       otherwise
    Missing markers treated as 0. Never raises."""
    try:
        cd31    = float(row.get("CD31",    0) if hasattr(row, "get") else getattr(row, "CD31",    0) if hasattr(row, "CD31")    else 0)
        keratin = float(row.get("Keratin", 0) if hasattr(row, "get") else getattr(row, "Keratin", 0) if hasattr(row, "Keratin") else 0)
        cd45    = float(row.get("CD45",    0) if hasattr(row, "get") else getattr(row, "CD45",    0) if hasattr(row, "CD45")    else 0)
        asma    = float(row.get("aSMA",    0) if hasattr(row, "get") else getattr(row, "aSMA",    0) if hasattr(row, "aSMA")    else 0)
    except Exception:
        return "other"

    try:
        if cd31    >= thresholds.get("CD31",    float("inf")):
            return "vasculature"
        if keratin >= thresholds.get("Keratin", float("inf")):
            return "tumor"
        if cd45    >= thresholds.get("CD45",    float("inf")):
            return "immune"
        if asma    >= thresholds.get("aSMA",    float("inf")):
            return "stromal"
    except Exception:
        pass

    return "other"


def assign_state(row: pd.Series, thresholds: dict[str, float]) -> str:
    """Assign cell state from marker intensities.
      proliferating if Ki67 >= thresholds['Ki67'] OR PCNA >= thresholds['PCNA']
      emt           if Vimentin >= thresholds['Vimentin'] AND Ecadherin < thresholds['Ecadherin']
                    (proliferating takes priority over emt)
      other         otherwise
    Missing markers treated as 0. Never raises."""
    try:
        ki67     = float(row.get("Ki67",     0) if hasattr(row, "get") else getattr(row, "Ki67",     0) if hasattr(row, "Ki67")     else 0)
        pcna     = float(row.get("PCNA",     0) if hasattr(row, "get") else getattr(row, "PCNA",     0) if hasattr(row, "PCNA")     else 0)
        vimentin = float(row.get("Vimentin", 0) if hasattr(row, "get") else getattr(row, "Vimentin", 0) if hasattr(row, "Vimentin") else 0)
        ecad     = float(row.get("Ecadherin",0) if hasattr(row, "get") else getattr(row, "Ecadherin",0) if hasattr(row, "Ecadherin") else 0)
    except Exception:
        return "other"

    try:
        if (ki67 >= thresholds.get("Ki67", float("inf")) or
                pcna >= thresholds.get("PCNA", float("inf"))):
            return "proliferating"
        if (vimentin >= thresholds.get("Vimentin", float("inf")) and
                ecad < thresholds.get("Ecadherin", float("-inf"))):
            return "emt"
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
) -> list[dict]:
    """Match each cell to the nearest CSV row within max_dist pixels (global space).
    Converts local centroid [lx, ly] to global: gx = x0 + lx, gy = y0 + ly.
    Queries KDTree with (gx, gy). If distance <= max_dist, assigns type+state from matched row.
    If no match, assigns cell_type='other', cell_state='other'.
    Modifies and returns the cells list in-place (adds 'cell_type', 'cell_state' keys)."""
    for cell in cells:
        try:
            centroid = cell.get("centroid", [0, 0])
            lx = float(centroid[0])
            ly = float(centroid[1])
            gx = x0 + lx
            gy = y0 + ly

            dist, idx = kdtree.query([gx, gy])
            if dist <= max_dist:
                matched_row = df.iloc[idx]
                cell["cell_type"]  = assign_type(matched_row, thresholds)
                cell["cell_state"] = assign_state(matched_row, thresholds)
            else:
                cell["cell_type"]  = "other"
                cell["cell_state"] = "other"
        except Exception:
            cell["cell_type"]  = "other"
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
        rgba  = color_map.get(label, color_map.get("other", (150, 150, 150, 150)))

        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], color=rgba)

    return canvas


def compute_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-marker thresholds from the full CSV.
    95th percentile for TYPE_MARKERS + Ki67, PCNA, Vimentin.
    5th  percentile for Ecadherin (low = EMT marker)."""
    thresholds: dict[str, float] = {}

    high_markers = TYPE_MARKERS + ["Ki67", "PCNA", "Vimentin"]
    for marker in high_markers:
        if marker in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thresholds[marker] = float(np.nanpercentile(df[marker].to_numpy(dtype=float), 95))
        else:
            logging.warning("Marker '%s' not found in CSV; threshold set to inf.", marker)
            thresholds[marker] = float("inf")

    # Ecadherin: low expression signals EMT
    if "Ecadherin" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thresholds["Ecadherin"] = float(np.nanpercentile(df["Ecadherin"].to_numpy(dtype=float), 5))
    else:
        logging.warning("Marker 'Ecadherin' not found in CSV; threshold set to -inf.")
        thresholds["Ecadherin"] = float("-inf")

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
    parser.add_argument("--cellvit-dir",  required=True,          help="Directory of {i}_{j}.json files from CellViT.")
    parser.add_argument("--features-csv", required=True,          help="CRC02.csv with Xt, Yt, marker columns.")
    parser.add_argument("--index",        required=True,          help="processed/index.json (patch grid).")
    parser.add_argument("--out",          default="processed/",   help="Output directory (default: processed/).")
    parser.add_argument("--max-dist",     type=float, default=15.0, help="Max nearest-neighbor distance in pixels (default: 15.0).")
    args = parser.parse_args()

    cellvit_dir  = pathlib.Path(args.cellvit_dir)
    features_csv = pathlib.Path(args.features_csv)
    index_path   = pathlib.Path(args.index)
    out_dir      = pathlib.Path(args.out)
    max_dist     = args.max_dist

    # Output subdirectories
    types_dir  = out_dir / "cell_types"
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
    with index_path.open() as fh:
        index = json.load(fh)

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    # ------------------------------------------------------------------
    # 3–6. Iterate patches
    # ------------------------------------------------------------------
    patch_size  = index.get("patch_size", 256)
    processed   = 0
    skipped     = 0
    total_cells = 0

    for patch_meta in patches:
        i  = patch_meta["i"]
        j  = patch_meta["j"]
        x0 = patch_meta.get("x0", patch_meta.get("x", 0))
        y0 = patch_meta.get("y0", patch_meta.get("y", 0))

        json_path = cellvit_dir / f"{i}_{j}.json"
        if not json_path.exists():
            log.warning("Missing CellViT file: %s — skipping.", json_path)
            skipped += 1
            continue

        with json_path.open() as fh:
            cells: list[dict] = json.load(fh).get("cells", [])

        # 4. Match + assign
        cells = match_cells(cells, kdtree, df, thresholds, x0, y0, max_dist)
        total_cells += len(cells)

        # 5. Rasterize
        type_img  = rasterize_cells(cells, patch_size, "cell_type",  CELL_TYPE_COLORS)
        state_img = rasterize_cells(cells, patch_size, "cell_state", CELL_STATE_COLORS)

        # 6. Save PNGs
        Image.fromarray(type_img,  mode="RGBA").save(types_dir  / f"{i}_{j}.png")
        Image.fromarray(state_img, mode="RGBA").save(states_dir / f"{i}_{j}.png")

        processed += 1
        if processed % 50 == 0:
            log.info("  Progress: %d patches processed, %d skipped …", processed, skipped)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("Done.")
    log.info("  Patches processed : %d", processed)
    log.info("  Patches skipped   : %d", skipped)
    log.info("  Total cells matched: %d", total_cells)
    log.info("  Cell type PNGs    → %s", types_dir)
    log.info("  Cell state PNGs   → %s", states_dir)


if __name__ == "__main__":
    main()
