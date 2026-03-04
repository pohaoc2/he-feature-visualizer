"""
multiplex_layers.py — Stage 4+5 of the histopathology analysis pipeline.

Stage 4 (spatial analysis) reads per-patch multiplex immunofluorescence arrays
produced by Stage 1 (patchify.py) and derives three biologically meaningful
proxy maps from the raw channel intensities:

    vasculature
        Binary segmentation of CD31+ vascular structures.  CD31 (PECAM-1) is an
        endothelial marker; the binarised mask delineates vessel lumens and
        capillaries in the tissue.  Rendered as red pixels on a transparent
        background for overlay on the H&E image.

    oxygen
        A spatial proxy for tissue oxygenation derived from the Euclidean
        distance transform of the CD31 mask.  Pixels at or near vessel walls
        are assumed to be well-oxygenated (blue in the RdYlBu colormap), while
        pixels far from vessels are assumed to be hypoxic (red).  This
        approximation follows the classical Krogh cylinder model of oxygen
        diffusion.

    glucose / metabolic demand
        A proxy for local metabolic activity based on the maximum of
        normalised Ki67 and PCNA intensities.  Ki67 marks cells in active
        proliferation (G1-M phases) and PCNA is a DNA-replication clamp loader
        also elevated during S-phase; their maximum highlights the most
        metabolically demanding regions.  Rendered with the 'hot' colormap
        (black → red → yellow/white as demand increases).

Stage 5 (visualisation) saves the three RGBA overlay PNGs to sub-directories
of the output folder so they can be served directly by server_patches.py and
composited on top of H&E tiles in viewer_patches.html.

Input
-----
processed/multiplex/{i}_{j}.npy   — (C, H, W) uint16 arrays from Stage 1

Output
------
{out}/vasculature/{i}_{j}.png     — RGBA: red vessels on transparent background
{out}/oxygen/{i}_{j}.png          — RGBA: RdYlBu oxygenation map
{out}/glucose/{i}_{j}.png         — RGBA: hot metabolic-demand map
"""

import argparse
import json
import logging
import pathlib

import cv2
import matplotlib
import matplotlib.cm
import numpy as np
import pandas as pd
import scipy.ndimage
from PIL import Image


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_multiplex_patch(npy_path: str) -> np.ndarray:
    """Load and return (C, H, W) uint16 array from .npy file."""
    return np.load(npy_path)


def get_channel_index(channel_names: list[str], target: str) -> int:
    """Return 0-based index of target in channel_names (case-insensitive).

    Raises ValueError with a helpful message if not found.
    """
    target_lower = target.lower()
    for idx, name in enumerate(channel_names):
        if name.lower() == target_lower:
            return idx
    raise ValueError(
        f"Channel '{target}' not found in channel list {channel_names}. "
        f"Available (case-insensitive): {channel_names}"
    )


def extract_channel(patch: np.ndarray, idx: int) -> np.ndarray:
    """Return patch[idx] as (H, W) uint16 array."""
    return patch[idx]


def percentile_norm(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Clip arr to [p_low, p_high] percentiles, normalize to [0.0, 1.0] float32.

    If p_high value == p_low value (uniform input), return zeros.
    """
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def binarize_otsu(arr: np.ndarray) -> np.ndarray:
    """Otsu threshold on (H, W) float32 [0,1] → bool mask.

    Implementation:
      scaled = (arr * 255).clip(0, 255).astype(np.uint8)
      _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      return binary.astype(bool)
    """
    scaled = (arr * 255).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def apply_colormap(arr_01: np.ndarray, colormap: str) -> np.ndarray:
    """Apply a matplotlib named colormap to (H, W) float32 [0,1].

    Returns (H, W, 4) RGBA uint8.

    Implementation:
      cmap = matplotlib.cm.get_cmap(colormap)
      rgba_float = cmap(arr_01)           # (H, W, 4) float64 in [0,1]
      return (rgba_float * 255).clip(0, 255).astype(np.uint8)
    """
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        cmap = matplotlib.cm.get_cmap(colormap)  # type: ignore[attr-defined]
    rgba_float = cmap(arr_01)  # (H, W, 4) float64 in [0, 1]
    return (rgba_float * 255).clip(0, 255).astype(np.uint8)


def make_vasculature_overlay(cd31_mask: np.ndarray,
                             color: tuple = (255, 60, 0, 200)) -> np.ndarray:
    """Binary bool mask → (H, W, 4) RGBA uint8.

    True pixels → color tuple; False → (0, 0, 0, 0).
    """
    h, w = cd31_mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[cd31_mask] = color
    return out


def make_oxygen_map(cd31_mask: np.ndarray) -> np.ndarray:
    """Oxygen proxy via distance transform.

    Steps:
      1. dist = scipy.ndimage.distance_transform_edt(~cd31_mask)   # 0 at vessel
      2. norm = percentile_norm(dist.astype(np.float32))           # 0=vessel, 1=far
      3. inverted = 1.0 - norm    # invert so 1=near vessel (oxygenated), 0=far (hypoxic)
      4. apply_colormap(inverted, 'RdYlBu')
         — RdYlBu: 0→red (hypoxic), 0.5→yellow, 1→blue (oxygenated)

    Returns (H, W, 4) RGBA uint8.
    """
    dist = scipy.ndimage.distance_transform_edt(~cd31_mask)
    norm = percentile_norm(dist.astype(np.float32))
    inverted = (1.0 - norm).astype(np.float32)
    return apply_colormap(inverted, "RdYlBu")


def make_glucose_map(ki67: np.ndarray, pcna: np.ndarray) -> np.ndarray:
    """Metabolic demand proxy.

    Steps:
      1. ki67_norm = percentile_norm(ki67.astype(np.float32))
      2. pcna_norm = percentile_norm(pcna.astype(np.float32))
      3. metabolic = np.maximum(ki67_norm, pcna_norm)
      4. apply_colormap(metabolic, 'hot')
         — hot: 0→black, 0.5→red, 1→yellow/white

    Returns (H, W, 4) RGBA uint8.
    """
    ki67_norm = percentile_norm(ki67.astype(np.float32))
    pcna_norm = percentile_norm(pcna.astype(np.float32))
    metabolic = np.maximum(ki67_norm, pcna_norm)
    return apply_colormap(metabolic, "hot")


def load_channel_names(metadata_csv: str, requested: list[str]) -> list[str]:
    """Parse metadata CSV (columns: Channel ID, Target Name).

    Check that every name in 'requested' appears in Target Name column
    (case-insensitive).  Returns requested unchanged if all found.
    Raises ValueError listing all missing names if any not found.
    """
    df = pd.read_csv(metadata_csv)
    df.columns = [c.strip() for c in df.columns]

    if "Target Name" not in df.columns:
        raise ValueError(
            f"metadata CSV must have a 'Target Name' column. Found: {list(df.columns)}"
        )

    available_lower = {str(v).strip().lower() for v in df["Target Name"].dropna()}

    missing = [
        name for name in requested
        if name.lower() not in available_lower
    ]
    if missing:
        raise ValueError(
            f"Channel(s) not found in metadata CSV 'Target Name' column: {missing}\n"
            f"Available target names (case-insensitive): {sorted(available_lower)}"
        )

    return requested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description=(
            "Stage 4+5: Generate vasculature, oxygen, and glucose overlay PNGs "
            "from per-patch multiplex .npy files."
        )
    )
    parser.add_argument(
        "--multiplex-dir", required=True,
        help="Directory containing {i}_{j}.npy multiplex patch files.",
    )
    parser.add_argument(
        "--index", required=True,
        help="Path to processed/index.json (patch grid manifest).",
    )
    parser.add_argument(
        "--metadata-csv", required=True,
        help="Channel metadata CSV with 'Channel ID' and 'Target Name' columns.",
    )
    parser.add_argument(
        "--out", default="processed/",
        help="Output directory (default: processed/).",
    )
    parser.add_argument(
        "--channels", nargs="+", default=["CD31", "Ki67", "PCNA"],
        metavar="NAME",
        help="Multiplex channel names present in the .npy files, in order "
             "(must match --channels passed to patchify.py). "
             "Default: CD31 Ki67 PCNA",
    )
    args = parser.parse_args()

    multiplex_dir = pathlib.Path(args.multiplex_dir)
    index_path    = pathlib.Path(args.index)
    out_dir       = pathlib.Path(args.out)

    # ------------------------------------------------------------------
    # 1. Validate channel names against metadata CSV
    # ------------------------------------------------------------------
    log.info("Validating channel names against metadata CSV: %s", args.metadata_csv)
    load_channel_names(args.metadata_csv, args.channels)
    log.info("  All requested channels found: %s", args.channels)

    # ------------------------------------------------------------------
    # 2. Resolve channel positions within the .npy arrays
    # ------------------------------------------------------------------
    cd31_idx = get_channel_index(args.channels, "CD31")
    ki67_idx = get_channel_index(args.channels, "Ki67")
    pcna_idx = get_channel_index(args.channels, "PCNA")
    log.info(
        "  Channel indices — CD31: %d, Ki67: %d, PCNA: %d",
        cd31_idx, ki67_idx, pcna_idx,
    )

    # ------------------------------------------------------------------
    # 3. Load index.json, prepare output directories
    # ------------------------------------------------------------------
    log.info("Loading patch index: %s", index_path)
    with index_path.open() as fh:
        index = json.load(fh)

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    vasc_dir    = out_dir / "vasculature"
    oxygen_dir  = out_dir / "oxygen"
    glucose_dir = out_dir / "glucose"
    for d in (vasc_dir, oxygen_dir, glucose_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4–6. Process each patch
    # ------------------------------------------------------------------
    processed = 0
    skipped   = 0

    for patch_meta in patches:
        i = patch_meta["i"]
        j = patch_meta["j"]

        npy_path = multiplex_dir / f"{i}_{j}.npy"
        if not npy_path.exists():
            log.warning("Missing multiplex file: %s — skipping.", npy_path)
            skipped += 1
            continue

        # a. Load (C, H, W) uint16 array
        patch = load_multiplex_patch(str(npy_path))

        # b. Extract individual channels
        cd31_raw = extract_channel(patch, cd31_idx)
        ki67_raw = extract_channel(patch, ki67_idx)
        pcna_raw = extract_channel(patch, pcna_idx)

        # c. Vasculature overlay
        cd31_norm = percentile_norm(cd31_raw.astype(np.float32))
        cd31_mask = binarize_otsu(cd31_norm)
        vasc_rgba = make_vasculature_overlay(cd31_mask)

        # d. Oxygen map
        oxygen_rgba = make_oxygen_map(cd31_mask)

        # e. Glucose / metabolic demand map
        glucose_rgba = make_glucose_map(ki67_raw, pcna_raw)

        # f. Save PNGs
        Image.fromarray(vasc_rgba,    "RGBA").save(vasc_dir    / f"{i}_{j}.png")
        Image.fromarray(oxygen_rgba,  "RGBA").save(oxygen_dir  / f"{i}_{j}.png")
        Image.fromarray(glucose_rgba, "RGBA").save(glucose_dir / f"{i}_{j}.png")

        processed += 1
        if processed % 50 == 0:
            log.info(
                "  Progress: %d patches processed, %d skipped …",
                processed, skipped,
            )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("Done.")
    log.info("  Patches processed : %d", processed)
    log.info("  Patches skipped   : %d", skipped)
    log.info("  Vasculature PNGs  → %s", vasc_dir)
    log.info("  Oxygen PNGs       → %s", oxygen_dir)
    log.info("  Glucose PNGs      → %s", glucose_dir)


if __name__ == "__main__":
    main()
