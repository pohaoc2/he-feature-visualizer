"""
extract_cell_features.py — Build CellViT-aligned marker features from MX patches.

Reads per-patch CellViT JSON cells and per-patch multiplex arrays
(`{x0}_{y0}.npy`, shape `(C, H, W)`), computes pixel-wise mean intensity inside
each CellViT contour, and writes a feature table CSV suitable for Stage 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from utils.channels import load_channel_metadata
from utils.marker_aliases import canonicalize_marker_name


def _channels_from_metadata_csv(metadata_csv: pathlib.Path) -> list[str]:
    """Return metadata marker names ordered by channel index."""
    seen = load_channel_metadata(str(metadata_csv))
    ordered = sorted((idx, marker) for idx, marker in seen.values())
    return [marker for _, marker in ordered]


def _resolve_channel_names(
    index_data: dict,
    channels_arg: list[str] | None,
    metadata_csv: pathlib.Path | None,
) -> list[str]:
    """Resolve channel names preference: --channels > index.json > metadata CSV."""
    if channels_arg:
        return [str(ch).strip() for ch in channels_arg]

    index_channels = index_data.get("channels", [])
    if index_channels:
        return [str(ch).strip() for ch in index_channels]

    if metadata_csv is not None:
        return _channels_from_metadata_csv(metadata_csv)

    raise ValueError(
        "Cannot infer channel names. Provide --channels, or ensure index.json "
        "contains 'channels', or pass --metadata-csv."
    )


def _cell_mask_from_contour(
    contour: list[list[float]], height: int, width: int
) -> np.ndarray | None:
    """Rasterize contour into boolean mask in patch coordinates."""
    if len(contour) < 3:
        return None

    pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    if canvas.max() == 0:
        return None
    return canvas.astype(bool)


# ---------------------------------------------------------------------------
# Parallel worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

_ECF_CTX: dict = {}


def _init_ecf_worker(ctx: dict) -> None:
    global _ECF_CTX
    _ECF_CTX = ctx


def _ecf_patch_worker(patch_meta: dict) -> dict:
    ctx = _ECF_CTX
    cellvit_dir: pathlib.Path = ctx["cellvit_dir"]
    multiplex_dir: pathlib.Path = ctx["multiplex_dir"]
    channel_names: list[str] = ctx["channel_names"]
    coord_scale: float = ctx["coord_scale"]

    x0 = int(patch_meta["x0"])
    y0 = int(patch_meta["y0"])
    patch_id = f"{x0}_{y0}"

    cell_json_path = cellvit_dir / f"{patch_id}.json"
    if not cell_json_path.exists():
        return {"rows": [], "missing_cellvit": 1, "missing_multiplex": 0,
                "skipped_cells": 0, "processed_patches": 0, "warn_truncated": False}

    mx_path = multiplex_dir / f"{patch_id}.npy"
    if not mx_path.exists():
        return {"rows": [], "missing_cellvit": 0, "missing_multiplex": 1,
                "skipped_cells": 0, "processed_patches": 0, "warn_truncated": False}

    with cell_json_path.open(encoding="utf-8") as fh:
        cells: list[dict] = json.load(fh).get("cells", [])

    patch = np.load(mx_path)
    if patch.ndim != 3:
        raise ValueError(
            f"Multiplex patch must have shape (C,H,W), got {patch.shape} for {mx_path}"
        )

    c, h, w = patch.shape
    if len(channel_names) < c:
        raise ValueError(
            f"Channel names ({len(channel_names)}) shorter than patch channels ({c}) "
            f"for {mx_path}. Provide full --channels in patch order."
        )

    patch_channel_names = channel_names[:c]
    warn_truncated = len(channel_names) > c

    rows: list[dict] = []
    skipped_cells = 0

    for cell_index, cell in enumerate(cells):
        mask = _cell_mask_from_contour(cell.get("contour", []), h, w)
        if mask is None:
            skipped_cells += 1
            continue

        area = int(mask.sum())
        if area <= 0:
            skipped_cells += 1
            continue

        centroid = cell.get("centroid", [0, 0])
        lx = float(centroid[0]) if len(centroid) > 0 else 0.0
        ly = float(centroid[1]) if len(centroid) > 1 else 0.0
        gx_he = x0 + lx
        gy_he = y0 + ly

        marker_means = patch[:, mask].mean(axis=1, dtype=np.float64)

        raw_values: dict[str, float] = {}
        canonical_values: dict[str, list[float]] = defaultdict(list)
        for idx, marker in enumerate(patch_channel_names):
            name = str(marker).strip()
            val = float(marker_means[idx])
            if name:
                raw_values[name] = val
                canonical_values[canonicalize_marker_name(name)].append(val)

        row: dict = {
            "PatchID": patch_id,
            "CellIndex": int(cell_index),
            "type_cellvit": int(cell.get("type_cellvit", 0)),
            "type_prob": float(cell.get("type_prob", 0.0)),
            "Xt": gx_he * coord_scale,
            "Yt": gy_he * coord_scale,
            "X": gx_he * coord_scale,
            "Y": gy_he * coord_scale,
            "X_he": gx_he,
            "Y_he": gy_he,
            "X_local": lx,
            "Y_local": ly,
            "Area_cellvit_px": area,
        }
        row.update(raw_values)
        for canonical, values in canonical_values.items():
            row[canonical] = float(np.mean(values))
        rows.append(row)

    return {
        "rows": rows,
        "missing_cellvit": 0,
        "missing_multiplex": 0,
        "skipped_cells": skipped_cells,
        "processed_patches": 1,
        "warn_truncated": warn_truncated,
    }


def extract_cell_features_table(
    cellvit_dir: pathlib.Path,
    multiplex_dir: pathlib.Path,
    index_path: pathlib.Path,
    channels: list[str] | None = None,
    metadata_csv: pathlib.Path | None = None,
    coord_scale: float = 1.0,
    logger: logging.Logger | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Extract a per-cell feature table from CellViT contours and MX patches."""
    log = logger or logging.getLogger(__name__)

    with index_path.open(encoding="utf-8") as fh:
        index_data = json.load(fh)

    channel_names = _resolve_channel_names(index_data, channels, metadata_csv)
    patches = index_data.get("patches", [])

    rows: list[dict] = []
    missing_cellvit = 0
    missing_multiplex = 0
    processed_patches = 0
    skipped_cells = 0
    truncated_channels_logged = False

    ctx = {
        "cellvit_dir": cellvit_dir,
        "multiplex_dir": multiplex_dir,
        "channel_names": channel_names,
        "coord_scale": coord_scale,
    }
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_ecf_worker,
        initargs=(ctx,),
    ) as executor:
        for result in executor.map(_ecf_patch_worker, patches):
            rows.extend(result["rows"])
            missing_cellvit += result["missing_cellvit"]
            missing_multiplex += result["missing_multiplex"]
            skipped_cells += result["skipped_cells"]
            processed_patches += result["processed_patches"]
            if result["warn_truncated"] and not truncated_channels_logged:
                log.warning(
                    "More channel names provided (%d) than patch channels. "
                    "Using first N names.",
                    len(channel_names),
                )
                truncated_channels_logged = True

    df = pd.DataFrame(rows)
    if not df.empty:
        if "PatchID" in df.columns and "CellIndex" in df.columns:
            df = df.sort_values(["PatchID", "CellIndex"]).reset_index(drop=True)
        df.insert(0, "CellID", np.arange(1, len(df) + 1, dtype=np.int64))
    else:
        df = pd.DataFrame(columns=["CellID", "PatchID", "CellIndex", "Xt", "Yt"])

    log.info(
        "Extracted %d cells from %d patches (missing CellViT JSON: %d, missing MX patch: %d, skipped cells: %d).",
        len(df),
        processed_patches,
        missing_cellvit,
        missing_multiplex,
        skipped_cells,
    )
    return df


def extract_cell_features_to_csv(
    cellvit_dir: pathlib.Path,
    multiplex_dir: pathlib.Path,
    index_path: pathlib.Path,
    out_csv: pathlib.Path,
    channels: list[str] | None = None,
    metadata_csv: pathlib.Path | None = None,
    coord_scale: float = 1.0,
    logger: logging.Logger | None = None,
    workers: int = 1,
) -> pathlib.Path:
    """Extract CellViT-aligned cell features and write CSV to ``out_csv``."""
    df = extract_cell_features_table(
        cellvit_dir=cellvit_dir,
        multiplex_dir=multiplex_dir,
        index_path=index_path,
        channels=channels,
        metadata_csv=metadata_csv,
        coord_scale=coord_scale,
        logger=logger,
        workers=workers,
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description=(
            "Extract per-cell marker features from CellViT contours and "
            "per-patch multiplex arrays."
        )
    )
    parser.add_argument(
        "--cellvit-dir",
        required=True,
        help="Directory of {x0}_{y0}.json CellViT files.",
    )
    parser.add_argument(
        "--multiplex-dir",
        required=True,
        help="Directory of {x0}_{y0}.npy multiplex patches (C,H,W).",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Path to processed/index.json patch manifest.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for extracted per-cell features.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Channel names in multiplex .npy order. If omitted, uses index.json "
            "channels, then --metadata-csv as fallback."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help=(
            "Optional channel metadata CSV used only if --channels and index.json "
            "channels are unavailable."
        ),
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for Xt/Yt coordinates in output CSV. Use the same value "
            "you pass to stages.assign_cells --coord-scale."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for patch extraction (default: 4).",
    )
    args = parser.parse_args()

    out_csv = pathlib.Path(args.out_csv)
    metadata_csv = pathlib.Path(args.metadata_csv) if args.metadata_csv else None

    written = extract_cell_features_to_csv(
        cellvit_dir=pathlib.Path(args.cellvit_dir),
        multiplex_dir=pathlib.Path(args.multiplex_dir),
        index_path=pathlib.Path(args.index),
        out_csv=out_csv,
        channels=args.channels,
        metadata_csv=metadata_csv,
        coord_scale=args.coord_scale,
        logger=log,
        workers=args.workers,
    )
    log.info("Wrote extracted cell features CSV: %s", written)


if __name__ == "__main__":
    main()
