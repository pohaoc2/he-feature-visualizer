"""Convert CellViT JSON output to per-patch binary cell masks.

Reads per-patch CellViT JSON files from a processed directory, fills each cell
contour as a filled polygon, and writes one binary PNG per patch into an output
directory (default: ``<processed-dir>/cell_masks/``).

Output filenames match the CellViT JSON names: ``{x0}_{y0}.png``.

Usage
-----
python -m tools.cellvit_to_binary_mask --processed-dir processed_crc33_crop

python -m tools.cellvit_to_binary_mask --processed-dir processed_crc33_crop \\
    --out-dir processed_crc33_crop/cell_masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_patch_binary_masks(
    processed_dir: Path,
    out_dir: Path,
) -> None:
    """Write one uint8 binary mask PNG per CellViT patch (255=cell, 0=background).

    Parameters
    ----------
    processed_dir:
        Root of a processed pipeline directory that contains ``index.json``
        and a ``cellvit/`` sub-directory with per-patch JSON files.
    out_dir:
        Directory where per-patch mask PNGs are written.
    """
    index_path = processed_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"index.json not found: {index_path}")

    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    patch_size: int = index["patch_size"]

    cellvit_dir = processed_dir / "cellvit"
    if not cellvit_dir.is_dir():
        raise FileNotFoundError(f"cellvit/ directory not found: {cellvit_dir}")

    json_files = sorted(cellvit_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {cellvit_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_cells = 0
    for json_path in json_files:
        with json_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        cells: list[dict] = data.get("cells", [])

        patch_buf = np.zeros((patch_size, patch_size), dtype=np.uint8)
        for cell in cells:
            contour = cell.get("contour", [])
            if len(contour) < 3:
                continue
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(patch_buf, [pts], color=255)

        out_path = out_dir / f"{json_path.stem}.png"
        Image.fromarray(patch_buf).save(str(out_path))
        total_cells += len(cells)

    coverage_per_patch = total_cells  # just count; coverage logged per patch is expensive
    print(
        f"[cellvit_to_binary_mask] {len(json_files)} patches, "
        f"{total_cells} cells → {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CellViT JSON output to per-patch binary cell masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--processed-dir",
        "--processed_dir",
        required=True,
        type=Path,
        help="Processed pipeline directory containing index.json and cellvit/",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for per-patch mask PNGs (default: <processed-dir>/cell_masks/)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.processed_dir / "cell_masks"
    build_patch_binary_masks(args.processed_dir, out_dir)


if __name__ == "__main__":
    main()
