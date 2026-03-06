"""Convert a cell-mask TIF to a tiled, compressed, pyramidal OME-TIFF.

Usage
-----
python build_pyramid.py --mask data/WD-76845-097.tif
python build_pyramid.py --mask data/WD-76845-097.tif --out data/mask_pyramid.ome.tif --tile-size 512
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile


def build_pyramid(src: Path, dst: Path, tile_size: int = 512) -> None:
    """Read a uint32 mask and write a tiled, compressed, pyramidal OME-TIFF.

    Pyramid levels are generated at 2x steps until the smallest dimension
    falls below one tile.  Nearest-neighbour downsampling preserves label IDs.
    """
    t0 = time.perf_counter()
    print(f"[pyramid] reading full mask from {src} ...")
    mask = tifffile.imread(str(src))
    print(
        f"[pyramid] shape={mask.shape}, dtype={mask.dtype}, "
        f"loaded in {time.perf_counter() - t0:.1f}s"
    )

    levels: list[np.ndarray] = [mask]
    current = mask
    while min(current.shape) > tile_size:
        current = current[::2, ::2]
        levels.append(current)
        print(f"[pyramid]   level {len(levels) - 1}: {current.shape}")

    print(f"[pyramid] writing {len(levels)} levels to {dst} ...")
    with tifffile.TiffWriter(str(dst), bigtiff=True) as tw:
        options = {"tile": (tile_size, tile_size), "compression": "deflate"}
        tw.write(levels[0], subifds=len(levels) - 1, **options)
        print(
            f"[pyramid]   wrote level 0 ({levels[0].shape}) – "
            f"{time.perf_counter() - t0:.1f}s"
        )
        for i in range(1, len(levels)):
            tw.write(levels[i], **options)
            print(
                f"[pyramid]   wrote level {i} ({levels[i].shape}) – "
                f"{time.perf_counter() - t0:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"[pyramid] done – {dst} ({size_mb:.1f} MB) in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a uint32 mask TIF to a pyramidal OME-TIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mask", required=True, type=Path, help="Path to the uint32 mask TIF"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <mask stem>_pyramid.ome.tif)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)"
    )
    args = parser.parse_args()

    out = args.out or args.mask.parent / f"{args.mask.stem}_pyramid.ome.tif"
    build_pyramid(args.mask, out, tile_size=args.tile_size)


if __name__ == "__main__":
    main()
