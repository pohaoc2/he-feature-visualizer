"""Build pyramid TIFFs and export downsampled views from pyramid TIFFs.

Usage
-----
python build_pyramid.py --mask data/WD-76845-097.tif
python build_pyramid.py --mask data/WD-76845-097.tif --out data/mask_pyramid.ome.tif --tile-size 512
python build_pyramid.py --pyramid data/mask_pyramid.ome.tif --downsample 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr


def _yx_shape(shape: tuple[int, ...], axes: str | None) -> tuple[int, int]:
    """Return (height, width) from array shape and OME axes."""
    if axes and "Y" in axes and "X" in axes:
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])
    if len(shape) < 2:
        raise ValueError(f"Expected at least 2D shape, got {shape}")
    return int(shape[-2]), int(shape[-1])


def _zarr_array_from_level(level: tifffile.TiffPageSeries) -> zarr.Array:
    """Open a tifffile pyramid level as a zarr.Array."""
    raw = zarr.open(level.aszarr(), mode="r")
    if isinstance(raw, zarr.Array):
        return raw
    return raw["0"]


def _prompt_downsample() -> int:
    """Prompt for a positive integer downsample factor."""
    while True:
        value = input("Enter target downsample factor (integer >= 1): ").strip()
        try:
            downsample = int(value)
            if downsample >= 1:
                return downsample
        except ValueError:
            pass
        print("Invalid value. Please enter an integer >= 1.")


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


def export_downsample_from_pyramid(
    src: Path,
    dst: Path,
    downsample: int,
    tile_size: int = 512,
) -> None:
    """Export a downsampled image from a pyramidal TIFF without full-res load."""
    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    t0 = time.perf_counter()
    print(f"[extract] opening pyramid image {src} ...")
    with tifffile.TiffFile(str(src)) as tif:
        series = tif.series[0]
        levels = list(series.levels)
        if not levels:
            raise ValueError(f"No image levels found in {src}")

        base_axes = levels[0].axes
        base_h, base_w = _yx_shape(levels[0].shape, base_axes)

        chosen_idx = 0
        chosen_factor = 1.0
        for i, level in enumerate(levels):
            lv_h, lv_w = _yx_shape(level.shape, level.axes)
            factor = max(base_h / lv_h, base_w / lv_w)
            if factor <= downsample and factor >= chosen_factor:
                chosen_idx = i
                chosen_factor = factor

        chosen_level = levels[chosen_idx]
        residual_step = max(1, int(round(downsample / chosen_factor)))
        effective_downsample = chosen_factor * residual_step
        print(
            f"[extract] selected level {chosen_idx} (axes={chosen_level.axes}, "
            f"shape={chosen_level.shape}, factor~{chosen_factor:.2f})"
        )
        print(
            f"[extract] applying residual step {residual_step} "
            f"(effective downsample~{effective_downsample:.2f})"
        )

        zarr_arr = _zarr_array_from_level(chosen_level)
        axes = chosen_level.axes
        slices: list[slice] = []
        for ax in axes:
            if ax in ("Y", "X"):
                slices.append(slice(None, None, residual_step))
            else:
                slices.append(slice(None))
        down = np.array(zarr_arr[tuple(slices)])

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[extract] writing downsampled image to {dst} ...")
    tifffile.imwrite(
        str(dst),
        down,
        tile=(tile_size, tile_size),
        compression="deflate",
        metadata={"axes": axes},
        bigtiff=True,
    )
    size_mb = dst.stat().st_size / 1024 / 1024
    print(
        f"[extract] done – shape={down.shape}, dtype={down.dtype}, "
        f"file={size_mb:.1f} MB, elapsed={time.perf_counter() - t0:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pyramidal OME-TIFF from a mask, or export a downsampled image "
            "from an existing pyramid."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Path to input mask TIF for pyramid-building mode.",
    )
    parser.add_argument(
        "--pyramid",
        type=Path,
        default=None,
        help="Path to existing pyramid TIFF for downsample-export mode.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults: <mask stem>_pyramid.ome.tif for --mask mode, "
            "<pyramid stem>_ds<factor>.ome.tif for --pyramid mode."
        ),
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Target downsample factor for --pyramid mode. Prompts if omitted.",
    )
    args = parser.parse_args()

    if bool(args.mask) == bool(args.pyramid):
        parser.error("Provide exactly one of --mask or --pyramid.")

    if args.mask is not None:
        out = args.out or args.mask.parent / f"{args.mask.stem}_pyramid.ome.tif"
        build_pyramid(args.mask, out, tile_size=args.tile_size)
        return

    downsample = (
        args.downsample if args.downsample is not None else _prompt_downsample()
    )
    out = (
        args.out or args.pyramid.parent / f"{args.pyramid.stem}_ds{downsample}.ome.tif"
    )
    export_downsample_from_pyramid(
        args.pyramid,
        out,
        downsample=downsample,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
