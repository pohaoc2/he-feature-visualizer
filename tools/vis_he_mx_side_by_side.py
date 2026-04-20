"""Side-by-side H&E and multiplex overview with selected marker colors.

Example:
  python -m tools.vis_he_mx_side_by_side \\
    --he data/he_crc33.ome.tif \\
    --mx data/mx_crc33.ome.tiff \\
    --markers-csv data/markers.csv \\
    --save-png paper/figures/he_mx_overview.png \\
    --downsample 64
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image

from utils.channels import resolve_channel_indices
from utils.ome import read_overview_chw


def _series_dims(series: tifffile.TiffPageSeries) -> tuple[int, int, str]:
    axes = series.axes.upper()
    shape = series.shape
    return int(shape[axes.index("X")]), int(shape[axes.index("Y")]), axes


def _open_series_store(series: tifffile.TiffPageSeries):
    try:
        raw = zarr.open(series.aszarr(), mode="r")
        if isinstance(raw, zarr.Array):
            return raw
        return raw["0"]
    except TypeError as exc:
        if "ZarrTiffStore" not in str(exc):
            raise
        warnings.warn(
            "tifffile.aszarr is unavailable; falling back to a memmapped TIFF read.",
            RuntimeWarning,
            stacklevel=2,
        )
        return series.asarray(out="memmap")


def _choose_pyramid_level(
    tif: tifffile.TiffFile, requested_downsample: int
) -> tuple[tifffile.TiffPageSeries, int, int, float]:
    """Return series level, index, residual stride, and effective downsample."""
    if requested_downsample < 1:
        raise ValueError("--downsample must be >= 1")

    base = tif.series[0]
    levels = list(getattr(base, "levels", []) or [base])
    base_w, base_h, _ = _series_dims(base)

    best_idx = 0
    best_factor = 1.0
    for idx, level in enumerate(levels):
        level_w, level_h, _ = _series_dims(level)
        factor = min(base_w / float(level_w), base_h / float(level_h))
        if factor <= requested_downsample and factor >= best_factor:
            best_idx = idx
            best_factor = factor

    residual = max(1, int(round(requested_downsample / best_factor)))
    effective = best_factor * residual
    return levels[best_idx], best_idx, residual, effective


def _read_fast_overview_chw(
    tif: tifffile.TiffFile,
    requested_downsample: int,
    channel_indices: list[int] | None = None,
) -> tuple[np.ndarray, int, float]:
    level, level_idx, residual, effective = _choose_pyramid_level(
        tif, requested_downsample
    )
    level_w, level_h, axes = _series_dims(level)
    chw = read_overview_chw(
        _open_series_store(level),
        axes,
        level_h,
        level_w,
        residual,
        channel_indices=channel_indices,
    )
    return chw, level_idx, effective


def _preserve_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to uint8 while keeping channel balance intact."""
    if rgb.dtype == np.uint8:
        return rgb

    arr = rgb.astype(np.float32, copy=False)
    lo = float(np.percentile(arr, 0.5))
    hi = float(np.percentile(arr, 99.8))
    if hi <= lo:
        return np.zeros(rgb.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _marker_norm(
    arr: np.ndarray,
    black_percentile: float = 70.0,
    white_percentile: float = 99.8,
    gamma: float = 0.8,
) -> np.ndarray:
    """Normalize one MX marker with a high black floor for dark backgrounds."""
    arr_f = arr.astype(np.float32, copy=False)
    lo = float(np.percentile(arr_f, black_percentile))
    hi = float(np.percentile(arr_f, white_percentile))
    if hi <= lo:
        hi = float(arr_f.max())
    if hi <= lo:
        return np.zeros(arr_f.shape, dtype=np.float32)
    norm = np.clip((arr_f - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma, dtype=np.float32)
    return norm.astype(np.float32)


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    for _ in range(max(0, int(radius))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def _background_mask(
    background: np.ndarray,
    black_percentile: float = 65.0,
    dilate_px: int = 2,
) -> np.ndarray:
    threshold = float(np.percentile(background.astype(np.float32), black_percentile))
    if threshold >= float(np.max(background)):
        return np.ones(background.shape, dtype=bool)
    return _dilate_binary(background > threshold, dilate_px)


def _render_he_rgb(chw: np.ndarray) -> np.ndarray:
    """CHW -> uint8 RGB, preserving H&E white balance."""
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW, got {chw.shape!r}")
    if chw.shape[0] >= 3:
        rgb = np.moveaxis(chw[:3], 0, -1)
    else:
        rgb = np.repeat(chw[0][..., np.newaxis], 3, axis=-1)
    return _preserve_rgb_uint8(rgb)


def _composite_markers_rgb(
    chw: np.ndarray,
    channel_indices: list[int],
    colors_rgb: np.ndarray,
    black_percentile: float = 70.0,
    white_percentile: float = 99.8,
    gamma: float = 0.8,
    background: np.ndarray | None = None,
    background_black_percentile: float = 65.0,
    background_dilate_px: int = 2,
) -> np.ndarray:
    """Blend normalized marker intensities onto RGB using additive color weights."""
    h, w = chw.shape[1], chw.shape[2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    colors_rgb = np.asarray(colors_rgb, dtype=np.float32).reshape(-1, 3)
    if len(channel_indices) != len(colors_rgb):
        raise ValueError("channel_indices and colors length mismatch")
    for idx, col in zip(channel_indices, colors_rgb):
        n = _marker_norm(
            chw[idx],
            black_percentile=black_percentile,
            white_percentile=white_percentile,
            gamma=gamma,
        )
        acc += n[..., np.newaxis] * col
    if background is not None:
        mask = _background_mask(
            background,
            black_percentile=background_black_percentile,
            dilate_px=background_dilate_px,
        )
        acc *= mask[..., np.newaxis]
    acc = np.clip(acc, 0.0, 1.0)
    return (acc * 255.0 + 0.5).astype(np.uint8)


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + gap + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + gap :] = right
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--he", required=True, type=Path, help="H&E OME-TIFF path")
    p.add_argument("--mx", required=True, type=Path, help="Multiplex OME-TIFF path")
    p.add_argument(
        "--markers-csv",
        required=True,
        type=Path,
        help="Channel metadata CSV (Channel_Number, Marker_Name)",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=64,
        help="Target overview downsample (larger = smaller output)",
    )
    p.add_argument(
        "--mx-black-percentile",
        type=float,
        default=70.0,
        help="MX values at or below this percentile render black (default: 70)",
    )
    p.add_argument(
        "--mx-white-percentile",
        type=float,
        default=99.8,
        help="MX values at this percentile render full intensity (default: 99.8)",
    )
    p.add_argument(
        "--mx-gamma",
        type=float,
        default=0.8,
        help="MX display gamma after percentile scaling (default: 0.8)",
    )
    p.add_argument(
        "--mx-background-marker",
        default="Hoechst",
        help="Marker used to mask MX non-tissue background; empty string disables",
    )
    p.add_argument(
        "--mx-background-percentile",
        type=float,
        default=65.0,
        help="Background-marker percentile used as tissue gate (default: 65)",
    )
    p.add_argument(
        "--mx-background-dilate",
        type=int,
        default=2,
        help="Dilate MX tissue gate by this many overview pixels (default: 2)",
    )
    p.add_argument(
        "--save-png",
        "--out",
        type=Path,
        default=None,
        dest="save_png",
        help="Output PNG path",
    )
    args = p.parse_args()

    names = ["CD45", "CD31", "Pan-CK", "SMA"]
    indices, resolved = resolve_channel_indices(str(args.markers_csv), names)
    mx_channel_indices = list(indices)
    background_index: int | None = None
    background_resolved: str | None = None
    background_marker = str(args.mx_background_marker).strip()
    if background_marker:
        try:
            bg_indices, bg_resolved = resolve_channel_indices(
                str(args.markers_csv), [background_marker]
            )
            background_index = int(bg_indices[0])
            background_resolved = bg_resolved[0]
        except ValueError as exc:
            warnings.warn(
                f"Cannot resolve MX background marker {background_marker!r}; "
                f"continuing without background masking. {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    # Colors aligned with *names* order: CD45 purple, CD31 yellow, PanCK green, SMA red
    colors = np.array(
        [
            [160, 32, 240],  # purple (approx orchid/darkorchid)
            [255, 220, 0],  # yellow
            [0, 220, 90],  # green
            [255, 40, 40],  # red
        ],
        dtype=np.float32,
    ) / 255.0

    with tifffile.TiffFile(str(args.he)) as tif:
        he_chw, he_level, he_effective_ds = _read_fast_overview_chw(
            tif, args.downsample
        )
    he_rgb = _render_he_rgb(he_chw)

    with tifffile.TiffFile(str(args.mx)) as tif:
        read_indices = (
            [background_index] + mx_channel_indices
            if background_index is not None
            else mx_channel_indices
        )
        mx_chw, mx_level, mx_effective_ds = _read_fast_overview_chw(
            tif,
            args.downsample,
            channel_indices=read_indices,
        )
    if background_index is not None:
        mx_background = mx_chw[0]
        mx_marker_chw = mx_chw[1:]
    else:
        mx_background = None
        mx_marker_chw = mx_chw

    mx_rgb = _composite_markers_rgb(
        mx_marker_chw,
        list(range(len(mx_channel_indices))),
        colors,
        black_percentile=args.mx_black_percentile,
        white_percentile=args.mx_white_percentile,
        gamma=args.mx_gamma,
        background=mx_background,
        background_black_percentile=args.mx_background_percentile,
        background_dilate_px=args.mx_background_dilate,
    )
    combo = side_by_side(he_rgb, mx_rgb)

    out = args.save_png
    if out is None:
        out = Path("paper/figures") / f"{args.he.stem}_vs_{args.mx.stem}_markers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combo).save(str(out))

    legend = "  ".join(f"{n}:{r}" for n, r in zip(names, resolved))
    print(f"Saved {out}  ({combo.shape[1]}x{combo.shape[0]})")
    print(f"Channels - {legend}")
    if background_resolved is not None:
        print(f"MX background gate - {background_resolved}")
    print(
        f"Downsample target={args.downsample}; "
        f"HE level={he_level} effective~{he_effective_ds:.2f}; "
        f"MX level={mx_level} effective~{mx_effective_ds:.2f}"
    )


if __name__ == "__main__":
    main()
