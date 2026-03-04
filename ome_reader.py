from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import tifffile
import zarr


@dataclass(frozen=True)
class OMEInfo:
    axes: str
    width: int
    height: int
    n_channels: int | None


def _open_store(tif: tifffile.TiffFile):
    series = tif.series[0]
    raw = zarr.open(series.aszarr(), mode="r")
    return raw if isinstance(raw, zarr.Array) else raw["0"]


def get_ome_info(path: str) -> OMEInfo:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = series.shape
        height = int(shape[axes.index("Y")])
        width = int(shape[axes.index("X")])
        n_channels = int(shape[axes.index("C")]) if "C" in axes else None
        return OMEInfo(axes=axes, width=width, height=height, n_channels=n_channels)


def read_region_channels(
    path: str,
    *,
    y0: int,
    x0: int,
    h: int,
    w: int,
    channels: Sequence[int],
) -> np.ndarray:
    """
    Read a window from a multichannel OME-TIFF as (h, w, len(channels)).

    Notes:
    - Coordinates are in full-resolution pixel space.
    - Out-of-bounds reads are padded with zeros.
    """
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    if not channels:
        raise ValueError("channels must be non-empty")

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = series.shape
        img_h = int(shape[axes.index("Y")])
        img_w = int(shape[axes.index("X")])

        y0i = int(y0)
        x0i = int(x0)
        y0c = max(0, y0i)
        x0c = max(0, x0i)
        y1c = min(img_h, y0i + int(h))
        x1c = min(img_w, x0i + int(w))

        store = _open_store(tif)

        # Build slice for the clamped region
        sl = []
        for ax in axes:
            if ax == "C":
                sl.append(list(channels))
            elif ax == "Y":
                sl.append(slice(y0c, y1c))
            elif ax == "X":
                sl.append(slice(x0c, x1c))
            else:
                sl.append(0)

        arr = np.array(store[tuple(sl)])

        # Convert to (H, W, C)
        if "C" in axes and arr.ndim >= 3 and axes.index("C") < axes.index("Y"):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = arr[..., None]

        rh, rw = arr.shape[:2]
        if (rh, rw) == (h, w) and y0c == y0i and x0c == x0i:
            return arr

        # Pad into (h,w,C)
        out = np.zeros((h, w, arr.shape[-1]), dtype=arr.dtype)
        dy = y0c - y0i
        dx = x0c - x0i
        out[dy : dy + rh, dx : dx + rw, :] = arr
        return out


def _normalize_to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    # percentile stretch, per patch
    lo, hi = np.percentile(rgb, (1, 99))
    if hi > lo:
        out = ((rgb.astype(np.float32) - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
        return out
    mx = float(np.max(rgb))
    if mx > 0:
        return (rgb.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)
    return np.zeros(rgb.shape, dtype=np.uint8)


def read_region_rgb(path: str, *, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """
    Read a window and return uint8 RGB (h,w,3).
    If the image is single-channel, it is replicated to RGB.
    """
    info = get_ome_info(path)
    if info.n_channels is None:
        ch = [0]
    else:
        ch = [0, 1, 2] if info.n_channels >= 3 else [0]
    arr = read_region_channels(path, y0=y0, x0=x0, h=h, w=w, channels=ch)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return _normalize_to_uint8(arr)

