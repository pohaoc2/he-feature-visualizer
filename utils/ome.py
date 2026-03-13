"""OME-TIFF utilities: pixel size, zarr access, overview reading."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import warnings

import numpy as np
import tifffile
import zarr

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _resolve_channel_axis(axes: str) -> str | None:
    """Return the preferred channel-like axis label from an axes string.

    Priority order matches common OME layouts:
      1) ``C`` (channels)
      2) ``I`` (image index / channels in some exports)
      3) ``S`` (samples, e.g. RGB YXS)
    """
    axes_up = axes.upper()
    for ax in ("C", "I", "S"):
        if ax in axes_up:
            return ax
    return None


class _NumpyStore:
    """Array-like fallback used when tifffile.aszarr is unavailable.

    This keeps the same slicing contract expected by patch readers
    (``shape``, ``ndim``, ``__getitem__``), but stores data in-memory.
    """

    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape
        self.ndim = self._arr.ndim

    def __getitem__(self, item):
        return self._arr[item]


def _safe_float(x: str | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def get_ome_mpp(tif: tifffile.TiffFile) -> tuple[float | None, float | None]:
    """Return ``(mpp_x, mpp_y)`` in um/pixel from OME-XML, or ``(None, None)``."""
    ome_xml = getattr(tif, "ome_metadata", None)
    if not ome_xml and hasattr(tif, "pages") and len(tif.pages) > 0:
        ome_xml = getattr(tif.pages[0], "description", None)
    if not ome_xml:
        return None, None
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None, None
    pixels = root.find(".//ome:Pixels", OME_NS)
    if pixels is None:
        return None, None
    return _safe_float(pixels.get("PhysicalSizeX")), _safe_float(
        pixels.get("PhysicalSizeY")
    )


def open_zarr_store(tif: tifffile.TiffFile) -> zarr.Array | _NumpyStore:
    """Return a read-only array-like store from an open ``TiffFile``.

    Preferred path uses ``tif.series[0].aszarr()``. Some version combinations
    of ``tifffile`` and ``zarr`` raise a ``TypeError`` while constructing the
    zarr store; in that case we fall back to an in-memory NumPy store.
    """

    try:
        raw = zarr.open(tif.series[0].aszarr(), mode="r")
        if isinstance(raw, zarr.Array):
            return raw
        return raw["0"]
    except TypeError as exc:
        msg = str(exc)
        if "ZarrTiffStore" not in msg:
            raise
        warnings.warn(
            "tifffile.aszarr is unavailable with current zarr/tifffile versions; "
            "falling back to memory-mapped array store.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Use tifffile's memmap output mode to avoid loading whole-slide data
        # into RAM when zarr access is unavailable in this environment.
        return _NumpyStore(tif.series[0].asarray(out="memmap"))


def get_image_dims(tif: tifffile.TiffFile) -> tuple[int, int, str]:
    """Return ``(img_w, img_h, axes_upper)`` from an open ``TiffFile``."""
    series = tif.series[0]
    axes = series.axes.upper()
    shape = series.shape
    return shape[axes.index("X")], shape[axes.index("Y")], axes


def read_overview_chw(
    store: zarr.Array | _NumpyStore,
    axes: str,
    img_h: int,
    img_w: int,
    ds: int,
) -> np.ndarray:
    """Read a subsampled overview from a zarr store as ``(C, H, W)``.

    Works for any axis ordering (CYX, YXC, XYCS, etc.) by building
    axis-aware slices then transposing to canonical ``(C, Y, X)`` order.
    If there is no C axis the result is ``(1, H, W)``.
    """
    ax_up = axes.upper()
    ch_axis = _resolve_channel_axis(ax_up)
    h_trunc = (img_h // ds) * ds
    w_trunc = (img_w // ds) * ds

    sl: list[int | slice] = []
    for ax in ax_up:
        if ch_axis is not None and ax == ch_axis:
            sl.append(slice(None))
        elif ax == "Y":
            sl.append(slice(0, h_trunc, ds))
        elif ax == "X":
            sl.append(slice(0, w_trunc, ds))
        else:
            sl.append(0)

    arr = np.array(store[tuple(sl)])

    active: list[str] = []
    for ax in ax_up:
        if ch_axis is not None and ax == ch_axis:
            active.append("C")
        elif ax in ("Y", "X"):
            active.append(ax)

    if "C" in active:
        target = [ax for ax in ("C", "Y", "X") if ax in active]
        if active != target:
            perm = [active.index(ax) for ax in target]
            arr = arr.transpose(perm)
    else:
        target = [ax for ax in ("Y", "X") if ax in active]
        if active != target:
            perm = [active.index(ax) for ax in target]
            arr = arr.transpose(perm)
        arr = arr[np.newaxis]

    return arr
