"""Windowed patch readers and patch-grid utilities."""

import numpy as np
import zarr

from utils.normalize import percentile_to_uint8


def get_tissue_patches(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    patch_size: int,
    stride: int,
    tissue_min: float,
    downsample: int,
) -> list[tuple[int, int]]:
    """Return list of (x0, y0) level-0 patch coords that meet tissue threshold.

    Only patches satisfying x0+patch_size <= img_w and y0+patch_size <= img_h
    are considered (no padding).
    """
    kept = []
    y0 = 0
    while y0 + patch_size <= img_h:
        x0 = 0
        while x0 + patch_size <= img_w:
            my0 = y0 // downsample
            mx0 = x0 // downsample
            my1 = max(my0 + 1, (y0 + patch_size) // downsample)
            mx1 = max(mx0 + 1, (x0 + patch_size) // downsample)
            my1 = min(my1, mask.shape[0])
            mx1 = min(mx1, mask.shape[1])
            region = mask[my0:my1, mx0:mx1]
            if region.size > 0 and float(region.mean()) >= tissue_min:
                kept.append((x0, y0))
            x0 += stride
        y0 += stride
    return kept


def get_patch_grid(
    img_w: int, img_h: int, patch_size: int, stride: int
) -> list[tuple[int, int]]:
    """Return list of (i, j) patch indices that are fully within the image.

    Patch top-left pixel coordinates: x0 = j * stride, y0 = i * stride.
    Only patches satisfying x0 + patch_size <= img_w and
    y0 + patch_size <= img_h are included.
    """
    coords = []
    i = 0
    while True:
        y0 = i * stride
        if y0 + patch_size > img_h:
            break
        j = 0
        while True:
            x0 = j * stride
            if x0 + patch_size > img_w:
                break
            coords.append((i, j))
            j += 1
        i += 1
    return coords


def _clip_and_read(
    store, axes: str, img_w: int, img_h: int, y0: int, x0: int, size_y: int, size_x: int
):
    """Read a clipped region from the store and return (arr, dy, dx, rh, rw).

    dy, dx: offsets within the output patch where valid data begins (for zero-padding).
    rh, rw: height and width of the clipped read region.
    """
    if not (
        hasattr(store, "__getitem__")
        and hasattr(store, "shape")
        and hasattr(store, "ndim")
    ):
        raw = zarr.open(store, mode="r")
        if hasattr(raw, "__getitem__") and hasattr(raw, "shape"):
            store = raw
        else:
            store = raw["0"]

    y0i = int(y0)
    x0i = int(x0)
    y1i = y0i + int(size_y)
    x1i = x0i + int(size_x)

    y0c = max(0, min(y0i, img_h))
    x0c = max(0, min(x0i, img_w))
    y1c = max(0, min(y1i, img_h))
    x1c = max(0, min(x1i, img_w))

    sl = []
    for ax in axes:
        if ax in ("C", "I", "S"):
            sl.append(slice(None))
        elif ax == "Y":
            sl.append(slice(y0c, y1c))
        elif ax == "X":
            sl.append(slice(x0c, x1c))
        else:
            sl.append(0)

    arr = np.array(store[tuple(sl)])
    dy = y0c - y0i
    dx = x0c - x0i
    rh = max(0, y1c - y0c)
    rw = max(0, x1c - x0c)
    return arr, dy, dx, rh, rw


def read_he_patch(
    zarr_store, axes: str, img_w: int, img_h: int, y0: int, x0: int, size: int
) -> np.ndarray:
    """Read H&E patch as uint8 RGB (size, size, 3).

    Handle axes permutations (CYX, YXC, YX, etc.).
    If store dtype != uint8: percentile normalize (p1/p99) to uint8.
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size, size
    )

    if arr.ndim == 3:
        c_pos = next((axes.index(a) for a in ("C", "S") if a in axes), -1)
        y_pos = axes.index("Y") if "Y" in axes else -1
        if c_pos != -1 and y_pos != -1 and c_pos < y_pos:
            arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = percentile_to_uint8(arr)
    else:
        arr = arr.astype(np.uint8)

    if arr.shape[0] != size or arr.shape[1] != size:
        out = np.zeros((size, size, 3), dtype=np.uint8)
        if rh > 0 and rw > 0:
            out[dy : dy + rh, dx : dx + rw] = arr[:rh, :rw]
        return out

    return arr


def read_mask_patch(
    zarr_store, axes: str, img_w: int, img_h: int, y0: int, x0: int, size: int
) -> np.ndarray:
    """Read a cell-segmentation mask patch as uint32 (size, size).

    The mask is in H&E pixel space; no coordinate transform is required.
    Pixel values are integer label IDs (0 = background).
    Clip to image bounds, zero-pad if the patch extends beyond the edge.
    """
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size, size
    )

    if arr.ndim == 3:
        active = [ax for ax in axes if ax in ("C", "Y", "X")]
        if "C" in active and active.index("C") > 0:
            perm = [active.index(a) for a in ("C", "Y", "X") if a in active]
            arr = arr.transpose(perm)
        arr = arr[0]

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr).astype(np.uint32)
    else:
        arr = arr.astype(np.uint32)

    if arr.shape[0] != size or arr.shape[1] != size:
        out = np.zeros((size, size), dtype=np.uint32)
        if rh > 0 and rw > 0:
            out[dy : dy + rh, dx : dx + rw] = arr[:rh, :rw]
        return out

    return arr


def read_multiplex_patch(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    y0: int,
    x0: int,
    size_y: int,
    size_x: int,
    channel_indices: list[int],
) -> np.ndarray:
    """Read multiplex patch for specific channel indices.

    Returns (C, size_y, size_x) uint16 where C = len(channel_indices).
    Handle axes permutations (CYX, YXC, etc.).
    Clip to image bounds, zero-pad if needed.
    """
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size_y, size_x
    )

    active_axes = [ax for ax in axes if ax in ("C", "I", "Y", "X")]
    ch_ax = next((ax for ax in ("C", "I") if ax in active_axes), None)

    if ch_ax is not None:
        target = [ax for ax in (ch_ax, "Y", "X") if ax in active_axes]
        if active_axes != target:
            perm = [active_axes.index(ax) for ax in target]
            arr = arr.transpose(perm)
        arr = arr[channel_indices]
    else:
        target = [ax for ax in ("Y", "X") if ax in active_axes]
        if active_axes != target:
            perm = [active_axes.index(ax) for ax in target]
            arr = arr.transpose(perm)
        arr = np.stack([arr] * len(channel_indices), axis=0)

    arr = arr.astype(np.uint16)
    n_ch = arr.shape[0]
    if arr.shape[1] != size_y or arr.shape[2] != size_x:
        out = np.zeros((n_ch, size_y, size_x), dtype=np.uint16)
        if rh > 0 and rw > 0:
            out[:, dy : dy + rh, dx : dx + rw] = arr[:, :rh, :rw]
        return out

    return arr
