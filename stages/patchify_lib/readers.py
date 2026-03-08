"""Windowed patch readers and patch-grid utilities."""

import cv2
import numpy as np
import zarr

from utils.normalize import percentile_to_uint8

from .registration import _he_patch_to_mx_local_affine, _transform_points_affine


def get_tissue_patches(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    patch_size: int,
    stride: int,
    tissue_min: float,
    downsample: int,
) -> list[tuple[int, int]]:
    """Return list of (x0, y0) level-0 patch coords that meet tissue threshold."""
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
    """Return list of (i, j) patch indices that are fully within the image."""
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
    """Read a clipped region from the store and return (arr, dy, dx, rh, rw)."""
    # Newer zarr releases may expose array types that are not zarr.Array
    # instances. Prefer array capability checks before attempting zarr.open().
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
        if ax == "C":
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
    """Read H&E patch as uint8 RGB (size, size, 3)."""
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size, size
    )

    if arr.ndim == 3:
        c_pos = axes.index("C") if "C" in axes else -1
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
    """Read a cell-segmentation mask patch as uint32 (size, size)."""
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size, size
    )

    if arr.ndim == 3:
        active = [ax for ax in axes if ax in ("C", "Y", "X")]
        if "C" in active and active.index("C") > 0:
            perm = [active.index(a) for a in ("C", "Y", "X") if a in active]
            arr = arr.transpose(perm)
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
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
    """Read multiplex patch for specific channel indices."""
    arr, dy, dx, rh, rw = _clip_and_read(
        zarr_store, axes, img_w, img_h, y0, x0, size_y, size_x
    )

    active_axes = [ax for ax in axes if ax in ("C", "Y", "X")]

    if "C" in active_axes:
        target = [ax for ax in ("C", "Y", "X") if ax in active_axes]
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


def multiplex_patch_overlap_fraction_affine(
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    mx_w: int,
    mx_h: int,
) -> float:
    """Fraction of a warped affine patch footprint that lies inside MX bounds."""
    if patch_size <= 1 or mx_w <= 0 or mx_h <= 0:
        return 0.0

    m_local = _he_patch_to_mx_local_affine(m_full, he_x0, he_y0)
    corners = np.array(
        [
            [0.0, 0.0],
            [patch_size - 1.0, 0.0],
            [patch_size - 1.0, patch_size - 1.0],
            [0.0, patch_size - 1.0],
        ],
        dtype=np.float64,
    )
    poly = _transform_points_affine(m_local, corners).astype(np.float32)
    full_area = float(abs(cv2.contourArea(poly)))
    if full_area <= 1e-9:
        return 0.0

    bounds = np.array(
        [
            [0.0, 0.0],
            [float(mx_w), 0.0],
            [float(mx_w), float(mx_h)],
            [0.0, float(mx_h)],
        ],
        dtype=np.float32,
    )
    inter_area, _ = cv2.intersectConvexConvex(poly, bounds)
    frac = float(inter_area) / full_area
    return float(np.clip(frac, 0.0, 1.0))


def read_multiplex_patch_affine(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    channel_indices: list[int],
) -> tuple[np.ndarray, bool]:
    """Read an affine-aligned MX patch in H&E patch frame."""
    n_ch = len(channel_indices)
    out = np.zeros((n_ch, patch_size, patch_size), dtype=np.uint16)

    m_local = _he_patch_to_mx_local_affine(m_full, he_x0, he_y0)
    corners = np.array(
        [
            [0.0, 0.0],
            [patch_size - 1.0, 0.0],
            [0.0, patch_size - 1.0],
            [patch_size - 1.0, patch_size - 1.0],
        ],
        dtype=np.float64,
    )
    corners_mx = _transform_points_affine(m_local, corners)
    xs = corners_mx[:, 0]
    ys = corners_mx[:, 1]
    inside = bool(
        np.all(xs >= 0.0)
        and np.all(ys >= 0.0)
        and np.all(xs < float(img_w))
        and np.all(ys < float(img_h))
    )

    x_min = int(np.floor(xs.min()))
    x_max = int(np.ceil(xs.max())) + 1
    y_min = int(np.floor(ys.min()))
    y_max = int(np.ceil(ys.max())) + 1
    src_w = max(1, x_max - x_min)
    src_h = max(1, y_max - y_min)

    src = read_multiplex_patch(
        zarr_store,
        axes,
        img_w,
        img_h,
        y0=y_min,
        x0=x_min,
        size_y=src_h,
        size_x=src_w,
        channel_indices=channel_indices,
    )

    m_patch = m_local.copy()
    m_patch[0, 2] -= x_min
    m_patch[1, 2] -= y_min
    m_patch = m_patch.astype(np.float32)

    for c in range(n_ch):
        warped = cv2.warpAffine(
            src[c].astype(np.float32),
            m_patch,
            (patch_size, patch_size),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[c] = np.clip(np.rint(warped), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return out, inside


def _deform_he_patch_to_mx_maps(
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    flow_dx_ov: np.ndarray,
    flow_dy_ov: np.ndarray,
    he_full_w: int,
    he_full_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build destination-pixel -> MX coordinate maps for deformable sampling."""
    uu, vv = np.meshgrid(
        np.arange(patch_size, dtype=np.float32),
        np.arange(patch_size, dtype=np.float32),
    )
    x_he = uu + float(he_x0)
    y_he = vv + float(he_y0)

    h_ov, w_ov = flow_dx_ov.shape
    he_sx = he_full_w / float(max(1, w_ov))
    he_sy = he_full_h / float(max(1, h_ov))
    x_ov = x_he / he_sx
    y_ov = y_he / he_sy
    dx_ov = cv2.remap(
        flow_dx_ov.astype(np.float32),
        x_ov.astype(np.float32),
        y_ov.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    dy_ov = cv2.remap(
        flow_dy_ov.astype(np.float32),
        x_ov.astype(np.float32),
        y_ov.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    x_corr = x_he - dx_ov * he_sx
    y_corr = y_he - dy_ov * he_sy

    a, b, tx = map(float, m_full[0])
    c, d, ty = map(float, m_full[1])
    map_mx_x = a * x_corr + b * y_corr + tx
    map_mx_y = c * x_corr + d * y_corr + ty
    return map_mx_x, map_mx_y


def multiplex_patch_overlap_fraction_deform(
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    flow_dx_ov: np.ndarray,
    flow_dy_ov: np.ndarray,
    he_full_w: int,
    he_full_h: int,
    mx_w: int,
    mx_h: int,
) -> float:
    """Fraction of deform-warped destination pixels with valid MX sampling coords."""
    if patch_size <= 0 or mx_w <= 0 or mx_h <= 0:
        return 0.0
    map_mx_x, map_mx_y = _deform_he_patch_to_mx_maps(
        he_x0=he_x0,
        he_y0=he_y0,
        patch_size=patch_size,
        m_full=m_full,
        flow_dx_ov=flow_dx_ov,
        flow_dy_ov=flow_dy_ov,
        he_full_w=he_full_w,
        he_full_h=he_full_h,
    )
    inside = (
        (map_mx_x >= 0.0)
        & (map_mx_y >= 0.0)
        & (map_mx_x < float(mx_w))
        & (map_mx_y < float(mx_h))
    )
    return float(inside.mean())


def read_multiplex_patch_affine_deform(
    zarr_store,
    axes: str,
    img_w: int,
    img_h: int,
    he_x0: int,
    he_y0: int,
    patch_size: int,
    m_full: np.ndarray,
    channel_indices: list[int],
    flow_dx_ov: np.ndarray,
    flow_dy_ov: np.ndarray,
    he_full_w: int,
    he_full_h: int,
) -> tuple[np.ndarray, bool]:
    """Read a deformable-refined MX patch in H&E patch frame."""
    n_ch = len(channel_indices)
    out = np.zeros((n_ch, patch_size, patch_size), dtype=np.uint16)

    map_mx_x, map_mx_y = _deform_he_patch_to_mx_maps(
        he_x0=he_x0,
        he_y0=he_y0,
        patch_size=patch_size,
        m_full=m_full,
        flow_dx_ov=flow_dx_ov,
        flow_dy_ov=flow_dy_ov,
        he_full_w=he_full_w,
        he_full_h=he_full_h,
    )

    inside = bool(
        np.all(map_mx_x >= 0.0)
        and np.all(map_mx_y >= 0.0)
        and np.all(map_mx_x < float(img_w))
        and np.all(map_mx_y < float(img_h))
    )

    x_min = int(np.floor(float(map_mx_x.min())))
    x_max = int(np.ceil(float(map_mx_x.max()))) + 1
    y_min = int(np.floor(float(map_mx_y.min())))
    y_max = int(np.ceil(float(map_mx_y.max()))) + 1
    src_w = max(1, x_max - x_min)
    src_h = max(1, y_max - y_min)

    src = read_multiplex_patch(
        zarr_store,
        axes,
        img_w,
        img_h,
        y0=y_min,
        x0=x_min,
        size_y=src_h,
        size_x=src_w,
        channel_indices=channel_indices,
    )

    local_x = (map_mx_x - float(x_min)).astype(np.float32)
    local_y = (map_mx_y - float(y_min)).astype(np.float32)
    for ch in range(n_ch):
        warped = cv2.remap(
            src[ch].astype(np.float32),
            local_x,
            local_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[ch] = np.clip(np.rint(warped), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return out, inside
