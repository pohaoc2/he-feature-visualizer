"""WSI-scale PDE solver for oxygen and glucose proxy maps.

Runs steady-state diffusion on the full WSI at a coarse downsampling factor,
then extracts per-patch slices from the solved field.  This is the physically
correct approach because the Krogh diffusion radius (160 µm for O₂, 450 µm
for glucose) is larger than a single 256-px patch, so per-patch PDE cannot
capture the true tissue-scale gradients.

Usage (called from multiplex_layers.py)
---------------------------------------
    stack = read_wsi_channel_stack(tiff_path, ds=8)
    o2_coarse = solve_wsi_pde_map(stack, cd31_idx=2, ki67_idx=16,
                                  mpp=0.325, ds=8, krogh_um=160.0)
    for x0, y0 in patch_coords:
        patch = extract_patch_from_coarse(o2_coarse, x0, y0, 256,
                                          warp_matrix, he_mpp, mx_mpp, ds=8)
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import tifffile

from stages.multiplex_layers_lib.masks import binarize_otsu, refine_vasculature_with_sma
from stages.multiplex_layers_lib.pde import (
    build_consumption_map,
    compute_metabolic_demand_map,
)
from utils.normalize import percentile_norm


def read_wsi_channel_stack(
    tiff_path: str | pathlib.Path,
    ds: int,
    channel_indices: list[int] | None = None,
) -> np.ndarray:
    """Read selected channels from an OME-TIFF using the best pyramid level.

    Parameters
    ----------
    tiff_path:
        Path to the OME-TIFF file.
    ds:
        Target downsampling factor relative to full resolution.
        The finest pyramid level with ``actual_ds <= ds`` is used; any
        remaining factor is applied via striding at read time.
    channel_indices:
        Optional list of channel indices to read from the full channel axis.
        Pass only the channels you need to keep memory manageable on large WSIs
        (e.g. ``[cd31_idx, ki67_idx, cd68_idx]``).  When ``None`` all channels
        are read.  The returned array's channel axis 0 corresponds to
        ``channel_indices[0]``, etc.

    Returns
    -------
    ``(len(channel_indices), H_c, W_c)`` float32 array.
    """
    import zarr as _zarr

    with tifffile.TiffFile(str(tiff_path)) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        ch_ax = None
        for _a in ("C", "I", "S"):
            if _a in axes:
                ch_ax = _a
                break

        full_h = series.shape[axes.index("Y")]
        n_total_ch = series.shape[axes.index(ch_ax)] if ch_ax else 1

        # ------------------------------------------------------------------
        # Pick the finest pyramid level whose actual_ds <= target ds.
        # This avoids loading the full-resolution plane and striding through it.
        # ------------------------------------------------------------------
        best_lvl_idx = 0
        best_actual_ds = 1
        for lvl_idx, lv in enumerate(series.levels):
            lv_h = lv.shape[axes.index("Y")]
            actual_ds = max(1, round(full_h / lv_h))
            if actual_ds <= ds:
                best_lvl_idx = lvl_idx
                best_actual_ds = actual_ds

        extra_stride = max(1, round(ds / best_actual_ds))

        # ------------------------------------------------------------------
        # Open the zarr group and select the chosen level.
        # ------------------------------------------------------------------
        try:
            raw = _zarr.open(series.aszarr(), mode="r")
            if isinstance(raw, _zarr.Array):
                lv_arr = raw  # single-level TIFF, no pyramid group
            else:
                lv_arr = raw[str(best_lvl_idx)]
        except TypeError:
            # Fallback: memory-map via tifffile (slower but always works)
            lv_arr = series.levels[best_lvl_idx].asarray(out="memmap")

        lv_h = lv_arr.shape[axes.index("Y")]
        lv_w = lv_arr.shape[axes.index("X")]
        h_trunc = (lv_h // extra_stride) * extra_stride
        w_trunc = (lv_w // extra_stride) * extra_stride
        out_h = h_trunc // extra_stride
        out_w = w_trunc // extra_stride

        ch_to_read = list(range(n_total_ch)) if channel_indices is None else channel_indices

        out = np.zeros((len(ch_to_read), out_h, out_w), dtype=np.float32)
        for out_c, ch in enumerate(ch_to_read):
            sl: list = []
            for ax in axes:
                if ax == ch_ax:
                    sl.append(ch)
                elif ax == "Y":
                    sl.append(slice(0, h_trunc, extra_stride))
                elif ax == "X":
                    sl.append(slice(0, w_trunc, extra_stride))
                else:
                    sl.append(0)
            out[out_c] = np.array(lv_arr[tuple(sl)], dtype=np.float32)

    return out


def solve_wsi_pde_map(
    channel_stack: np.ndarray,
    cd31_idx: int,
    ki67_idx: int,
    mpp: float,
    ds: int,
    krogh_um: float,
    k_base: float = 0.1,
    demand_weight: float = 0.3,
    immune_weight: float = 0.1,
    sma_idx: int | None = None,
    sma_adjacency_px: int = 2,
    cd68_idx: int | None = None,
    pcna_idx: int | None = None,
    max_iters: int | None = None,
    tol: float = 1e-4,
) -> np.ndarray:
    """Compute WSI-scale nutrient proxy using exponential decay + demand modulation.

    Uses an O(N) approach: Euclidean distance transform + spatially varying
    decay length (WKB approximation of the steady-state PDE).

    For homogeneous k (no demand), ``exp(-dist/L)`` is the *exact* solution
    to ``D∇²u − ku = 0`` (Krogh cylinder model).  For spatially varying k(x),
    the local decay length is adjusted::

        L(x) = krogh_um / mpp_coarse / sqrt(k(x) / k_base)

    so high-demand regions (high Ki67/CD68) appear more depleted.  This is
    the WKB (geometric optics) approximation — exact for slowly varying k.

    Returns ``(H_c, W_c)`` float32 map in ``[0, 1]``:
    ``1`` = vessel pixel (fully supplied), ``0`` = fully depleted.
    """
    cd31_norm = percentile_norm(channel_stack[cd31_idx])
    vessel_mask = binarize_otsu(cd31_norm)
    if sma_idx is not None:
        sma_norm = percentile_norm(channel_stack[sma_idx])
        sma_mask = binarize_otsu(sma_norm)
        vessel_mask = refine_vasculature_with_sma(
            vessel_mask, sma_mask, adjacency_px=sma_adjacency_px
        )

    pcna_raw = channel_stack[pcna_idx] if pcna_idx is not None else None
    demand_map = compute_metabolic_demand_map(channel_stack[ki67_idx], pcna_raw)
    immune_map = (
        percentile_norm(channel_stack[cd68_idx]) if cd68_idx is not None else None
    )

    from scipy.ndimage import distance_transform_edt

    mpp_coarse = mpp * ds
    L_base = krogh_um / mpp_coarse  # decay length in coarse pixels for k = k_base

    # Build spatially varying consumption map k(x).
    k_map = build_consumption_map(
        demand_map,
        base_rate=k_base,
        demand_weight=demand_weight,
        immune_map=immune_map,
        immune_weight=immune_weight,
    )

    # Exponential decay with local demand modulation (WKB approximation).
    # For the homogeneous PDE (constant k), exp(-dist/L) is the exact solution.
    # With spatially varying k(x), L(x) = L_base / sqrt(k(x)/k_base) gives the
    # locally-adjusted decay length: high-demand regions appear more hypoxic.
    dist = distance_transform_edt(~vessel_mask).astype(np.float32)  # coarse pixels
    L_local = (L_base / np.sqrt(k_map / k_base)).astype(np.float32)
    u = np.where(vessel_mask, 1.0, np.exp(-dist / L_local)).astype(np.float32)
    return u


def extract_patch_from_coarse(
    coarse_arr: np.ndarray,
    x0: int,
    y0: int,
    patch_size: int,
    warp_matrix: list,
    he_mpp: float,
    mx_mpp: float,
    ds: int,
) -> np.ndarray:
    """Extract and resize a patch region from the coarse WSI PDE solution.

    Transforms the patch origin ``(x0, y0)`` from H&E full-resolution space to
    multiplex coarse space using ``warp_matrix``, crops the corresponding
    region, and resizes to ``(patch_size, patch_size)``.

    Parameters
    ----------
    coarse_arr:
        ``(H_c, W_c)`` float32 PDE solution from :func:`solve_wsi_pde_map`.
    x0, y0:
        Patch origin in H&E full-resolution pixels (as stored in index.json).
    patch_size:
        Output size in pixels (e.g. 256).
    warp_matrix:
        2×3 affine stored in index.json: ``[[a, b, tx], [c, d, ty]]``.
        Maps H&E full-res → multiplex full-res.
    he_mpp, mx_mpp:
        Microns-per-pixel for H&E and multiplex images.
    ds:
        Downsampling factor used when building ``coarse_arr``.

    Returns
    -------
    ``(patch_size, patch_size)`` float32 in ``[0, 1]``.
    """
    a, b, tx = warp_matrix[0]
    c, d, ty = warp_matrix[1]
    mx_x = a * x0 + b * y0 + tx
    mx_y = c * x0 + d * y0 + ty

    # Patch footprint in MX full-res pixels, then convert to coarse grid
    patch_size_mx = patch_size * (he_mpp / mx_mpp)
    size_c = patch_size_mx / ds

    H_c, W_c = coarse_arr.shape
    cx = int(np.clip(round(mx_x / ds), 0, W_c - 1))
    cy = int(np.clip(round(mx_y / ds), 0, H_c - 1))
    size_c_int = max(1, int(round(size_c)))

    region = coarse_arr[cy : min(cy + size_c_int, H_c), cx : min(cx + size_c_int, W_c)]
    if region.size == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    return cv2.resize(
        region, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
