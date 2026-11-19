"""Shared helpers: hfv Stage-4 (stages/multiplex_layers.py) vasculature / oxygen / glucose, computed from the raw
multiplex OME-TIFF without the per-patch multiplex npy files.

Every number is produced by the hfv functions themselves (imported unmodified from the hfv checkout given as HFV):
  vasculature : per tile, level-0 crop of CD31 (+ SMA if sma=True) ->
                percentile_norm -> binarize_otsu [-> refine_vasculature_with_sma] -> cleanup_vasculature_mask(0, 0, 0)
                -> apply_vessel_mask_quality_fallback(0.98), exactly as multiplex_layers._ml_patch_worker
                (the crop equals what stages/patchify.py writes to multiplex/<tile>.npy: raw uint16 level-0 window,
                he_mpp == mx_mpp so the window is [y0:y0+256, x0:x0+256]).
  oxygen/glucose : read_wsi_channel_stack(tiff, ds, [CD31, Ki67, CD68(, SMA)]) -> solve_wsi_pde_map(...) once per field,
                then extract_patch_from_coarse per tile, as multiplex_layers.main + _ml_patch_worker.
The channel order passed to read_wsi_channel_stack is the one multiplex_layers.main builds
([cd31, ki67, cd68, sma, pcna], PCNA absent from the Orion panel).
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

HFV = os.environ.get("HFV")
if not HFV:
    raise SystemExit("set HFV to the hfv code checkout")
sys.path.insert(0, HFV)

from stages.multiplex_layers_lib.masks import (  # noqa: E402
    apply_vessel_mask_quality_fallback,
    binarize_otsu,
    cleanup_vasculature_mask,
    refine_vasculature_with_sma,
)
from stages.multiplex_layers_lib.wsi_pde import (  # noqa: E402
    extract_patch_from_coarse,
    read_wsi_channel_stack,
    solve_wsi_pde_map,
)
from utils.normalize import percentile_norm  # noqa: E402

PATCH = 256
# multiplex_layers.py argparse defaults (not exposed here; the CRC33 run did not change them, see README)
K_BASE = 0.1           # --oxygen/--glucose-consumption-base
DEMAND_W = 0.3         # --oxygen/--glucose-consumption-demand-weight
CD68_W = 0.1           # --cd68-consumption-weight
SMA_ADJ = 2            # --sma-adjacency-px
NOISY_MAX = 0.98       # --vasc-noisy-max-fraction


def channel_index(markers_csv: str) -> dict:
    names = [ln.strip() for ln in open(markers_csv) if ln.strip()]
    if names and names[0].lower() in ("marker_name", "target name", "channel_number,marker_name"):
        names = names[1:]
    low = [n.lower() for n in names]
    return {k: low.index(k.lower()) for k in ("CD31", "Ki67", "CD68", "SMA")}, names


def wsi_fields(tiff: str, ci: dict, ds: int, krogh_um: dict, mpp: float, sma: bool) -> dict:
    """krogh_um = {'oxygen': L_o2, 'glucose': L_glc} (µm). Returns coarse fields + the channel stack shape."""
    need = [ci["CD31"], ci["Ki67"], ci["CD68"]] + ([ci["SMA"]] if sma else [])
    stack = read_wsi_channel_stack(tiff, ds, channel_indices=need)
    out = {"_shape": list(stack.shape)}
    for name, L in krogh_um.items():
        out[name] = solve_wsi_pde_map(
            channel_stack=stack, cd31_idx=0, ki67_idx=1, mpp=mpp, ds=ds, krogh_um=L,
            k_base=K_BASE, demand_weight=DEMAND_W, immune_weight=CD68_W,
            sma_idx=3 if sma else None, sma_adjacency_px=SMA_ADJ, cd68_idx=2, pcna_idx=None)
    del stack
    return out


def vessel_mask(cd31_raw: np.ndarray, sma_raw: np.ndarray | None) -> np.ndarray:
    cd31_base = binarize_otsu(percentile_norm(cd31_raw.astype(np.float32)))
    m = cd31_base.copy()
    if sma_raw is not None:
        m = refine_vasculature_with_sma(m, binarize_otsu(percentile_norm(sma_raw.astype(np.float32))),
                                        adjacency_px=SMA_ADJ)
    m = cleanup_vasculature_mask(m, open_kernel_size=0, close_kernel_size=0, min_area=0)
    m, _ = apply_vessel_mask_quality_fallback(candidate_mask=m, cd31_fallback_mask=cd31_base,
                                              noisy_max_fraction=NOISY_MAX)
    return m.astype(bool)


def crop(plane: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """patchify read_multiplex_patch: uint16 window, zero-padded where it leaves the image."""
    h, w = plane.shape
    out = np.zeros((PATCH, PATCH), dtype=np.uint16)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + PATCH), min(w, x0 + PATCH)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = plane[ys:ye, xs:xe]
    return out


def coarse_tile(field: np.ndarray, x0: int, y0: int, ds: int, mpp: float) -> np.ndarray:
    return extract_patch_from_coarse(field, x0, y0, PATCH, mpp, mpp, ds)


def npy_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, arr, allow_pickle=False)
    return b.getvalue()


def read_plane(tiff: str, ch: int) -> np.ndarray:
    """Full level-0 plane of one channel (uint16)."""
    import tifffile
    import zarr
    with tifffile.TiffFile(tiff) as tif:
        s = tif.series[0]
        ax = s.axes.upper()
        raw = zarr.open(s.aszarr(), mode="r")
        lv = raw if isinstance(raw, zarr.Array) else raw["0"]
        sl = tuple(ch if a in ("C", "I", "S") else slice(None) if a in ("Y", "X") else 0 for a in ax)
        return np.asarray(lv[sl]).astype(np.uint16)


def ome_info(tiff: str) -> dict:
    import tifffile
    from utils.ome import get_ome_mpp
    with tifffile.TiffFile(tiff) as tif:
        s = tif.series[0]
        mpp = get_ome_mpp(tif)
        return dict(axes=s.axes, shape=list(s.shape), levels=[list(lv.shape) for lv in s.levels],
                    dtype=str(s.dtype), mpp=list(mpp))
