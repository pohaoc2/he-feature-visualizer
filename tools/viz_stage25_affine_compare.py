#!/usr/bin/env python3
"""
viz_stage25_affine_compare.py — Stage-1 vs Stage-2.5 affine alignment comparison.

Creates a 4-panel figure:
  a) Cropped original H&E
  b) Cropped MX DNA channel aligned with Stage-1 affine (warp_matrix)
  c) Overlap of (a) and (b)
  d) Overlap of (a) and MX DNA aligned with Stage-2.5 updated affine

The Stage-2.5 "updated affine" is composed as:
  M_stage25 = M_stage1 @ M_icp
where:
  M_stage1 = index.json["warp_matrix"]           (H&E -> MX)
  M_icp    = index_icp_tps.json["icp_matrix"]    (H&E -> corrected H&E)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from utils.normalize import percentile_to_uint8
from utils.ome import open_zarr_store, read_overview_chw


def _read_channel_overview(
    store, axes: str, img_h: int, img_w: int, ds: int, c_idx: int
) -> np.ndarray:
    """Read a single channel at overview resolution, returning (H, W)."""
    ax = axes.upper()
    h_t = (img_h // ds) * ds
    w_t = (img_w // ds) * ds
    sl: list[int | slice] = []
    for a in ax:
        if a == "C":
            sl.append(c_idx)
        elif a == "Y":
            sl.append(slice(0, h_t, ds))
        elif a == "X":
            sl.append(slice(0, w_t, ds))
        else:
            sl.append(0)
    arr = np.array(store[tuple(sl)])
    active = [a for a in ax if a in ("Y", "X")]
    target = [a for a in ("Y", "X") if a in active]
    if active != target:
        arr = arr.transpose([active.index(a) for a in target])
    return arr


def _to_mat3(m_2x3: np.ndarray) -> np.ndarray:
    m3 = np.eye(3, dtype=np.float64)
    m3[:2, :] = m_2x3.astype(np.float64)
    return m3


def _compose_stage25_affine(m_stage1: np.ndarray, m_icp: np.ndarray) -> np.ndarray:
    """Compose H&E->MX affine with ICP correction in H&E space."""
    m_stage1_3 = _to_mat3(m_stage1)
    m_icp_3 = _to_mat3(m_icp)
    m_new_3 = m_stage1_3 @ m_icp_3
    return m_new_3[:2, :]


def _warp_mx_to_he_overview(
    mx_ch_ov: np.ndarray, m_full: np.ndarray, ds: int, out_w: int, out_h: int
) -> np.ndarray:
    """Warp MX overview channel to H&E overview frame."""
    m_disp = m_full.astype(np.float64).copy()
    m_disp[0, 2] /= ds
    m_disp[1, 2] /= ds
    return cv2.warpAffine(
        mx_ch_ov.astype(np.float32),
        m_disp,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )


def _colorize_hot(gray_f32: np.ndarray) -> np.ndarray:
    u8 = percentile_to_uint8(gray_f32.astype(np.float32))
    rgba = matplotlib.colormaps["hot"](u8.astype(np.float32) / 255.0)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _blend(he_rgb: np.ndarray, mx_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return (
        he_rgb.astype(np.float32) * (1.0 - alpha) + mx_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)


def _crop_bounds(
    he_w: int,
    he_h: int,
    crop_x0: int | None,
    crop_y0: int | None,
    crop_size: int,
) -> tuple[int, int, int, int]:
    size = max(1, int(crop_size))
    if crop_x0 is None:
        crop_x0 = max(0, (he_w - size) // 2)
    if crop_y0 is None:
        crop_y0 = max(0, (he_h - size) // 2)
    x0 = max(0, min(int(crop_x0), max(0, he_w - 1)))
    y0 = max(0, min(int(crop_y0), max(0, he_h - 1)))
    x1 = min(he_w, x0 + size)
    y1 = min(he_h, y0 + size)
    return x0, y0, x1, y1


def main() -> None:
    p = argparse.ArgumentParser(
        description="4-panel Stage-1 vs Stage-2.5 affine alignment visualization."
    )
    p.add_argument("--he-image", required=True, help="H&E OME-TIFF path.")
    p.add_argument("--multiplex-image", required=True, help="Multiplex OME-TIFF path.")
    p.add_argument("--index-stage1", required=True, help="Stage-1 index.json path.")
    p.add_argument(
        "--index-stage25",
        required=True,
        help="Stage-2.5 index file path (e.g., index_icp_tps.json).",
    )
    p.add_argument(
        "--dna-channel",
        type=int,
        default=0,
        help="Multiplex DNA channel index (default: 0).",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=8,
        help="Overview downsample factor used for plotting.",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Crop size in full-resolution H&E pixels.",
    )
    p.add_argument(
        "--crop-x0",
        type=int,
        default=None,
        help="Optional crop top-left x in full-resolution H&E pixels.",
    )
    p.add_argument(
        "--crop-y0",
        type=int,
        default=None,
        help="Optional crop top-left y in full-resolution H&E pixels.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for panels c/d.",
    )
    p.add_argument("--out-png", required=True, help="Output PNG path.")
    args = p.parse_args()

    ds = max(1, int(args.downsample))
    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    with open(args.index_stage1, encoding="utf-8") as f:
        stage1 = json.load(f)
    with open(args.index_stage25, encoding="utf-8") as f:
        stage25 = json.load(f)

    if "warp_matrix" not in stage1:
        raise ValueError(f"{args.index_stage1} has no 'warp_matrix'.")
    if "icp_matrix" not in stage25:
        raise ValueError(
            f"{args.index_stage25} has no 'icp_matrix'. Run Stage 2.5 first."
        )

    m_stage1 = np.array(stage1["warp_matrix"], dtype=np.float64)
    m_icp = np.array(stage25["icp_matrix"], dtype=np.float64)
    m_stage25 = _compose_stage25_affine(m_stage1, m_icp)

    with tifffile.TiffFile(args.he_image) as he_tif:
        he_s = he_tif.series[0]
        he_ax = he_s.axes.upper()
        he_h = int(he_s.shape[he_ax.index("Y")])
        he_w = int(he_s.shape[he_ax.index("X")])
        he_store = open_zarr_store(he_tif)
        he_chw = read_overview_chw(he_store, he_ax, he_h, he_w, ds)
        he_chw = he_chw[:3] if he_chw.shape[0] >= 3 else np.repeat(he_chw[:1], 3, 0)
        if he_chw.dtype != np.uint8:
            he_chw = percentile_to_uint8(he_chw)
        he_rgb = np.moveaxis(he_chw.astype(np.uint8), 0, -1)

    with tifffile.TiffFile(args.multiplex_image) as mx_tif:
        mx_s = mx_tif.series[0]
        mx_ax = mx_s.axes.upper()
        mx_h = int(mx_s.shape[mx_ax.index("Y")])
        mx_w = int(mx_s.shape[mx_ax.index("X")])
        mx_store = open_zarr_store(mx_tif)
        mx_ch_ov = _read_channel_overview(
            mx_store,
            mx_ax,
            mx_h,
            mx_w,
            ds,
            c_idx=int(args.dna_channel),
        ).astype(np.float32)

    h_he_ov, w_he_ov = he_rgb.shape[0], he_rgb.shape[1]
    sx = he_w / float(max(1, w_he_ov))
    sy = he_h / float(max(1, h_he_ov))

    mx_stage1_ov = _warp_mx_to_he_overview(mx_ch_ov, m_stage1, ds, w_he_ov, h_he_ov)
    mx_stage25_ov = _warp_mx_to_he_overview(mx_ch_ov, m_stage25, ds, w_he_ov, h_he_ov)

    mx_stage1_rgb = _colorize_hot(mx_stage1_ov)
    mx_stage25_rgb = _colorize_hot(mx_stage25_ov)

    x0, y0, x1, y1 = _crop_bounds(
        he_w=he_w,
        he_h=he_h,
        crop_x0=args.crop_x0,
        crop_y0=args.crop_y0,
        crop_size=args.crop_size,
    )
    x0_ov = int(np.floor(x0 / sx))
    y0_ov = int(np.floor(y0 / sy))
    x1_ov = int(np.ceil(x1 / sx))
    y1_ov = int(np.ceil(y1 / sy))
    x0_ov = max(0, min(x0_ov, w_he_ov - 1))
    y0_ov = max(0, min(y0_ov, h_he_ov - 1))
    x1_ov = max(x0_ov + 1, min(x1_ov, w_he_ov))
    y1_ov = max(y0_ov + 1, min(y1_ov, h_he_ov))

    he_crop = he_rgb[y0_ov:y1_ov, x0_ov:x1_ov]
    mx_stage1_crop = mx_stage1_rgb[y0_ov:y1_ov, x0_ov:x1_ov]
    mx_stage25_crop = mx_stage25_rgb[y0_ov:y1_ov, x0_ov:x1_ov]
    ov_stage1 = _blend(he_crop, mx_stage1_crop, alpha)
    ov_stage25 = _blend(he_crop, mx_stage25_crop, alpha)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    ax_a.imshow(he_crop)
    ax_a.set_title("a) Cropped original H&E")
    ax_b.imshow(mx_stage1_crop)
    ax_b.set_title("b) MX DNA (Stage-1 aligned)")
    ax_c.imshow(ov_stage1)
    ax_c.set_title("c) Overlap: H&E + Stage-1 aligned MX DNA")
    ax_d.imshow(ov_stage25)
    ax_d.set_title("d) Overlap: H&E + Stage-2.5 updated affine MX DNA")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")

    fig.suptitle(
        f"Crop HE px: x=[{x0},{x1}) y=[{y0},{y1}) | ds={ds} | "
        f"dna_channel={args.dna_channel}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")
    print(f"Stage-1 affine:\n{m_stage1}")
    print(f"ICP affine:\n{m_icp}")
    print(f"Stage-2.5 composed affine (M_stage1 @ M_icp):\n{m_stage25}")


if __name__ == "__main__":
    main()
