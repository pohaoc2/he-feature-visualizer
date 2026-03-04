from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import pandas as pd
from scipy.ndimage import distance_transform_edt

from cell_state import assign_cell_state, compute_marker_thresholds, state_to_rgba
from ome_reader import get_ome_info, read_region_channels, read_region_rgb
from tissue_mask import tissue_fraction_rgb


def _stretch_to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    lo, hi = np.percentile(x, (1, 99))
    if hi > lo:
        return ((x.astype(np.float32) - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    mx = float(np.max(x))
    if mx > 0:
        return (x.astype(np.float32) / mx * 255.0).clip(0, 255).astype(np.uint8)
    return np.zeros(x.shape, dtype=np.uint8)


def _mux_to_rgb(mux: np.ndarray) -> np.ndarray:
    """
    mux: (H,W,3) numeric
    returns uint8 RGB (H,W,3)
    """
    if mux.ndim != 3 or mux.shape[-1] != 3:
        raise ValueError("mux must have shape (H,W,3)")
    out = np.zeros_like(mux, dtype=np.uint8)
    for k in range(3):
        out[..., k] = _stretch_to_uint8(mux[..., k])
    return out


def _draw_dots(*, colors: np.ndarray, lx: np.ndarray, ly: np.ndarray, size: int, radius: int) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    r = int(radius)
    for k in range(len(lx)):
        x = int(np.round(lx[k]))
        y = int(np.round(ly[k]))
        x0 = max(0, x - r)
        x1 = min(size, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(size, y + r + 1)
        out[y0:y1, x0:x1, :] = colors[k].reshape(1, 1, 4)
    return out


def generate_processed(
    *,
    he_path: str,
    mux_path: str,
    out_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    mux_rgb_channels: Sequence[int] = (0, 10, 35),
    tissue_min: float = 0.1,
    cd31_channel: int | None = None,
    proxy_lambda: float = 50.0,
    features_csv: str | None = None,
    dot_radius: int = 2,
) -> None:
    """
    Minimal patch generator for tests + future pipeline.

    Writes:
    - out_dir/index.json
    - out_dir/he/{i}_{j}.png
    - out_dir/mux_rgb/{i}_{j}.png
    """
    out = Path(out_dir)
    (out / "he").mkdir(parents=True, exist_ok=True)
    (out / "mux_rgb").mkdir(parents=True, exist_ok=True)
    if cd31_channel is not None:
        (out / "vasculature").mkdir(parents=True, exist_ok=True)
        (out / "oxygen").mkdir(parents=True, exist_ok=True)
        (out / "glucose").mkdir(parents=True, exist_ok=True)
    if features_csv is not None:
        (out / "cell_state").mkdir(parents=True, exist_ok=True)

    he_info = get_ome_info(he_path)
    if he_info.width <= 0 or he_info.height <= 0:
        raise ValueError("Invalid H&E image dimensions")

    img_w, img_h = he_info.width, he_info.height
    patches = []

    df = None
    thresholds = None
    states = None
    if features_csv is not None:
        df = pd.read_csv(features_csv)
        thresholds = compute_marker_thresholds(df, markers=["Ki67", "PCNA", "Vimentin", "Ecadherin"], pct=95)
        states = assign_cell_state(df, thresholds)
        df = df.copy()
        df["_state"] = states

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")

    n_cols = max(0, (img_w - patch_size) // stride + 1) if img_w >= patch_size else 0
    n_rows = max(0, (img_h - patch_size) // stride + 1) if img_h >= patch_size else 0

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * stride
            x0 = j * stride

            he_patch = read_region_rgb(he_path, y0=y0, x0=x0, h=patch_size, w=patch_size)
            if tissue_fraction_rgb(he_patch) < float(tissue_min):
                continue

            patch_id = f"{i}_{j}"
            Image.fromarray(he_patch).save(out / "he" / f"{patch_id}.png")

            mux_patch = read_region_channels(
                mux_path,
                y0=y0,
                x0=x0,
                h=patch_size,
                w=patch_size,
                channels=list(mux_rgb_channels),
            )
            mux_rgb = _mux_to_rgb(mux_patch)
            Image.fromarray(mux_rgb).save(out / "mux_rgb" / f"{patch_id}.png")

            if cd31_channel is not None:
                cd31 = read_region_channels(
                    mux_path,
                    y0=y0,
                    x0=x0,
                    h=patch_size,
                    w=patch_size,
                    channels=[int(cd31_channel)],
                )[..., 0]
                cd31_u8 = _stretch_to_uint8(cd31)
                vasc_rgba = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)
                vasc_rgba[..., 0] = 255
                vasc_rgba[..., 3] = cd31_u8
                Image.fromarray(vasc_rgba, mode="RGBA").save(out / "vasculature" / f"{patch_id}.png")

                # Vessel binary + distance-to-vessel proxy
                thr = float(np.percentile(cd31, 95))
                if thr <= 0:
                    vessel = cd31 > 0
                else:
                    vessel = cd31 >= thr
                dist = distance_transform_edt(~vessel).astype(np.float32)
                lam = float(proxy_lambda) if float(proxy_lambda) > 0 else 1.0
                proxy = np.exp(-dist / lam)
                proxy_u8 = (proxy * 255.0).clip(0, 255).astype(np.uint8)
                oxy_rgb = np.repeat(proxy_u8[..., None], 3, axis=-1)
                glu_rgb = np.repeat(proxy_u8[..., None], 3, axis=-1)
                Image.fromarray(oxy_rgb, mode="RGB").save(out / "oxygen" / f"{patch_id}.png")
                Image.fromarray(glu_rgb, mode="RGB").save(out / "glucose" / f"{patch_id}.png")

            if df is not None:
                if "Xt" not in df.columns or "Yt" not in df.columns:
                    raise ValueError("features_csv must contain Xt and Yt columns")
                in_patch = (
                    (df["Xt"] >= x0)
                    & (df["Xt"] < x0 + patch_size)
                    & (df["Yt"] >= y0)
                    & (df["Yt"] < y0 + patch_size)
                )
                sub = df.loc[in_patch]
                if len(sub) > 0:
                    lx = (sub["Xt"].values - x0).astype(np.float32)
                    ly = (sub["Yt"].values - y0).astype(np.float32)
                    colors = state_to_rgba(sub["_state"])
                    layer = _draw_dots(colors=colors, lx=lx, ly=ly, size=patch_size, radius=dot_radius)
                else:
                    layer = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)
                Image.fromarray(layer, mode="RGBA").save(out / "cell_state" / f"{patch_id}.png")

            patches.append(
                {"i": i, "j": j, "x0": x0, "y0": y0, "x1": x0 + patch_size, "y1": y0 + patch_size}
            )

    index = {
        "patches": patches,
        "patch_size": patch_size,
        "stride": stride,
        "tissue_min": tissue_min,
        "img_w": img_w,
        "img_h": img_h,
        "mux_rgb_channels": list(mux_rgb_channels),
        "cd31_channel": cd31_channel,
        "proxy_lambda": proxy_lambda,
        "features_csv": features_csv,
        "dot_radius": dot_radius,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate per-patch H&E + multiplex RGB composites.")
    parser.add_argument("--he", required=True, help="Path to H&E OME-TIFF (e.g. CRC02-HE.ome.tif)")
    parser.add_argument("--mux", required=True, help="Path to multiplex OME-TIFF (e.g. CRC02.ome.tif)")
    parser.add_argument("--out", default="processed", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tissue-min", type=float, default=0.1)
    parser.add_argument("--mux-rgb-channels", nargs=3, type=int, default=[0, 10, 35])
    args = parser.parse_args()

    generate_processed(
        he_path=args.he,
        mux_path=args.mux,
        out_dir=args.out,
        patch_size=args.patch_size,
        stride=args.stride,
        tissue_min=args.tissue_min,
        mux_rgb_channels=tuple(args.mux_rgb_channels),
    )

