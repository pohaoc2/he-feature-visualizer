# patchify.py Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `patchify.py` to use whole-slide overview-mask tissue detection, mpp-based H&E↔multiplex alignment, and a side-by-side patch visualization.

**Architecture:** Download a ~1/64 resolution H&E overview via zarr stride sampling, run HSV+Otsu to build a whole-slide tissue mask, filter patch candidates from that mask, then read paired H&E/multiplex patches with mpp-derived coordinate scaling. A side-by-side JPEG visualizes patch coverage on both modalities.

**Tech Stack:** `tifffile`, `zarr`, `cv2`, `numpy`, `PIL`, `pytest`

---

## Reference

Design doc: `docs/plans/2026-03-05-patchify-rewrite-design.md`

Key constants (CRC02):

- H&E: 82,230 × 54,363 px, mpp=0.325 µm/px, axes=CYX, dtype=uint8
- Multiplex: 41,094 × 24,017 px, mpp=0.65 µm/px, axes=CYX, dtype=uint16, 36 channels
- Scale factor: `he_mpp / mx_mpp = 0.5`

Functions to **keep unchanged**: `tissue_mask_hsv`, `read_multiplex_patch`, `load_channel_indices`, `get_ome_mpp`, `_open_zarr_store`, `_clip_and_read`, `_get_image_dims`

Functions to **add**: `build_tissue_mask`, `get_tissue_patches`, `visualize`

Functions to **rewrite**: `main` (replace per-patch loop with overview-mask approach)

Test file: `tests/test_patchify.py`

---

## Setup

Verify test runner works:

```bash
cd /Users/pohaochiu/Documents/UW/bagherilab/he-feature-visualizer
source .venv/bin/activate
pip install pytest
pytest tests/ -v
```

---

## Task 1: `build_tissue_mask`

Read a downsampled overview via zarr stride sampling and return a boolean tissue mask.

**Files:**

- Modify: `patchify.py` (add function after `tissue_fraction`)
- Test: `tests/test_patchify.py`

**Step 1: Write the failing test**

```python
# tests/test_patchify.py
import numpy as np
import pytest
import tifffile
import zarr

from patchify import build_tissue_mask, _open_zarr_store, _get_image_dims


def _make_cyx_tif(tmp_path, arr, name="test.ome.tif"):
    """Write a CYX OME-TIFF and return (store, axes, img_w, img_h)."""
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, ome=True, metadata={"axes": "CYX"})
    tif = tifffile.TiffFile(str(p))
    store = _open_zarr_store(tif)
    img_w, img_h, axes = _get_image_dims(tif)
    return store, axes, img_w, img_h


def test_build_tissue_mask_shape(tmp_path):
    """Mask shape should be (img_h//downsample, img_w//downsample)."""
    arr = np.zeros((3, 128, 192), dtype=np.uint8)
    store, axes, img_w, img_h = _make_cyx_tif(tmp_path, arr)
    mask = build_tissue_mask(store, axes, img_w, img_h, downsample=16)
    assert mask.shape == (128 // 16, 192 // 16)  # (8, 12)
    assert mask.dtype == bool


def test_build_tissue_mask_detects_tissue(tmp_path):
    """High-saturation pixels (pink tissue) should be detected as tissue."""
    arr = np.zeros((3, 128, 128), dtype=np.uint8)
    # Paint a tissue-like region: high R, low G, medium B -> high HSV saturation
    arr[0, 40:80, 40:80] = 220  # R
    arr[1, 40:80, 40:80] = 60   # G
    arr[2, 40:80, 40:80] = 100  # B
    store, axes, img_w, img_h = _make_cyx_tif(tmp_path, arr)
    mask = build_tissue_mask(store, axes, img_w, img_h, downsample=8)
    # Tissue region is rows 5-9, cols 5-9 in mask (40//8=5, 80//8=10)
    assert mask[5:10, 5:10].any(), "Expected tissue in tissue region"
    assert not mask[:3, :3].any(), "Expected no tissue in blank corner"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_patchify.py::test_build_tissue_mask_shape tests/test_patchify.py::test_build_tissue_mask_detects_tissue -v
```

Expected: `FAILED` — `ImportError: cannot import name 'build_tissue_mask'`

**Step 3: Implement `build_tissue_mask`**

Add after `tissue_fraction` in `patchify.py`:

```python
def build_tissue_mask(
    store, axes: str, img_w: int, img_h: int, downsample: int = 64
) -> np.ndarray:
    """Build a boolean tissue mask from a downsampled H&E overview.

    Downloads only ~(img_h/downsample * img_w/downsample * 3) bytes.

    Parameters
    ----------
    store:      zarr Array opened from tifffile series.
    axes:       Axes string (e.g. 'CYX' or 'YXC').
    img_w/h:    Full-resolution image dimensions.
    downsample: Stride for overview sampling (default 64).

    Returns
    -------
    bool ndarray of shape (img_h // downsample, img_w // downsample).
    """
    axes = axes.upper()
    c_first = "C" in axes and axes.index("C") < axes.index("Y")

    if c_first:
        raw = np.array(store[:, ::downsample, ::downsample])  # (C, H//ds, W//ds)
        overview = np.moveaxis(raw, 0, -1)                    # (H//ds, W//ds, C)
    else:
        overview = np.array(store[::downsample, ::downsample, :])

    if overview.shape[-1] > 3:
        overview = overview[..., :3]
    if overview.dtype != np.uint8:
        p1 = float(np.percentile(overview, 1))
        p99 = float(np.percentile(overview, 99))
        if p99 > p1:
            overview = ((overview.astype(np.float32) - p1) / (p99 - p1) * 255).clip(0, 255).astype(np.uint8)
        else:
            overview = np.zeros_like(overview, dtype=np.uint8)

    return tissue_mask_hsv(overview)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_patchify.py::test_build_tissue_mask_shape tests/test_patchify.py::test_build_tissue_mask_detects_tissue -v
```

Expected: `PASSED` for both.

**Step 5: Commit**

```bash
git add patchify.py tests/test_patchify.py
git commit -m "feat: add build_tissue_mask for overview-based tissue detection"
```

---

## Task 2: `get_tissue_patches`

Filter a patch grid by tissue mask fraction, returning level-0 `(x0, y0)` coordinates.

**Files:**

- Modify: `patchify.py` (add function after `build_tissue_mask`)
- Test: `tests/test_patchify.py`

**Step 1: Write the failing tests**

```python
from patchify import get_tissue_patches


def test_get_tissue_patches_keeps_tissue():
    """Patch overlapping tissue region should be kept."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True  # tissue at mask rows 3-6, cols 3-6
    patches = get_tissue_patches(
        mask, img_w=640, img_h=640,
        patch_size=64, stride=64, tissue_min=0.1, downsample=64,
    )
    assert (192, 192) in patches


def test_get_tissue_patches_discards_background():
    """Patch in blank region should be discarded."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    patches = get_tissue_patches(
        mask, img_w=640, img_h=640,
        patch_size=64, stride=64, tissue_min=0.1, downsample=64,
    )
    assert (0, 0) not in patches


def test_get_tissue_patches_only_within_bounds():
    """All returned patches must fit fully within the image."""
    mask = np.ones((10, 10), dtype=bool)
    patches = get_tissue_patches(
        mask, img_w=600, img_h=600,
        patch_size=64, stride=64, tissue_min=0.0, downsample=64,
    )
    for x0, y0 in patches:
        assert x0 + 64 <= 600
        assert y0 + 64 <= 600
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_patchify.py::test_get_tissue_patches_keeps_tissue tests/test_patchify.py::test_get_tissue_patches_discards_background tests/test_patchify.py::test_get_tissue_patches_only_within_bounds -v
```

Expected: `FAILED` — `ImportError`

**Step 3: Implement `get_tissue_patches`**

Add after `build_tissue_mask` in `patchify.py`:

```python
def get_tissue_patches(
    mask: np.ndarray,
    img_w: int, img_h: int,
    patch_size: int, stride: int,
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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_patchify.py::test_get_tissue_patches_keeps_tissue tests/test_patchify.py::test_get_tissue_patches_discards_background tests/test_patchify.py::test_get_tissue_patches_only_within_bounds -v
```

Expected: `PASSED` for all three.

**Step 5: Commit**

```bash
git add patchify.py tests/test_patchify.py
git commit -m "feat: add get_tissue_patches for mask-based patch filtering"
```

---

## Task 3: `visualize`

Produce `vis_patches.jpg`: H&E thumbnail (left) and multiplex composite (right) with patch grid overlaid.

**Files:**

- Modify: `patchify.py` (add before `main`)
- Test: `tests/test_patchify.py`

**Step 1: Write the failing tests**

```python
from pathlib import Path
from patchify import visualize


def test_visualize_creates_file(tmp_path):
    """visualize() must save vis_patches.jpg to out_dir."""
    he_overview = np.zeros((20, 30, 3), dtype=np.uint8)
    mx_overview = np.zeros((36, 10, 20), dtype=np.uint16)
    coords    = [(4, 5), (8, 5)]
    mx_coords = [(2, 2), (4, 2)]

    visualize(
        he_overview=he_overview,
        mx_overview=mx_overview,
        he_coords=coords,
        mx_coords=mx_coords,
        vis_channels=[0, 10, 20],
        out_dir=tmp_path,
    )
    assert (tmp_path / "vis_patches.jpg").exists()


def test_visualize_output_dimensions(tmp_path):
    """Output width = he_width + gap + mx_width."""
    he_overview = np.zeros((20, 30, 3), dtype=np.uint8)
    mx_overview = np.zeros((36, 10, 20), dtype=np.uint16)

    visualize(
        he_overview=he_overview,
        mx_overview=mx_overview,
        he_coords=[],
        mx_coords=[],
        vis_channels=[0, 10, 20],
        out_dir=tmp_path,
        gap=10,
    )
    from PIL import Image
    img = Image.open(tmp_path / "vis_patches.jpg")
    assert img.width == 30 + 10 + 20   # he_w + gap + mx_w
    assert img.height == max(20, 10)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_patchify.py::test_visualize_creates_file tests/test_patchify.py::test_visualize_output_dimensions -v
```

Expected: `FAILED` — `ImportError`

**Step 3: Implement `visualize`**

Add before `main` in `patchify.py`:

```python
def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale a 2D array to uint8 using p1/p99 percentile normalization."""
    p1  = float(np.percentile(arr, 1))
    p99 = float(np.percentile(arr, 99))
    if p99 > p1:
        return ((arr.astype(np.float32) - p1) / (p99 - p1) * 255).clip(0, 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def visualize(
    he_overview: np.ndarray,
    mx_overview: np.ndarray,
    he_coords: list[tuple[int, int]],
    mx_coords: list[tuple[int, int]],
    vis_channels: list[int],
    out_dir: Path,
    patch_size_ov: int = 4,
    gap: int = 10,
) -> None:
    """Save side-by-side vis_patches.jpg to out_dir.

    Parameters
    ----------
    he_overview:    (H_ov, W_ov, 3) uint8 H&E overview.
    mx_overview:    (C, H_mx_ov, W_mx_ov) uint16 multiplex overview.
    he_coords:      (x0, y0) in overview pixel coords for H&E panel.
    mx_coords:      Corresponding coords in multiplex overview pixels.
    vis_channels:   3 channel indices for multiplex RGB composite.
    patch_size_ov:  Box size in overview pixels (default 4).
    gap:            White gap between panels in pixels (default 10).
    """
    import PIL.ImageDraw

    # H&E panel
    he_panel = Image.fromarray(he_overview.astype(np.uint8), "RGB").copy()
    draw_he  = PIL.ImageDraw.Draw(he_panel)
    for x0, y0 in he_coords:
        draw_he.rectangle([x0, y0, x0 + patch_size_ov - 1, y0 + patch_size_ov - 1],
                          outline=(0, 200, 0), width=1)

    # Multiplex composite panel
    c0 = _norm_to_uint8(mx_overview[vis_channels[0]])
    c1 = _norm_to_uint8(mx_overview[vis_channels[1]])
    c2 = _norm_to_uint8(mx_overview[vis_channels[2]])
    mx_panel = Image.fromarray(np.stack([c0, c1, c2], axis=-1), "RGB").copy()
    draw_mx  = PIL.ImageDraw.Draw(mx_panel)
    for x0, y0 in mx_coords:
        draw_mx.rectangle([x0, y0, x0 + patch_size_ov - 1, y0 + patch_size_ov - 1],
                          outline=(0, 200, 0), width=1)

    # Combine
    h   = max(he_panel.height, mx_panel.height)
    w   = he_panel.width + gap + mx_panel.width
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(he_panel, (0, 0))
    out.paste(mx_panel, (he_panel.width + gap, 0))
    out.save(str(out_dir / "vis_patches.jpg"), quality=90)
    print(f"Saved vis_patches.jpg ({w}x{h} px, {len(he_coords)} patches shown)")
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_patchify.py::test_visualize_creates_file tests/test_patchify.py::test_visualize_output_dimensions -v
```

Expected: `PASSED` for both.

**Step 5: Commit**

```bash
git add patchify.py tests/test_patchify.py
git commit -m "feat: add visualize for side-by-side H&E + multiplex patch overview"
```

---

## Task 4: Rewrite `main`

Wire `build_tissue_mask` → `get_tissue_patches` → extract patches → `visualize`. Add `--overview-downsample` and `--vis-channels` CLI args.

**Files:**

- Modify: `patchify.py` — replace `main()` body

**Step 1: Replace `main()`**

Replace the existing `main()` function entirely:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 -- Extract H&E and multiplex patches from OME-TIFFs."
    )
    parser.add_argument("--he-image",            required=True)
    parser.add_argument("--multiplex-image",      required=True)
    parser.add_argument("--metadata-csv",         required=True)
    parser.add_argument("--out",                  default="processed")
    parser.add_argument("--patch-size",           type=int,   default=256)
    parser.add_argument("--stride",               type=int,   default=256)
    parser.add_argument("--tissue-min",           type=float, default=0.1)
    parser.add_argument("--channels",             nargs="+",  default=["CD31", "Ki67", "CD45", "PCNA"])
    parser.add_argument("--overview-downsample",  type=int,   default=64,
                        help="Stride for H&E overview sampling (default 64)")
    parser.add_argument("--vis-channels",         type=int,   nargs=3, default=[0, 10, 20],
                        help="3 multiplex channel indices for RGB composite in vis")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "he").mkdir(parents=True, exist_ok=True)
    (out_dir / "multiplex").mkdir(parents=True, exist_ok=True)

    ds         = args.overview_downsample
    patch_size = args.patch_size

    print("Resolving channel indices ...")
    channel_indices, resolved_names = load_channel_indices(args.metadata_csv, args.channels)

    print("Opening H&E image ...")
    he_tif   = tifffile.TiffFile(args.he_image)
    he_w, he_h, he_axes = _get_image_dims(he_tif)
    he_store = _open_zarr_store(he_tif)
    he_mpp_x, _ = get_ome_mpp(he_tif)
    print(f"  {he_w} x {he_h}  mpp={he_mpp_x}")

    print("Opening multiplex image ...")
    mx_tif   = tifffile.TiffFile(args.multiplex_image)
    mx_w, mx_h, mx_axes = _get_image_dims(mx_tif)
    mx_store = _open_zarr_store(mx_tif)
    mx_mpp_x, _ = get_ome_mpp(mx_tif)
    print(f"  {mx_w} x {mx_h}  mpp={mx_mpp_x}")

    scale = (he_mpp_x / mx_mpp_x) if (he_mpp_x and mx_mpp_x) else (he_w / mx_w)
    print(f"  scale H&E -> multiplex: {scale:.4f}")

    print(f"Building tissue mask (downsample={ds}) ...")
    he_axes_up = he_axes.upper()
    c_first    = "C" in he_axes_up and he_axes_up.index("C") < he_axes_up.index("Y")
    if c_first:
        he_overview = np.moveaxis(np.array(he_store[:, ::ds, ::ds]), 0, -1).astype(np.uint8)
    else:
        he_overview = np.array(he_store[::ds, ::ds, :]).astype(np.uint8)
    if he_overview.shape[-1] > 3:
        he_overview = he_overview[..., :3]

    mask = tissue_mask_hsv(he_overview)
    print(f"  Tissue fraction: {mask.mean():.2%}")

    print("Selecting tissue patches ...")
    coords = get_tissue_patches(mask, he_w, he_h, patch_size, args.stride, args.tissue_min, ds)
    print(f"  {len(coords)} patches selected")

    print("Extracting patches ...")
    index: list[dict] = []
    he_vis_coords: list[tuple[int, int]] = []
    mx_vis_coords: list[tuple[int, int]] = []

    for idx, (x0, y0) in enumerate(coords):
        if idx % 500 == 0:
            print(f"  {idx}/{len(coords)} ...")

        he_patch = read_he_patch(he_store, he_axes, he_w, he_h, y0, x0, patch_size)
        Image.fromarray(he_patch).save(out_dir / "he" / f"{x0}_{y0}.png")

        x0_mx   = round(x0 * scale)
        y0_mx   = round(y0 * scale)
        size_mx = max(1, round(patch_size * scale))
        has_mx  = (x0_mx + size_mx <= mx_w) and (y0_mx + size_mx <= mx_h)

        if has_mx:
            mx_patch = read_multiplex_patch(
                mx_store, mx_axes, mx_w, mx_h,
                y0_mx, x0_mx, size_mx, size_mx, channel_indices,
            )
            if mx_patch.shape[1] != patch_size or mx_patch.shape[2] != patch_size:
                resized = np.zeros((mx_patch.shape[0], patch_size, patch_size), dtype=mx_patch.dtype)
                for c in range(mx_patch.shape[0]):
                    resized[c] = cv2.resize(mx_patch[c], (patch_size, patch_size),
                                            interpolation=cv2.INTER_LINEAR)
                mx_patch = resized
            np.save(out_dir / "multiplex" / f"{x0}_{y0}.npy", mx_patch)
            mx_vis_coords.append((x0_mx // ds, y0_mx // ds))

        he_vis_coords.append((x0 // ds, y0 // ds))
        index.append({"x0": x0, "y0": y0, "has_multiplex": has_mx})

    n_mx = sum(p["has_multiplex"] for p in index)
    print(f"  Done. {n_mx}/{len(index)} patches have multiplex.")

    print("Generating vis_patches.jpg ...")
    mx_axes_up = mx_axes.upper()
    mx_c_first = "C" in mx_axes_up and mx_axes_up.index("C") < mx_axes_up.index("Y")
    if mx_c_first:
        mx_overview = np.array(mx_store[:, ::ds, ::ds])
    else:
        mx_overview = np.moveaxis(np.array(mx_store[::ds, ::ds, :]), -1, 0)
    vis_ch = [min(c, mx_overview.shape[0] - 1) for c in args.vis_channels]
    visualize(he_overview, mx_overview, he_vis_coords, mx_vis_coords, vis_ch, out_dir)

    with open(out_dir / "index.json", "w") as f:
        json.dump({
            "patches": index,
            "patch_size": patch_size,
            "stride": args.stride,
            "tissue_min": args.tissue_min,
            "img_w": he_w, "img_h": he_h,
            "he_mpp": he_mpp_x, "mx_mpp": mx_mpp_x,
            "scale_he_to_mx": scale,
            "channels": resolved_names,
        }, f, indent=2)

    print(f"Index written to {out_dir / 'index.json'}")
    print("Stage 1 complete.")
```

**Step 2: Run full test suite**

```bash
pytest tests/test_patchify.py -v
```

Expected: all tests pass.

**Step 3: Smoke test with synthetic data**

Create `tests/smoke_patchify.py`:

```python
"""Smoke test: run main() on tiny synthetic OME-TIFFs and verify outputs."""
import json, sys, shutil, tempfile
import numpy as np
import tifffile
from pathlib import Path


def main():
    tmp = Path(tempfile.mkdtemp())
    he_path   = tmp / "he.ome.tif"
    mx_path   = tmp / "mx.ome.tif"
    meta_path = tmp / "meta.csv"
    out_dir   = tmp / "out"

    # 3-channel H&E, CYX uint8, 256x256
    he = np.zeros((3, 256, 256), dtype=np.uint8)
    he[:, 64:192, 64:192] = [200, 60, 100]   # tissue region
    tifffile.imwrite(str(he_path), he, ome=True, metadata={"axes": "CYX"})

    # 4-channel multiplex, CYX uint16, 128x128
    mx = np.zeros((4, 128, 128), dtype=np.uint16)
    mx[0, 32:96, 32:96] = 1000
    tifffile.imwrite(str(mx_path), mx, ome=True, metadata={"axes": "CYX"})

    meta_path.write_text("Channel ID,Target Name\nChannel:0:0,CD31\n")

    sys.argv = [
        "patchify.py",
        "--he-image",        str(he_path),
        "--multiplex-image", str(mx_path),
        "--metadata-csv",    str(meta_path),
        "--out",             str(out_dir),
        "--patch-size",      "64",
        "--stride",          "64",
        "--tissue-min",      "0.05",
        "--channels",        "CD31",
        "--overview-downsample", "8",
        "--vis-channels",    "0", "0", "0",
    ]
    from patchify import main as run
    run()

    assert (out_dir / "vis_patches.jpg").exists(), "vis_patches.jpg missing"
    assert (out_dir / "index.json").exists(),       "index.json missing"
    idx = json.loads((out_dir / "index.json").read_text())
    assert len(idx["patches"]) > 0, "No patches extracted"
    print(f"OK — {len(idx['patches'])} patches extracted, vis saved.")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
```

Run:

```bash
python tests/smoke_patchify.py
```

Expected: `OK — N patches extracted, vis saved.`

**Step 4: Commit**

```bash
git add patchify.py tests/smoke_patchify.py
git commit -m "feat: rewrite main() with overview-mask tissue detection and vis"
```

---

## Final check

```bash
pytest tests/ -v
```

All tests pass. Open `vis_patches.jpg` from a real run to visually confirm H&E and multiplex panels align.