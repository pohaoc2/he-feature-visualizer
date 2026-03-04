"""
Tests for the redesigned patchify.py — Stage 1 of the histopathology pipeline.

Stage 1 contract:
  - CLAM-style HSV tissue detection (no CSV cell drawing)
  - Extract 256×256 patches from two OME-TIFFs: H&E and multiplex
  - Save H&E patches as PNG (uint8 RGB), multiplex patches as .npy (C, H, W) uint16
  - Save index.json listing kept patches with schema metadata

All tests use synthetic OME-TIFFs created with tifffile — no real data required.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ome_tiff(path, arr: np.ndarray, axes: str) -> None:
    """Write a synthetic OME-TIFF to *path* with the given axes string."""
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": axes})


def _patchify_script() -> str:
    """Absolute path to patchify.py."""
    return str(Path(__file__).resolve().parent.parent / "patchify.py")


def _minimal_metadata_csv(path: Path, targets: dict) -> None:
    """Write a minimal metadata CSV to *path*.

    Args:
        path: destination file path.
        targets: mapping of {Target Name: Channel ID (0-indexed)}.

    The CSV has at minimum the two columns required by patchify.py:
        Channel ID, Target Name
    """
    lines = ["Channel ID,Target Name"]
    for target_name, channel_id in targets.items():
        lines.append(f"{channel_id},{target_name}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Unit tests — module-level functions
# ---------------------------------------------------------------------------

def test_tissue_mask_hsv_detects_tissue():
    """
    Contract: tissue_mask_hsv returns a bool (H, W) mask.

    - Shape must equal the input image's (H, W).
    - dtype must be bool.
    - Pixels inside a chromatic (pink, tissue-like) block are classified True.
    - Achromatic white background pixels are classified False.

    Implementation note: CLAM-style detection thresholds on the HSV saturation
    channel, so high-saturation pink tissue is reliably separated from the
    zero-saturation white background regardless of brightness.
    """
    from patchify import tissue_mask_hsv  # noqa: WPS433

    h, w = 64, 64
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # white background
    # Place a 20×20 pink (tissue-like) block at the image center
    rgb[22:42, 22:42, :] = np.array([180, 60, 120], dtype=np.uint8)

    mask = tissue_mask_hsv(rgb)

    assert mask.shape == (h, w), "Mask shape must match input (H, W)"
    assert mask.dtype == bool, "Mask must be boolean"

    # Center of the pink block must be classified as tissue
    assert bool(mask[32, 32]) is True, "Center of pink block should be tissue"
    # Top-left white corner must be classified as background
    assert bool(mask[0, 0]) is False, "White corner should be background"


def test_tissue_mask_hsv_rejects_gray_background():
    """
    Contract: tissue_mask_hsv rejects low-saturation (achromatic) pixels.

    A uniform gray image (R=G=B=200) has near-zero HSV saturation.  The
    CLAM-style approach thresholds on saturation, so gray must not be
    classified as tissue even though it is darker than white.

    Verifies mean(mask) < 0.1 across the entire uniform-gray image, confirming
    that fewer than 10 % of pixels are falsely labelled as tissue.
    """
    from patchify import tissue_mask_hsv  # noqa: WPS433

    h, w = 64, 64
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)  # uniform dark-ish gray

    mask = tissue_mask_hsv(rgb)

    assert mask.mean() < 0.1, (
        "Uniform gray image should be almost entirely classified as background"
    )


def test_tissue_fraction_range():
    """
    Contract: tissue_fraction returns the proportion of pixels classified as tissue.

    A 64×64 image whose left half is pink (tissue) and right half is white
    (background) should yield a fraction near 0.5.  We allow a generous range
    of (0.3, 0.7) to account for boundary effects at the colour boundary and
    Otsu threshold variance on small images.

    Verifies: 0.3 < tissue_fraction(rgb) < 0.7
    """
    from patchify import tissue_fraction  # noqa: WPS433

    h, w = 64, 64
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)  # white background
    rgb[:, : w // 2, :] = np.array([180, 60, 120], dtype=np.uint8)  # left half pink

    frac = tissue_fraction(rgb)

    assert 0.3 < frac < 0.7, (
        f"Expected fraction ~0.5 for half-tissue image, got {frac:.3f}"
    )


def test_get_patch_grid_no_overlap():
    """
    Contract: get_patch_grid returns the correct (row, col) grid for a perfectly
    divisible image with stride == patch_size (non-overlapping tiling).

    For a 512×512 image with patch_size=256 and stride=256 there are exactly
    four non-overlapping patches arranged in a 2×2 grid:
        (i=0, j=0), (i=0, j=1), (i=1, j=0), (i=1, j=1)

    Verifies the returned set equals exactly {(0,0),(0,1),(1,0),(1,1)} and that
    no duplicates are present.
    """
    from patchify import get_patch_grid  # noqa: WPS433

    patches = get_patch_grid(img_w=512, img_h=512, patch_size=256, stride=256)

    assert set(patches) == {(0, 0), (0, 1), (1, 0), (1, 1)}, (
        f"Expected 4 patches in 2×2 grid, got {patches}"
    )
    assert len(patches) == 4, "No duplicate patches expected"


def test_get_patch_grid_excludes_edge_patches():
    """
    Contract: get_patch_grid must NOT include patches that extend beyond image bounds.

    For a 600×600 image with patch_size=256 and stride=256:
    - Patches at columns/rows 0 start at pixel 0, end at 256 — within bounds.
    - Patches at columns/rows 1 start at pixel 256, end at 512 — within bounds.
    - A patch at column/row 2 would start at 512 and end at 768 > 600 — excluded.

    Verifies that every returned patch satisfies:
        x1 = x0 + patch_size <= img_w  and  y1 = y0 + patch_size <= img_h
    """
    from patchify import get_patch_grid  # noqa: WPS433

    patch_size = 256
    img_w = img_h = 600
    patches = get_patch_grid(img_w=img_w, img_h=img_h, patch_size=patch_size, stride=patch_size)

    for i, j in patches:
        x0 = j * patch_size
        y0 = i * patch_size
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        assert x1 <= img_w, f"Patch ({i},{j}) x1={x1} exceeds img_w={img_w}"
        assert y1 <= img_h, f"Patch ({i},{j}) y1={y1} exceeds img_h={img_h}"


def test_read_he_patch_cyx_axes(tmp_path):
    """
    Contract: read_he_patch reads a 256×256 region from a CYX OME-TIFF and
    returns a uint8 RGB array of shape (256, 256, 3).

    A synthetic 512×512 CYX OME-TIFF is created with distinct constant values
    per channel so that channel-to-RGB ordering can be verified implicitly by
    shape and dtype — exact colour correctness is tested elsewhere.

    Verifies:
    - output.shape == (256, 256, 3)
    - output.dtype == uint8
    """
    from patchify import read_he_patch  # noqa: WPS433

    img_h, img_w = 512, 512
    arr = np.zeros((3, img_h, img_w), dtype=np.uint8)
    arr[0] = 180  # R channel
    arr[1] = 60   # G channel
    arr[2] = 120  # B channel

    he_path = tmp_path / "he.ome.tif"
    _write_ome_tiff(he_path, arr, axes="CYX")

    with tifffile.TiffFile(str(he_path)) as tif:
        zarr_store = tif.aszarr()

    patch = read_he_patch(
        zarr_store=zarr_store,
        axes="CYX",
        img_w=img_w,
        img_h=img_h,
        y0=0,
        x0=0,
        size=256,
    )

    assert patch.shape == (256, 256, 3), (
        f"Expected (256, 256, 3), got {patch.shape}"
    )
    assert patch.dtype == np.uint8, f"Expected uint8, got {patch.dtype}"


def test_read_multiplex_patch_selects_channels(tmp_path):
    """
    Contract: read_multiplex_patch returns (C, H, W) uint16 where
    C == len(channel_indices), and each output slice contains data from the
    correct source channel.

    A 10-channel CYX OME-TIFF is created where channel k is filled with the
    constant value k * 100.  Requesting channel_indices=[2, 5, 9] must yield:
    - output[0].mean() ≈ 200   (channel 2 → 2 * 100)
    - output[1].mean() ≈ 500   (channel 5 → 5 * 100)
    - output[2].mean() ≈ 900   (channel 9 → 9 * 100)

    Verifies shape (3, 256, 256), dtype uint16, and per-channel mean values.
    """
    from patchify import read_multiplex_patch  # noqa: WPS433

    img_h, img_w = 512, 512
    arr = np.zeros((10, img_h, img_w), dtype=np.uint16)
    for k in range(10):
        arr[k] = k * 100

    mux_path = tmp_path / "mux.ome.tif"
    _write_ome_tiff(mux_path, arr, axes="CYX")

    with tifffile.TiffFile(str(mux_path)) as tif:
        zarr_store = tif.aszarr()

    patch = read_multiplex_patch(
        zarr_store=zarr_store,
        axes="CYX",
        img_w=img_w,
        img_h=img_h,
        y0=0,
        x0=0,
        size_y=256,
        size_x=256,
        channel_indices=[2, 5, 9],
    )

    assert patch.shape == (3, 256, 256), (
        f"Expected (3, 256, 256), got {patch.shape}"
    )
    assert patch.dtype == np.uint16, f"Expected uint16, got {patch.dtype}"
    assert abs(patch[0].mean() - 200) < 1, (
        f"Channel 2 mean should be ~200, got {patch[0].mean()}"
    )
    assert abs(patch[1].mean() - 500) < 1, (
        f"Channel 5 mean should be ~500, got {patch[1].mean()}"
    )
    assert abs(patch[2].mean() - 900) < 1, (
        f"Channel 9 mean should be ~900, got {patch[2].mean()}"
    )


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_cli_creates_outputs(tmp_path):
    """
    Contract: the CLI produces he/{i}_{j}.png, multiplex/{i}_{j}.npy, and
    index.json under the --out directory.

    Setup:
    - 512×512 H&E OME-TIFF (CYX uint8): pink tissue fills top-left 256×256.
    - 512×512 multiplex OME-TIFF (CYX, 36 channels, uint16).
    - Metadata CSV mapping CD31 → channel 5, Ki67 → channel 12.
    - CLI run with --tissue-min 0.0 so all four patches are retained.

    Verifies:
    - processed/he/0_0.png exists, is 256×256, mode RGB.
    - processed/multiplex/0_0.npy exists, shape (2, 256, 256), dtype uint16.
    - processed/index.json exists and lists at least 1 patch.
    """
    h = w = 512

    # H&E: white background, pink top-left quadrant
    he_arr = np.full((3, h, w), 255, dtype=np.uint8)
    he_arr[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "he.ome.tif"
    _write_ome_tiff(he_path, he_arr, axes="CYX")

    # Multiplex: 36 channels, each filled with a distinct constant
    mux_arr = np.zeros((36, h, w), dtype=np.uint16)
    for k in range(36):
        mux_arr[k] = k * 10
    mux_path = tmp_path / "mux.ome.tif"
    _write_ome_tiff(mux_path, mux_arr, axes="CYX")

    # Metadata CSV: two channels requested
    meta_path = tmp_path / "metadata.csv"
    _minimal_metadata_csv(meta_path, {"CD31": 5, "Ki67": 12})

    out_dir = tmp_path / "processed"

    cmd = [
        sys.executable,
        _patchify_script(),
        "--he-image", str(he_path),
        "--multiplex-image", str(mux_path),
        "--metadata-csv", str(meta_path),
        "--out", str(out_dir),
        "--patch-size", "256",
        "--stride", "256",
        "--tissue-min", "0.0",
        "--channels", "CD31", "Ki67",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"patchify.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # H&E patch must exist and be a 256×256 RGB PNG
    he_patch_path = out_dir / "he" / "0_0.png"
    assert he_patch_path.exists(), "he/0_0.png must be created"
    from PIL import Image  # noqa: WPS433
    img = Image.open(str(he_patch_path))
    assert img.size == (256, 256), f"Expected 256×256 PNG, got {img.size}"
    assert img.mode == "RGB", f"Expected RGB mode, got {img.mode}"

    # Multiplex patch must have correct shape and dtype
    mux_patch_path = out_dir / "multiplex" / "0_0.npy"
    assert mux_patch_path.exists(), "multiplex/0_0.npy must be created"
    mux_patch = np.load(str(mux_patch_path))
    assert mux_patch.shape == (2, 256, 256), (
        f"Expected (2, 256, 256), got {mux_patch.shape}"
    )
    assert mux_patch.dtype == np.uint16, f"Expected uint16, got {mux_patch.dtype}"

    # index.json must list at least one patch
    index_path = out_dir / "index.json"
    assert index_path.exists(), "index.json must be created"
    data = json.loads(index_path.read_text())
    assert len(data["patches"]) >= 1, "index.json must list at least one patch"


def test_cli_tissue_filter_drops_background_patches(tmp_path):
    """
    Contract: patches whose tissue fraction falls below --tissue-min are
    excluded from the output and not written to disk.

    Setup:
    - 512×512 H&E OME-TIFF: only the top-left 256×256 quadrant is pink
      (tissue); the other three quadrants are achromatic white.
    - CLI run with --tissue-min 0.5.

    Because the white quadrants have near-zero saturation, their tissue
    fraction will be far below 0.5 and must be dropped.  The top-left patch
    contains uniform pink and must survive.

    Verifies:
    - index.json contains exactly 1 patch.
    - That patch has i=0, j=0.
    """
    h = w = 512

    he_arr = np.full((3, h, w), 255, dtype=np.uint8)  # white everywhere
    he_arr[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "he.ome.tif"
    _write_ome_tiff(he_path, he_arr, axes="CYX")

    mux_arr = np.zeros((36, h, w), dtype=np.uint16)
    mux_path = tmp_path / "mux.ome.tif"
    _write_ome_tiff(mux_path, mux_arr, axes="CYX")

    meta_path = tmp_path / "metadata.csv"
    _minimal_metadata_csv(meta_path, {"CD31": 0})

    out_dir = tmp_path / "processed"

    cmd = [
        sys.executable,
        _patchify_script(),
        "--he-image", str(he_path),
        "--multiplex-image", str(mux_path),
        "--metadata-csv", str(meta_path),
        "--out", str(out_dir),
        "--patch-size", "256",
        "--stride", "256",
        "--tissue-min", "0.5",
        "--channels", "CD31",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"patchify.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    data = json.loads((out_dir / "index.json").read_text())
    kept = [(p["i"], p["j"]) for p in data["patches"]]

    assert len(kept) == 1, (
        f"Expected exactly 1 patch to survive tissue filter, got {len(kept)}: {kept}"
    )
    assert kept[0] == (0, 0), (
        f"The surviving patch should be (i=0, j=0), got {kept[0]}"
    )


def test_index_json_schema(tmp_path):
    """
    Contract: index.json must carry a fixed schema of top-level metadata keys
    and per-patch coordinate keys.

    Required top-level keys:
        patches, stride, patch_size, tissue_min, img_w, img_h, channels

    Required per-patch keys (for every entry in patches):
        i, j, x0, y0, x1, y1

    A minimal synthetic 512×512 dataset is used (single channel, tissue-min 0.0
    so at least one patch is guaranteed to be written) to exercise the schema.
    """
    h = w = 512

    he_arr = np.full((3, h, w), 255, dtype=np.uint8)
    he_arr[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "he.ome.tif"
    _write_ome_tiff(he_path, he_arr, axes="CYX")

    mux_arr = np.zeros((4, h, w), dtype=np.uint16)
    mux_path = tmp_path / "mux.ome.tif"
    _write_ome_tiff(mux_path, mux_arr, axes="CYX")

    meta_path = tmp_path / "metadata.csv"
    _minimal_metadata_csv(meta_path, {"CD45": 2})

    out_dir = tmp_path / "processed"

    cmd = [
        sys.executable,
        _patchify_script(),
        "--he-image", str(he_path),
        "--multiplex-image", str(mux_path),
        "--metadata-csv", str(meta_path),
        "--out", str(out_dir),
        "--patch-size", "256",
        "--stride", "256",
        "--tissue-min", "0.0",
        "--channels", "CD45",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"patchify.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    index_path = out_dir / "index.json"
    assert index_path.exists(), "index.json must be created"
    data = json.loads(index_path.read_text())

    # Top-level schema
    required_top_level = {
        "patches", "stride", "patch_size", "tissue_min", "img_w", "img_h", "channels"
    }
    missing_top = required_top_level - set(data.keys())
    assert not missing_top, f"index.json missing top-level keys: {missing_top}"

    # Per-patch schema — need at least one patch to validate
    assert len(data["patches"]) >= 1, "Need at least one patch to validate per-patch schema"
    required_patch = {"i", "j", "x0", "y0", "x1", "y1"}
    for patch in data["patches"]:
        missing_patch = required_patch - set(patch.keys())
        assert not missing_patch, (
            f"Patch entry missing keys {missing_patch}: {patch}"
        )
