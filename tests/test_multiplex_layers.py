"""
Tests for multiplex_layers.py — Stage 4+5 of the histopathology pipeline.

Stage 4+5 contract:
  - Load per-patch multiplex .npy files (C, H, W) uint16 produced by Stage 1.
  - Extract specific channels (CD31, Ki67, PCNA) by name from channel metadata CSV.
  - Produce three RGBA overlay images per patch:
      vasculature: binary CD31 mask → red vessels on transparent background
      oxygen:      distance-transform proxy → RdYlBu colormap
      glucose:     max(Ki67, PCNA) metabolic demand → hot colormap

All tests use synthetic data created in tmp_path — no real data required.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multiplex_layers_script() -> str:
    """Absolute path to multiplex_layers.py."""
    return str(Path(__file__).resolve().parent.parent / "multiplex_layers.py")


def _write_metadata_csv(path: Path, targets: dict) -> None:
    """Write a minimal channel metadata CSV to *path*.

    Args:
        path: destination file path.
        targets: mapping of {Target Name: Channel ID string}, e.g.
                 {'CD31': 'Channel:0:0'}.

    The CSV format matches the one consumed by load_channel_names:
        Channel ID,Target Name
    """
    lines = ["Channel ID,Target Name"]
    for target_name, channel_id in targets.items():
        lines.append(f"{channel_id},{target_name}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Unit tests — load_multiplex_patch
# ---------------------------------------------------------------------------

def test_load_multiplex_patch(tmp_path):
    """
    Contract: load_multiplex_patch reads a .npy file and returns the array
    with the correct shape and dtype.

    A (3, 64, 64) uint16 array is saved to disk; the function must return an
    array with identical shape and dtype.

    Verifies:
    - output.shape == (3, 64, 64)
    - output.dtype == uint16
    """
    from multiplex_layers import load_multiplex_patch  # noqa: WPS433

    arr = np.zeros((3, 64, 64), dtype=np.uint16)
    arr[0] = 100
    arr[1] = 200
    arr[2] = 300

    npy_path = tmp_path / "0_0.npy"
    np.save(str(npy_path), arr)

    result = load_multiplex_patch(str(npy_path))

    assert result.shape == (3, 64, 64), (
        f"Expected shape (3, 64, 64), got {result.shape}"
    )
    assert result.dtype == np.uint16, f"Expected uint16, got {result.dtype}"


# ---------------------------------------------------------------------------
# Unit tests — get_channel_index
# ---------------------------------------------------------------------------

def test_get_channel_index_case_insensitive():
    """
    Contract: get_channel_index returns the 0-based index of the target name
    within the channel_names list using case-insensitive matching.

    A list of four channel names is used:
    - 'cd31' (lowercase) must match 'CD31' at index 1.
    - 'Ki67' (mixed case) must match 'Ki67' at index 2.
    - 'CD45' (exact case) must match 'CD45' at index 3.

    Verifies the returned index for each of the three lookups.
    """
    from multiplex_layers import get_channel_index  # noqa: WPS433

    names = ["DNA", "CD31", "Ki67", "CD45"]

    assert get_channel_index(names, "cd31") == 1, (
        "Case-insensitive lookup of 'cd31' should return index 1"
    )
    assert get_channel_index(names, "Ki67") == 2, (
        "Exact-case lookup of 'Ki67' should return index 2"
    )
    assert get_channel_index(names, "CD45") == 3, (
        "Exact-case lookup of 'CD45' should return index 3"
    )


def test_get_channel_index_raises_on_missing():
    """
    Contract: get_channel_index raises ValueError when the target is not
    present in the channel_names list.

    A list without 'PCNA' is used; requesting 'PCNA' must raise ValueError.

    Verifies pytest.raises(ValueError) is triggered.
    """
    from multiplex_layers import get_channel_index  # noqa: WPS433

    names = ["DNA", "CD31", "Ki67"]

    with pytest.raises(ValueError):
        get_channel_index(names, "PCNA")


# ---------------------------------------------------------------------------
# Unit tests — extract_channel
# ---------------------------------------------------------------------------

def test_extract_channel_shape():
    """
    Contract: extract_channel returns the (H, W) slice at the given index from
    a (C, H, W) array, with the correct shape, dtype, and values.

    A (5, 64, 64) uint16 patch is created where channel 2 is set to 999.
    Extracting index 2 must yield:
    - shape (64, 64)
    - dtype uint16
    - mean == 999 (entire slice is filled with 999)

    Verifies shape, dtype, and mean value of the extracted channel.
    """
    from multiplex_layers import extract_channel  # noqa: WPS433

    patch = np.zeros((5, 64, 64), dtype=np.uint16)
    patch[2] = 999

    ch = extract_channel(patch, 2)

    assert ch.shape == (64, 64), f"Expected shape (64, 64), got {ch.shape}"
    assert ch.dtype == np.uint16, f"Expected uint16, got {ch.dtype}"
    assert ch.mean() == 999, f"Expected mean 999, got {ch.mean()}"


# ---------------------------------------------------------------------------
# Unit tests — percentile_norm
# ---------------------------------------------------------------------------

def test_percentile_norm_range():
    """
    Contract: percentile_norm clips input to the [p_low, p_high] percentile
    range and normalizes the result to [0.0, 1.0] float32.

    A (64, 64) array with values spanning 0 to 65535 (via linspace) is
    normalized using the default percentiles.  The output must:
    - have dtype float32
    - have minimum ≈ 0.0 (within 0.01)
    - have maximum ≈ 1.0 (within 0.01)
    - preserve the original shape (64, 64)

    Verifies dtype, min, max, and shape.
    """
    from multiplex_layers import percentile_norm  # noqa: WPS433

    arr = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.float32)

    result = percentile_norm(arr)

    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert result.shape == (64, 64), f"Expected shape (64, 64), got {result.shape}"
    assert result.min() >= 0.0 - 0.01, f"Expected min ≈ 0.0, got {result.min()}"
    assert result.max() <= 1.0 + 0.01, f"Expected max ≈ 1.0, got {result.max()}"


def test_percentile_norm_uniform_returns_zeros():
    """
    Contract: percentile_norm returns an all-zero array when p_low == p_high,
    which occurs when the input array is uniform (all values identical).

    A (64, 64) array filled with the constant value 1000 is used.  The
    p_low and p_high percentiles are identical, so the function must return
    zeros rather than producing a divide-by-zero error or NaN values.

    Verifies that the output is entirely zero.
    """
    from multiplex_layers import percentile_norm  # noqa: WPS433

    arr = np.full((64, 64), 1000, dtype=np.float32)

    result = percentile_norm(arr)

    assert np.all(result == 0.0), (
        f"Uniform input should produce all-zero output, got max={result.max()}"
    )


# ---------------------------------------------------------------------------
# Unit tests — binarize_otsu
# ---------------------------------------------------------------------------

def test_binarize_otsu_splits_bimodal():
    """
    Contract: binarize_otsu applies Otsu thresholding to a (H, W) float32
    array in [0, 1] and returns a boolean mask of the same shape.

    A bimodal (64, 64) array is created:
    - Left half (columns 0–31): 0.1 (background)
    - Right half (columns 32–63): 0.9 (signal)

    After Otsu thresholding:
    - The left half should be almost entirely False (mean < 0.1).
    - The right half should be almost entirely True (mean > 0.9).

    Verifies dtype bool, shape (64, 64), and per-half mean values.
    """
    from multiplex_layers import binarize_otsu  # noqa: WPS433

    arr = np.zeros((64, 64), dtype=np.float32)
    arr[:, :32] = 0.1   # background
    arr[:, 32:] = 0.9   # signal

    mask = binarize_otsu(arr)

    assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"
    assert mask.shape == (64, 64), f"Expected shape (64, 64), got {mask.shape}"
    assert mask[:, :32].mean() < 0.1, (
        f"Left (background) half should be mostly False, got mean={mask[:, :32].mean():.3f}"
    )
    assert mask[:, 32:].mean() > 0.9, (
        f"Right (signal) half should be mostly True, got mean={mask[:, 32:].mean():.3f}"
    )


# ---------------------------------------------------------------------------
# Unit tests — apply_colormap
# ---------------------------------------------------------------------------

def test_apply_colormap_shape_and_dtype():
    """
    Contract: apply_colormap maps a (H, W) float32 array in [0, 1] through a
    named matplotlib colormap and returns a (H, W, 4) RGBA uint8 array.

    A 64×64 gradient from 0 to 1 is passed through the 'viridis' colormap.
    The output must:
    - have shape (64, 64, 4)
    - have dtype uint8
    - contain only values in [0, 255]

    Verifies shape, dtype, and value range.
    """
    from multiplex_layers import apply_colormap  # noqa: WPS433

    arr = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)

    out = apply_colormap(arr, "viridis")

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"
    assert out.min() >= 0, f"Values must be >= 0, got min={out.min()}"
    assert out.max() <= 255, f"Values must be <= 255, got max={out.max()}"


# ---------------------------------------------------------------------------
# Unit tests — make_vasculature_overlay
# ---------------------------------------------------------------------------

def test_make_vasculature_overlay_colors():
    """
    Contract: make_vasculature_overlay converts a boolean mask to a (H, W, 4)
    RGBA uint8 array where True pixels receive the given color and False pixels
    are fully transparent (alpha = 0).

    A (64, 64) mask is created with a 24×24 vessel block (True) centered at
    rows/cols 20–44; the rest is False.  The overlay is generated with
    color=(255, 60, 0, 200).

    Verifies:
    - shape (64, 64, 4), dtype uint8
    - vessel center (32, 32): R == 255, alpha == 200
    - background corner (0, 0): alpha == 0 (transparent)
    """
    from multiplex_layers import make_vasculature_overlay  # noqa: WPS433

    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True

    overlay = make_vasculature_overlay(mask, color=(255, 60, 0, 200))

    assert overlay.shape == (64, 64, 4), (
        f"Expected shape (64, 64, 4), got {overlay.shape}"
    )
    assert overlay.dtype == np.uint8, f"Expected uint8, got {overlay.dtype}"

    # Vessel pixel must have correct R channel and alpha
    assert overlay[32, 32, 0] == 255, (
        f"Vessel pixel R should be 255, got {overlay[32, 32, 0]}"
    )
    assert overlay[32, 32, 3] == 200, (
        f"Vessel pixel alpha should be 200, got {overlay[32, 32, 3]}"
    )

    # Background pixel must be fully transparent
    assert overlay[0, 0, 3] == 0, (
        f"Background pixel alpha should be 0, got {overlay[0, 0, 3]}"
    )


# ---------------------------------------------------------------------------
# Unit tests — make_oxygen_map
# ---------------------------------------------------------------------------

def test_make_oxygen_map_shape_and_dtype():
    """
    Contract: make_oxygen_map applies a distance-transform oxygen proxy to a
    binary vessel mask and returns a (H, W, 4) RGBA uint8 array.

    The RdYlBu colormap is used with inverted distances so that pixels near
    a vessel (distance ≈ 0) map to the blue (oxygenated) end and distant
    pixels map to the red (hypoxic) end.

    A (64, 64) mask with a center 8×8 vessel blob is used.  The output must:
    - have shape (64, 64, 4), dtype uint8
    - have higher blue channel at the vessel center than at the image corners
      (near-vessel pixels are oxygenated → blue end of RdYlBu)

    Verifies shape, dtype, and that vessel-center blue > corner blue.
    """
    from multiplex_layers import make_oxygen_map  # noqa: WPS433

    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 28:36] = True  # center 8×8 vessel blob

    out = make_oxygen_map(mask)

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    # Near vessel (center) should have higher blue than far from vessel (corner)
    center_blue = int(out[32, 32, 2])
    corner_blue = int(out[0, 0, 2])
    assert center_blue > corner_blue, (
        f"Vessel center blue ({center_blue}) should exceed corner blue ({corner_blue})"
    )


# ---------------------------------------------------------------------------
# Unit tests — make_glucose_map
# ---------------------------------------------------------------------------

def test_make_glucose_map_high_ki67_gives_bright_pixels():
    """
    Contract: make_glucose_map produces a metabolic demand proxy from Ki67
    and PCNA channels using the 'hot' colormap and returns (H, W, 4) RGBA uint8.

    The metabolic demand is max(percentile_norm(ki67), percentile_norm(pcna)).
    In the 'hot' colormap, high values map to bright (R=255, high G) and low
    values map to dark (R≈0).

    Setup:
    - ki67: a 24×24 block of value 5000 centered at rows/cols 20–44; rest zero.
    - pcna: all zeros.

    Verifies:
    - shape (64, 64, 4), dtype uint8
    - High-Ki67 center pixel (32, 32) has a higher R value than the zero
      background corner (0, 0) — confirming brighter color for high demand.
    """
    from multiplex_layers import make_glucose_map  # noqa: WPS433

    ki67 = np.zeros((64, 64), dtype=np.uint16)
    ki67[20:44, 20:44] = 5000

    pcna = np.zeros((64, 64), dtype=np.uint16)

    out = make_glucose_map(ki67.astype(np.float32), pcna.astype(np.float32))

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    # High-Ki67 center should be brighter (higher R) than zero background
    center_r = int(out[32, 32, 0])
    corner_r = int(out[0, 0, 0])
    assert center_r > corner_r, (
        f"High-Ki67 center R ({center_r}) should exceed background corner R ({corner_r})"
    )


# ---------------------------------------------------------------------------
# Unit tests — load_channel_names
# ---------------------------------------------------------------------------

def test_load_channel_names_validates_missing(tmp_path):
    """
    Contract: load_channel_names raises ValueError when any requested channel
    name is not present in the metadata CSV (case-insensitive comparison).

    A minimal metadata CSV listing only CD31 and Ki67 is created.  Requesting
    ['CD31', 'CD45'] must raise ValueError because CD45 is absent.

    Verifies pytest.raises(ValueError) is triggered.
    """
    from multiplex_layers import load_channel_names  # noqa: WPS433

    csv_path = tmp_path / "metadata.csv"
    _write_metadata_csv(csv_path, {"CD31": "Channel:0:0", "Ki67": "Channel:0:1"})

    with pytest.raises(ValueError):
        load_channel_names(str(csv_path), ["CD31", "CD45"])


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_cli_creates_all_three_layers(tmp_path):
    """
    Contract: the CLI produces vasculature/{i}_{j}.png, oxygen/{i}_{j}.png,
    and glucose/{i}_{j}.png under --out for each patch in index.json.

    Setup:
    - processed/multiplex/0_0.npy: (3, 256, 256) uint16
        channel 0 (CD31): 20×20 block of 5000, rest 0
        channel 1 (Ki67): 30×30 block of 8000, rest 0
        channel 2 (PCNA): all zeros
    - metadata.csv: CD31 → Channel:0:0, Ki67 → Channel:0:1, PCNA → Channel:0:2
    - index.json: one patch '0_0' at (x0=0, y0=0)
    - CLI run with --channels CD31 Ki67 PCNA

    Verifies:
    - vasculature/0_0.png exists and is a valid RGBA image (H, W, 4)
    - oxygen/0_0.png exists and is a valid RGBA image
    - glucose/0_0.png exists and is a valid RGBA image
    """
    from PIL import Image  # noqa: WPS433

    # Build directory layout
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    # (C=3, H=256, W=256) uint16 patch
    patch = np.zeros((3, 256, 256), dtype=np.uint16)
    patch[0, 118:138, 118:138] = 5000  # CD31: 20×20 vessel block
    patch[1, 113:143, 113:143] = 8000  # Ki67: 30×30 proliferating block
    # patch[2] PCNA: remains all zeros

    np.save(str(mux_dir / "0_0.npy"), patch)

    # Metadata CSV
    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    # index.json with a single patch entry
    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "patch_size": 256,
        "stride": 256,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        _multiplex_layers_script(),
        "--multiplex-dir", str(mux_dir),
        "--index", str(index_path),
        "--metadata-csv", str(meta_path),
        "--out", str(out_dir),
        "--channels", "CD31", "Ki67", "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"multiplex_layers.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Vasculature overlay
    vasc_path = out_dir / "vasculature" / "0_0.png"
    assert vasc_path.exists(), "vasculature/0_0.png must be created"
    vasc_img = Image.open(str(vasc_path))
    assert vasc_img.mode == "RGBA", f"Expected RGBA mode, got {vasc_img.mode}"
    vasc_arr = np.array(vasc_img)
    assert vasc_arr.shape == (256, 256, 4), (
        f"Expected (256, 256, 4), got {vasc_arr.shape}"
    )

    # Oxygen map
    oxy_path = out_dir / "oxygen" / "0_0.png"
    assert oxy_path.exists(), "oxygen/0_0.png must be created"
    oxy_img = Image.open(str(oxy_path))
    assert oxy_img.mode == "RGBA", f"Expected RGBA mode, got {oxy_img.mode}"

    # Glucose map
    gluc_path = out_dir / "glucose" / "0_0.png"
    assert gluc_path.exists(), "glucose/0_0.png must be created"
    gluc_img = Image.open(str(gluc_path))
    assert gluc_img.mode == "RGBA", f"Expected RGBA mode, got {gluc_img.mode}"


def test_cli_skips_missing_npy(tmp_path):
    """
    Contract: the CLI gracefully skips patches listed in index.json whose
    corresponding .npy file does not exist on disk (no crash).

    Setup:
    - index.json lists two patches: 0_0 and 0_1.
    - Only multiplex/0_0.npy exists; multiplex/0_1.npy is absent.
    - CLI run with --channels CD31 Ki67 PCNA.

    Verifies:
    - CLI exits with return code 0 (no crash).
    - vasculature/0_0.png exists (processed successfully).
    - vasculature/0_1.png does NOT exist (skipped without error).
    """
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    # Only patch 0_0 present
    patch = np.zeros((3, 256, 256), dtype=np.uint16)
    patch[0, 118:138, 118:138] = 5000
    np.save(str(mux_dir / "0_0.npy"), patch)
    # 0_1.npy intentionally omitted

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [
            {"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256},
            {"i": 0, "j": 1, "x0": 256, "y0": 0, "x1": 512, "y1": 256},
        ],
        "patch_size": 256,
        "stride": 256,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        _multiplex_layers_script(),
        "--multiplex-dir", str(mux_dir),
        "--index", str(index_path),
        "--metadata-csv", str(meta_path),
        "--out", str(out_dir),
        "--channels", "CD31", "Ki67", "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"multiplex_layers.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Patch 0_0 must be processed
    assert (out_dir / "vasculature" / "0_0.png").exists(), (
        "vasculature/0_0.png must be created for the existing patch"
    )

    # Patch 0_1 must be silently skipped
    assert not (out_dir / "vasculature" / "0_1.png").exists(), (
        "vasculature/0_1.png must NOT be created when .npy is missing"
    )
