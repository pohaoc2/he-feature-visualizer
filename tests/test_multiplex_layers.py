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

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _multiplex_layers_cmd() -> list[str]:
    """Return subprocess args for running stages.multiplex_layers as a module."""
    return [sys.executable, "-m", "stages.multiplex_layers"]


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
    from stages.multiplex_layers import load_multiplex_patch  # noqa: WPS433

    arr = np.zeros((3, 64, 64), dtype=np.uint16)
    arr[0] = 100
    arr[1] = 200
    arr[2] = 300

    npy_path = tmp_path / "0_0.npy"
    np.save(str(npy_path), arr)

    result = load_multiplex_patch(str(npy_path))

    assert result.shape == (
        3,
        64,
        64,
    ), f"Expected shape (3, 64, 64), got {result.shape}"
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
    from stages.multiplex_layers import get_channel_index  # noqa: WPS433

    names = ["DNA", "CD31", "Ki67", "CD45"]

    assert (
        get_channel_index(names, "cd31") == 1
    ), "Case-insensitive lookup of 'cd31' should return index 1"
    assert (
        get_channel_index(names, "Ki67") == 2
    ), "Exact-case lookup of 'Ki67' should return index 2"
    assert (
        get_channel_index(names, "CD45") == 3
    ), "Exact-case lookup of 'CD45' should return index 3"


def test_get_channel_index_raises_on_missing():
    """
    Contract: get_channel_index raises ValueError when the target is not
    present in the channel_names list.

    A list without 'PCNA' is used; requesting 'PCNA' must raise ValueError.

    Verifies pytest.raises(ValueError) is triggered.
    """
    from stages.multiplex_layers import get_channel_index  # noqa: WPS433

    names = ["DNA", "CD31", "Ki67"]

    with pytest.raises(ValueError):
        get_channel_index(names, "PCNA")


def test_get_first_matching_channel_index_respects_candidate_order():
    """
    Contract: get_first_matching_channel_index returns the first candidate name
    (by candidate list order) found in channel_names.
    """
    from stages.multiplex_layers import get_first_matching_channel_index  # noqa: WPS433

    names = ["DNA", "CD31", "Ki67", "PCNA"]
    candidates = ["missing", "pcna", "cd31"]
    idx = get_first_matching_channel_index(names, candidates)
    assert idx == 3, f"Expected first matched candidate index 3 (PCNA), got {idx}"


def test_get_first_matching_channel_index_returns_none_when_missing():
    """If none of the candidate names exist, function returns None."""
    from stages.multiplex_layers import get_first_matching_channel_index  # noqa: WPS433

    names = ["DNA", "CD31", "Ki67"]
    idx = get_first_matching_channel_index(names, ["foo", "bar", "baz"])
    assert idx is None


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
    from stages.multiplex_layers import extract_channel  # noqa: WPS433

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
    from utils.normalize import percentile_norm  # noqa: WPS433

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
    from utils.normalize import percentile_norm  # noqa: WPS433

    arr = np.full((64, 64), 1000, dtype=np.float32)

    result = percentile_norm(arr)

    assert np.all(
        result == 0.0
    ), f"Uniform input should produce all-zero output, got max={result.max()}"


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
    from stages.multiplex_layers import binarize_otsu  # noqa: WPS433

    arr = np.zeros((64, 64), dtype=np.float32)
    arr[:, :32] = 0.1  # background
    arr[:, 32:] = 0.9  # signal

    mask = binarize_otsu(arr)

    assert mask.dtype == bool, f"Expected bool dtype, got {mask.dtype}"
    assert mask.shape == (64, 64), f"Expected shape (64, 64), got {mask.shape}"
    assert (
        mask[:, :32].mean() < 0.1
    ), f"Left (background) half should be mostly False, got mean={mask[:, :32].mean():.3f}"
    assert (
        mask[:, 32:].mean() > 0.9
    ), f"Right (signal) half should be mostly True, got mean={mask[:, 32:].mean():.3f}"


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
    from stages.multiplex_layers import apply_colormap  # noqa: WPS433

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
    from stages.multiplex_layers import make_vasculature_overlay  # noqa: WPS433

    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True

    overlay = make_vasculature_overlay(mask, color=(255, 60, 0, 200))

    assert overlay.shape == (
        64,
        64,
        4,
    ), f"Expected shape (64, 64, 4), got {overlay.shape}"
    assert overlay.dtype == np.uint8, f"Expected uint8, got {overlay.dtype}"

    # Vessel pixel must have correct R channel and alpha
    assert (
        overlay[32, 32, 0] == 255
    ), f"Vessel pixel R should be 255, got {overlay[32, 32, 0]}"
    assert (
        overlay[32, 32, 3] == 200
    ), f"Vessel pixel alpha should be 200, got {overlay[32, 32, 3]}"

    # Background pixel must be fully transparent
    assert (
        overlay[0, 0, 3] == 0
    ), f"Background pixel alpha should be 0, got {overlay[0, 0, 3]}"


def test_refine_vasculature_with_sma_keeps_only_adjacent_sma():
    """
    Contract: SMA refinement must only add SMA-positive pixels adjacent to CD31.

    Setup:
    - cd31_mask contains a 2x2 vessel at rows/cols 4:6.
    - sma_mask contains one adjacent pixel at (6, 5) and one far pixel at (0, 0).
    - adjacency radius is 1 px.

    Verifies:
    - adjacent SMA pixel is included in refined mask.
    - far SMA pixel is not included.
    """
    from stages.multiplex_layers import refine_vasculature_with_sma  # noqa: WPS433

    cd31_mask = np.zeros((10, 10), dtype=bool)
    cd31_mask[4:6, 4:6] = True

    sma_mask = np.zeros((10, 10), dtype=bool)
    sma_mask[6, 5] = True  # adjacent to vessel component
    sma_mask[0, 0] = True  # far from vessel

    refined = refine_vasculature_with_sma(cd31_mask, sma_mask, adjacency_px=1)

    assert refined[6, 5], "Adjacent SMA-positive pixel should be included"
    assert not refined[0, 0], "Far SMA-positive pixel should not be included"


def test_cleanup_vasculature_mask_removes_small_components():
    """Cleanup should remove components smaller than min_area."""
    from stages.multiplex_layers import cleanup_vasculature_mask  # noqa: WPS433

    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True  # area 16, should remain
    mask[10, 10] = True  # area 1, should be removed
    mask[15, 15] = True  # area 1, should be removed

    cleaned = cleanup_vasculature_mask(
        mask,
        open_kernel_size=0,
        close_kernel_size=0,
        min_area=4,
    )

    assert cleaned.dtype == bool
    assert cleaned[3, 3], "Large component should be retained"
    assert not cleaned[10, 10], "Small speckle should be removed"
    assert not cleaned[15, 15], "Small speckle should be removed"


def test_apply_vessel_mask_quality_fallback_handles_empty_and_noisy():
    """Quality fallback should handle empty and noisy masks deterministically."""
    from stages.multiplex_layers import (
        apply_vessel_mask_quality_fallback,
    )  # noqa: WPS433

    fallback = np.zeros((8, 8), dtype=bool)
    fallback[3:5, 3:5] = True

    empty = np.zeros((8, 8), dtype=bool)
    empty_out, empty_status = apply_vessel_mask_quality_fallback(
        candidate_mask=empty,
        cd31_fallback_mask=fallback,
        noisy_max_fraction=0.9,
    )
    assert empty_status == "empty_fallback"
    assert np.array_equal(empty_out, fallback)

    noisy = np.ones((8, 8), dtype=bool)
    noisy_out, noisy_status = apply_vessel_mask_quality_fallback(
        candidate_mask=noisy,
        cd31_fallback_mask=fallback,
        noisy_max_fraction=0.9,
    )
    assert noisy_status == "noisy_fallback"
    assert np.array_equal(noisy_out, fallback)


def test_apply_vessel_mask_quality_fallback_returns_empty_when_both_empty():
    """When both candidate and fallback are empty, status should be 'empty'."""
    from stages.multiplex_layers import (
        apply_vessel_mask_quality_fallback,
    )  # noqa: WPS433

    candidate = np.zeros((10, 10), dtype=bool)
    fallback = np.zeros((10, 10), dtype=bool)
    out, status = apply_vessel_mask_quality_fallback(
        candidate_mask=candidate,
        cd31_fallback_mask=fallback,
        noisy_max_fraction=0.95,
    )
    assert status == "empty"
    assert out.dtype == bool
    assert not np.any(out)


def test_build_vessel_source_map_applies_mask_and_weight():
    """Vessel source map should keep vessel pixels and apply optional weighting."""
    from stages.multiplex_layers import build_vessel_source_map  # noqa: WPS433

    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True

    weight = np.zeros((4, 4), dtype=np.float32)
    weight[1, 1] = 0.25
    weight[2, 2] = 0.75

    source = build_vessel_source_map(mask, weight)

    assert source.dtype == np.float32
    assert source.shape == (4, 4)
    assert source[1, 1] == pytest.approx(0.25)
    assert source[2, 2] == pytest.approx(0.75)
    assert source[0, 0] == pytest.approx(0.0)


def test_build_consumption_map_combines_base_and_demand_weight():
    """Consumption map should be base + demand_weight*demand."""
    from stages.multiplex_layers import build_consumption_map  # noqa: WPS433

    demand = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    k = build_consumption_map(demand, base_rate=0.1, demand_weight=0.4)

    assert k.dtype == np.float32
    assert k.shape == (1, 3)
    assert k[0, 0] == pytest.approx(0.1)
    assert k[0, 1] == pytest.approx(0.3)
    assert k[0, 2] == pytest.approx(0.5)


def test_solve_steady_state_diffusion_is_bounded_and_nonzero_for_source():
    """PDE solver should return bounded field and respond to nonzero source."""
    from stages.multiplex_layers import solve_steady_state_diffusion  # noqa: WPS433

    src = np.zeros((33, 33), dtype=np.float32)
    src[16, 16] = 1.0
    k = np.full((33, 33), 0.2, dtype=np.float32)

    u = solve_steady_state_diffusion(
        source_map=src,
        consumption_map=k,
        diffusion=1.0,
        max_iters=800,
        tol=1e-5,
    )

    assert u.dtype == np.float32
    assert u.shape == (33, 33)
    assert float(u.min()) >= 0.0
    assert float(u.max()) <= 1.0
    assert float(u[16, 16]) > float(u[0, 0]), "Center should exceed corner"
    assert float(u[16, 16]) > 0.0, "Center response must be nonzero"


def test_solve_steady_state_diffusion_has_radial_decay_from_center_source():
    """Homogeneous medium should decay with distance from a central source."""
    from stages.multiplex_layers import solve_steady_state_diffusion  # noqa: WPS433

    src = np.zeros((65, 65), dtype=np.float32)
    src[32, 32] = 1.0
    k = np.full((65, 65), 0.15, dtype=np.float32)

    u = solve_steady_state_diffusion(
        source_map=src,
        consumption_map=k,
        diffusion=1.0,
        max_iters=1000,
        tol=1e-5,
    )

    center = float(u[32, 32])
    near = float(u[32, 37])  # distance 5
    far = float(u[32, 47])  # distance 15
    assert center > near > far, (
        f"Expected center>near>far, got center={center:.4f}, "
        f"near={near:.4f}, far={far:.4f}"
    )


def test_solve_steady_state_diffusion_raises_on_shape_mismatch():
    """Solver should reject incompatible source/consumption shapes."""
    from stages.multiplex_layers import solve_steady_state_diffusion  # noqa: WPS433

    src = np.zeros((8, 8), dtype=np.float32)
    k = np.zeros((7, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        solve_steady_state_diffusion(src, k)


def test_compute_metabolic_demand_map_uses_channelwise_max():
    """Demand map should be max(norm(Ki67), norm(PCNA)) per pixel."""
    from stages.multiplex_layers import compute_metabolic_demand_map  # noqa: WPS433

    ki67 = np.zeros((4, 4), dtype=np.float32)
    pcna = np.zeros((4, 4), dtype=np.float32)
    ki67[1, 1] = 100.0
    pcna[2, 2] = 200.0

    demand = compute_metabolic_demand_map(ki67, pcna)

    assert demand.shape == (4, 4)
    assert demand.dtype == np.float32
    assert float(demand[1, 1]) > 0.0
    assert float(demand[2, 2]) > 0.0
    assert float(demand[0, 0]) == pytest.approx(0.0)


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
    from stages.multiplex_layers import make_oxygen_map  # noqa: WPS433

    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 28:36] = True  # center 8×8 vessel blob

    out = make_oxygen_map(mask)

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    # Near vessel (center) should have higher blue than far from vessel (corner)
    center_blue = int(out[32, 32, 2])
    corner_blue = int(out[0, 0, 2])
    assert (
        center_blue > corner_blue
    ), f"Vessel center blue ({center_blue}) should exceed corner blue ({corner_blue})"


def test_make_oxygen_map_pde_shape_and_dtype():
    """PDE oxygen map should be RGBA and higher near vessel source."""
    from stages.multiplex_layers import make_oxygen_map_pde  # noqa: WPS433

    vessel_mask = np.zeros((64, 64), dtype=bool)
    vessel_mask[28:36, 28:36] = True
    demand_map = np.zeros((64, 64), dtype=np.float32)

    out = make_oxygen_map_pde(
        vessel_mask=vessel_mask,
        demand_map=demand_map,
        diffusion=1.0,
        max_iters=500,
        tol=1e-5,
        base_consumption=0.1,
        demand_weight=0.3,
    )

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    center_blue = int(out[32, 32, 2])
    corner_blue = int(out[0, 0, 2])
    assert (
        center_blue > corner_blue
    ), f"PDE oxygen center blue ({center_blue}) should exceed corner blue ({corner_blue})"


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
    from stages.multiplex_layers import make_glucose_map  # noqa: WPS433

    ki67 = np.zeros((64, 64), dtype=np.uint16)
    ki67[20:44, 20:44] = 5000

    pcna = np.zeros((64, 64), dtype=np.uint16)

    out = make_glucose_map(ki67.astype(np.float32), pcna.astype(np.float32))

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    # High-Ki67 center should be brighter (higher R) than zero background
    center_r = int(out[32, 32, 0])
    corner_r = int(out[0, 0, 0])
    assert (
        center_r > corner_r
    ), f"High-Ki67 center R ({center_r}) should exceed background corner R ({corner_r})"


def test_make_glucose_map_pde_shape_and_dtype():
    """PDE glucose map should be RGBA and brighter near vessel source."""
    from stages.multiplex_layers import make_glucose_map_pde  # noqa: WPS433

    vessel_mask = np.zeros((64, 64), dtype=bool)
    vessel_mask[28:36, 28:36] = True
    demand_map = np.zeros((64, 64), dtype=np.float32)
    demand_map[:, :32] = 1.0  # heterogeneous demand to exercise consumption map

    out = make_glucose_map_pde(
        vessel_mask=vessel_mask,
        demand_map=demand_map,
        diffusion=1.0,
        max_iters=500,
        tol=1e-5,
        base_consumption=0.1,
        demand_weight=0.3,
    )

    assert out.shape == (64, 64, 4), f"Expected shape (64, 64, 4), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    center_r = int(out[32, 32, 0])
    corner_r = int(out[0, 0, 0])
    assert (
        center_r > corner_r
    ), f"PDE glucose center R ({center_r}) should exceed corner R ({corner_r})"


# ---------------------------------------------------------------------------
# Unit tests — resolve_channel_indices (was load_channel_names)
# ---------------------------------------------------------------------------


def test_resolve_channel_indices_validates_missing(tmp_path):
    """
    Contract: resolve_channel_indices raises ValueError when any requested
    channel name is not present in the metadata CSV (case-insensitive).

    A minimal metadata CSV listing only CD31 and Ki67 is created. Requesting
    ['CD31', 'CD45'] must raise ValueError because CD45 is absent.
    """
    from utils.channels import resolve_channel_indices  # noqa: WPS433

    csv_path = tmp_path / "metadata.csv"
    _write_metadata_csv(csv_path, {"CD31": "Channel:0:0", "Ki67": "Channel:0:1"})

    with pytest.raises(ValueError):
        resolve_channel_indices(str(csv_path), ["CD31", "CD45"])


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_main_inprocess_runs_and_writes_outputs(tmp_path, monkeypatch):
    """In-process main() execution should write all expected output artifacts."""
    import stages.multiplex_layers as m  # noqa: WPS433

    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)
    patch = np.zeros((3, 128, 128), dtype=np.uint16)
    patch[0, 56:72, 56:72] = 5000  # CD31
    patch[1, 40:88, 40:88] = 8000  # Ki67
    np.save(str(mux_dir / "0_0.npy"), patch)

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 128, "y1": 128}],
        "patch_size": 128,
        "stride": 128,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multiplex_layers.py",
            "--multiplex-dir",
            str(mux_dir),
            "--index",
            str(index_path),
            "--metadata-csv",
            str(meta_path),
            "--out",
            str(out_dir),
            "--channels",
            "CD31",
            "Ki67",
            "PCNA",
        ],
    )

    m.main()

    assert (out_dir / "vasculature" / "0_0.png").exists()
    assert (out_dir / "vasculature_mask" / "0_0.npy").exists()
    assert (out_dir / "oxygen" / "0_0.png").exists()
    assert (out_dir / "glucose" / "0_0.png").exists()


def test_main_inprocess_rejects_even_open_kernel_size(monkeypatch):
    """In-process main() should raise ValueError for even open-kernel size."""
    import stages.multiplex_layers as m  # noqa: WPS433

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multiplex_layers.py",
            "--multiplex-dir",
            "dummy",
            "--index",
            "dummy",
            "--metadata-csv",
            "dummy",
            "--vasc-open-kernel-size",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="vasc_open_kernel_size must be odd"):
        m.main()


def test_cli_creates_all_three_layers(tmp_path):
    """
    Contract: the CLI produces vasculature/{i}_{j}.png,
    vasculature_mask/{i}_{j}.npy, oxygen/{i}_{j}.png, and glucose/{i}_{j}.png
    under --out for each patch in index.json.

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
    - vasculature_mask/0_0.npy exists and is a boolean mask with shape (256, 256)
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
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    vasc_path = out_dir / "vasculature" / "0_0.png"
    assert vasc_path.exists(), "vasculature/0_0.png must be created"
    vasc_img = Image.open(str(vasc_path))
    assert vasc_img.mode == "RGBA", f"Expected RGBA mode, got {vasc_img.mode}"
    vasc_arr = np.array(vasc_img)
    assert vasc_arr.shape == (
        256,
        256,
        4,
    ), f"Expected (256, 256, 4), got {vasc_arr.shape}"

    # Vasculature binary mask
    vasc_mask_path = out_dir / "vasculature_mask" / "0_0.npy"
    assert vasc_mask_path.exists(), "vasculature_mask/0_0.npy must be created"
    vasc_mask = np.load(str(vasc_mask_path))
    assert vasc_mask.shape == (
        256,
        256,
    ), f"Expected (256, 256), got {vasc_mask.shape}"
    assert (
        vasc_mask.dtype == np.bool_
    ), f"Expected bool mask dtype, got {vasc_mask.dtype}"

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
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Patch 0_0 must be processed
    assert (
        out_dir / "vasculature" / "0_0.png"
    ).exists(), "vasculature/0_0.png must be created for the existing patch"
    assert (
        out_dir / "vasculature_mask" / "0_0.npy"
    ).exists(), "vasculature_mask/0_0.npy must be created for the existing patch"

    # Patch 0_1 must be silently skipped
    assert not (
        out_dir / "vasculature" / "0_1.png"
    ).exists(), "vasculature/0_1.png must NOT be created when .npy is missing"
    assert not (
        out_dir / "vasculature_mask" / "0_1.npy"
    ).exists(), "vasculature_mask/0_1.npy must NOT be created when .npy is missing"


def test_cli_rejects_invalid_vasc_noisy_max_fraction():
    """CLI must reject vessel-mask noisy-threshold values outside (0, 1]."""
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        "dummy_multiplex",
        "--index",
        "dummy_index.json",
        "--metadata-csv",
        "dummy_metadata.csv",
        "--vasc-noisy-max-fraction",
        "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode != 0
    assert "--vasc-noisy-max-fraction must be in (0, 1]" in result.stderr


def test_cli_rejects_negative_vasc_min_area():
    """CLI must reject negative values for --vasc-min-area."""
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        "dummy_multiplex",
        "--index",
        "dummy_index.json",
        "--metadata-csv",
        "dummy_metadata.csv",
        "--vasc-min-area",
        "-1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode != 0
    assert "--vasc-min-area must be >= 0" in result.stderr


def test_cli_rejects_even_vasc_open_kernel_size():
    """CLI must reject even values for --vasc-open-kernel-size."""
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        "dummy_multiplex",
        "--index",
        "dummy_index.json",
        "--metadata-csv",
        "dummy_metadata.csv",
        "--vasc-open-kernel-size",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode != 0
    assert "vasc_open_kernel_size must be odd" in result.stderr


def test_cli_rejects_even_vasc_close_kernel_size():
    """CLI must reject even values for --vasc-close-kernel-size."""
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        "dummy_multiplex",
        "--index",
        "dummy_index.json",
        "--metadata-csv",
        "dummy_metadata.csv",
        "--vasc-close-kernel-size",
        "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode != 0
    assert "vasc_close_kernel_size must be odd" in result.stderr


def test_cli_runs_pde_models_and_writes_outputs(tmp_path):
    """
    Contract: the CLI accepts --oxygen-model pde and --glucose-model pde
    and produces output artifacts.
    """
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    patch = np.zeros((3, 128, 128), dtype=np.uint16)
    patch[0, 56:72, 56:72] = 5000
    patch[1, 40:88, 40:88] = 8000
    np.save(str(mux_dir / "0_0.npy"), patch)

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 128, "y1": 128}],
        "patch_size": 128,
        "stride": 128,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
        "--oxygen-model",
        "pde",
        "--glucose-model",
        "pde",
        "--pde-max-iters",
        "120",
        "--pde-tol",
        "1e-4",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (
        "Falling back" not in result.stderr
    ), "PDE model run should not log fallback behavior once PDE wiring is enabled"

    assert (
        out_dir / "vasculature_mask" / "0_0.npy"
    ).exists(), "vasculature_mask/0_0.npy must be created in pde mode"
    assert (out_dir / "oxygen" / "0_0.png").exists(), "oxygen/0_0.png must be created"
    assert (out_dir / "glucose" / "0_0.png").exists(), "glucose/0_0.png must be created"


def test_cli_no_vessel_pixels_produces_deterministic_empty_mask(tmp_path):
    """No-vessel input should complete and write an all-false vessel mask."""
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    # CD31 is all zeros; Ki67 carries a small nonzero region for nontrivial demand map.
    patch = np.zeros((3, 96, 96), dtype=np.uint16)
    patch[1, 40:56, 40:56] = 6000
    np.save(str(mux_dir / "0_0.npy"), patch)

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 96, "y1": 96}],
        "patch_size": 96,
        "stride": 96,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    vasc_mask = np.load(str(out_dir / "vasculature_mask" / "0_0.npy"))
    assert vasc_mask.dtype == np.bool_
    assert not np.any(vasc_mask), "No-vessel patch should produce an all-false mask"
    assert "vessel mask is empty" in result.stderr


def test_cli_uniform_channels_runs_and_writes_outputs(tmp_path):
    """Uniform channels should not crash and should still write all artifacts."""
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    # Uniform values across all channels exercise percentile_norm degenerate path.
    patch = np.full((3, 80, 80), 1234, dtype=np.uint16)
    np.save(str(mux_dir / "0_0.npy"), patch)

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 80, "y1": 80}],
        "patch_size": 80,
        "stride": 80,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    assert (out_dir / "vasculature_mask" / "0_0.npy").exists()
    assert (out_dir / "vasculature" / "0_0.png").exists()
    assert (out_dir / "oxygen" / "0_0.png").exists()
    assert (out_dir / "glucose" / "0_0.png").exists()


def test_cli_missing_sma_channel_without_refinement_runs(tmp_path):
    """Missing SMA channel should be irrelevant when SMA refinement is disabled."""
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    patch = np.zeros((3, 96, 96), dtype=np.uint16)
    patch[0, 42:54, 42:54] = 5000
    patch[1, 36:60, 36:60] = 7000
    np.save(str(mux_dir / "0_0.npy"), patch)

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 96, "y1": 96}],
        "patch_size": 96,
        "stride": 96,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SMA refinement requested" not in result.stderr


def test_cli_missing_sma_channel_with_refinement_warns_and_runs(tmp_path):
    """Enabled SMA refinement should warn and continue when SMA channel is absent."""
    mux_dir = tmp_path / "multiplex"
    mux_dir.mkdir(parents=True)

    patch = np.zeros((3, 96, 96), dtype=np.uint16)
    patch[0, 42:54, 42:54] = 5000
    patch[1, 36:60, 36:60] = 7000
    np.save(str(mux_dir / "0_0.npy"), patch)

    meta_path = tmp_path / "metadata.csv"
    _write_metadata_csv(
        meta_path,
        {"CD31": "Channel:0:0", "Ki67": "Channel:0:1", "PCNA": "Channel:0:2"},
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 96, "y1": 96}],
        "patch_size": 96,
        "stride": 96,
        "channels": ["CD31", "Ki67", "PCNA"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    cmd = [
        *_multiplex_layers_cmd(),
        "--multiplex-dir",
        str(mux_dir),
        "--index",
        str(index_path),
        "--metadata-csv",
        str(meta_path),
        "--out",
        str(out_dir),
        "--channels",
        "CD31",
        "Ki67",
        "PCNA",
        "--vasc-sma-refine",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.multiplex_layers exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "SMA refinement requested, but none of" in result.stderr
