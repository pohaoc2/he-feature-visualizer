"""
Tests for H&E ↔ CyCIF (multiplex) image mapping in stages/patchify.py.

Covered functions
-----------------
transform_he_to_mx_point  -- affine point mapping from H&E coords to MX coords
register_he_mx_affine     -- ECC registration + mpp-scale fallback
build_mx_tissue_mask      -- DNA-channel binary tissue mask for registration
get_tissue_patches        -- tissue-threshold coordinate selection
read_he_patch             -- boundary clamping, zero-padding, axis layouts
read_multiplex_patch      -- boundary clamping, channel selection, zero-padding
CLI --no-register         -- scale-only warp matrix stored in index.json
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stages.patchify as m

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _write_ome(path: Path, arr: np.ndarray, axes: str) -> None:
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": axes})


def _write_metadata_csv(path: Path, names: list[str]) -> None:
    lines = ["Channel ID,Target Name\n"]
    for i, n in enumerate(names):
        lines.append(f"Channel:0:{i},{n}\n")
    path.write_text("".join(lines))


def _identity_M() -> np.ndarray:
    """Return a 2×3 identity affine matrix (no transform)."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


# ===========================================================================
# transform_he_to_mx_point
# ===========================================================================


class TestTransformHEToMXPoint:
    """Unit tests for the affine point-mapping helper."""

    def test_identity_maps_to_self(self):
        """Identity matrix: (x, y) -> (x, y)."""
        M = _identity_M()
        assert m.transform_he_to_mx_point(M, 100, 200) == (100, 200)

    def test_pure_x_translation(self):
        """Translation in x only shifts x-coordinate."""
        M = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 0, 0) == (50, 0)

    def test_pure_y_translation(self):
        """Translation in y only shifts y-coordinate."""
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 30.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 0, 0) == (0, 30)

    def test_combined_translation(self):
        """Translation in x and y both applied correctly."""
        M = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 75.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 0, 0) == (100, 75)

    def test_uniform_scale_down(self):
        """Scale of 0.5 maps (200, 400) -> (100, 200)."""
        s = 0.5
        M = np.array([[s, 0.0, 0.0], [0.0, s, 0.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 200, 400) == (100, 200)

    def test_uniform_scale_up(self):
        """Scale of 2.0 maps (100, 50) -> (200, 100)."""
        s = 2.0
        M = np.array([[s, 0.0, 0.0], [0.0, s, 0.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 100, 50) == (200, 100)

    def test_scale_with_translation(self):
        """Scale 0.5 plus translation: (256, 256) -> (128 + tx, 128 + ty)."""
        s, tx, ty = 0.5, 20.0, -10.0
        M = np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float32)
        x_mx, y_mx = m.transform_he_to_mx_point(M, 256, 256)
        assert x_mx == round(256 * s + tx)
        assert y_mx == round(256 * s + ty)

    def test_rounding_half_pixel(self):
        """Half-pixel offsets are rounded (standard Python round-half-to-even)."""
        # 0.5 scale on 1 -> 0.5, rounds to 0
        M = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
        x_mx, y_mx = m.transform_he_to_mx_point(M, 1, 1)
        assert isinstance(x_mx, int)
        assert isinstance(y_mx, int)

    def test_origin_maps_to_translation(self):
        """Origin (0, 0) with any M maps to the translation column."""
        tx, ty = 333, 777
        M = np.array([[2.0, 0.5, float(tx)], [0.3, 1.5, float(ty)]], dtype=np.float32)
        x_mx, y_mx = m.transform_he_to_mx_point(M, 0, 0)
        assert x_mx == tx
        assert y_mx == ty

    def test_non_uniform_scale(self):
        """Different x and y scale factors are applied independently."""
        M = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 10, 10) == (20, 30)

    def test_general_affine_shear(self):
        """General affine with shear: result matches numpy dot product."""
        M = np.array([[1.2, 0.1, 15.0], [0.2, 0.9, 25.0]], dtype=np.float32)
        x0, y0 = 100, 200
        pt = np.array([x0, y0, 1.0], dtype=np.float64)
        expected = M.astype(np.float64) @ pt
        x_mx, y_mx = m.transform_he_to_mx_point(M, x0, y0)
        assert x_mx == int(round(expected[0]))
        assert y_mx == int(round(expected[1]))

    def test_negative_translation(self):
        """Negative translation (MX is larger / shifted) handled correctly."""
        M = np.array([[1.0, 0.0, -50.0], [0.0, 1.0, -30.0]], dtype=np.float32)
        assert m.transform_he_to_mx_point(M, 100, 80) == (50, 50)

    def test_float32_matrix_used_correctly(self):
        """float32 matrix coerced to float64 for the dot product."""
        M = np.array([[0.999, 0.0, 0.0], [0.0, 0.999, 0.0]], dtype=np.float32)
        x_mx, y_mx = m.transform_he_to_mx_point(M, 1000, 1000)
        assert isinstance(x_mx, int)
        assert isinstance(y_mx, int)
        assert x_mx == int(round(999.0))
        assert y_mx == int(round(999.0))


# ===========================================================================
# register_he_mx_affine — output contract and fallback
# ===========================================================================


class TestRegisterHEMXAffine:
    """Tests for the ECC-based registration function."""

    def test_output_shape_and_dtype(self):
        """Always returns a (2, 3) float32 ndarray."""
        # Use small identical masks so ECC has a chance to converge
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        M = m.register_he_mx_affine(
            mask, mask, ds=16, he_h=128, he_w=128, mx_h=128, mx_w=128
        )
        assert M.shape == (2, 3)
        assert M.dtype == np.float32

    def test_resize_center_mapping_correction_applied(self, monkeypatch):
        """Resized-space ECC translation is converted with pixel-center correction."""
        he_mask = np.zeros((120, 180), dtype=bool)
        mx_mask = np.zeros((80, 90), dtype=bool)
        ds = 64

        m_ov = np.array(
            [[1.02, 0.03, 4.25], [-0.02, 0.98, -7.75]],
            dtype=np.float32,
        )

        def _fake_ecc(*args, **kwargs):
            return 1.0, m_ov.copy()

        monkeypatch.setattr(m._registration.cv2, "findTransformECC", _fake_ecc)

        M = m.register_he_mx_affine(
            he_mask,
            mx_mask,
            ds=ds,
            he_h=he_mask.shape[0] * ds,
            he_w=he_mask.shape[1] * ds,
            mx_h=mx_mask.shape[0] * ds,
            mx_w=mx_mask.shape[1] * ds,
        )

        rx = he_mask.shape[1] / mx_mask.shape[1]
        ry = he_mask.shape[0] / mx_mask.shape[0]
        expected = np.array(
            [
                [
                    m_ov[0, 0] / rx,
                    m_ov[0, 1] / rx,
                    (((m_ov[0, 2] + 0.5) / rx) - 0.5) * ds,
                ],
                [
                    m_ov[1, 0] / ry,
                    m_ov[1, 1] / ry,
                    (((m_ov[1, 2] + 0.5) / ry) - 0.5) * ds,
                ],
            ],
            dtype=np.float32,
        )

        assert M == pytest.approx(expected, abs=1e-5)

    def test_fallback_on_uniform_masks(self):
        """All-zero masks cause ECC to fail; fallback returns scale-only matrix."""
        he_mask = np.zeros((8, 8), dtype=bool)
        mx_mask = np.zeros((8, 8), dtype=bool)
        # he_w=256, mx_w=128 → expected fallback scale = he_w/mx_w = 2.0
        # M_full = [[1/scale, 0, 0], [0, 1/scale, 0]] = [[0.5, 0, 0], [0, 0.5, 0]]
        M = m.register_he_mx_affine(
            he_mask, mx_mask, ds=16, he_h=128, he_w=256, mx_h=64, mx_w=128
        )
        assert M.shape == (2, 3)
        # Scale is 1/2 since HE is 2x larger than MX
        assert abs(M[0, 0] - 0.5) < 0.01, f"Expected scale ~0.5, got {M[0, 0]}"
        assert abs(M[1, 1] - 0.5) < 0.01, f"Expected scale ~0.5, got {M[1, 1]}"
        # No shear in fallback
        assert M[0, 1] == pytest.approx(0.0, abs=0.01)
        assert M[1, 0] == pytest.approx(0.0, abs=0.01)

    def test_fallback_scale_identity_when_same_size(self):
        """Fallback on same-size images produces scale=1 (identity) matrix."""
        he_mask = np.zeros((8, 8), dtype=bool)
        mx_mask = np.zeros((8, 8), dtype=bool)
        M = m.register_he_mx_affine(
            he_mask, mx_mask, ds=16, he_h=128, he_w=128, mx_h=128, mx_w=128
        )
        assert abs(M[0, 0] - 1.0) < 0.01
        assert abs(M[1, 1] - 1.0) < 0.01

    def test_identical_masks_near_identity(self):
        """Identical masks with sufficient structure → M_full close to identity."""
        rng = np.random.default_rng(0)
        # 32x32 mask with a large centered blob — gives ECC enough signal
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        M = m.register_he_mx_affine(
            mask, mask, ds=4, he_h=128, he_w=128, mx_h=128, mx_w=128
        )
        assert M.shape == (2, 3)
        # Diagonal should be close to 1.0 (near-identity warp)
        assert abs(M[0, 0] - 1.0) < 0.2, f"M[0,0]={M[0,0]}: expected ~1.0"
        assert abs(M[1, 1] - 1.0) < 0.2, f"M[1,1]={M[1,1]}: expected ~1.0"

    def test_recenters_residual_centroid_translation(self, monkeypatch):
        """Residual centroid bias after ECC is corrected via translation recentering."""
        he_mask = np.zeros((64, 64), dtype=bool)
        he_mask[18:46, 18:46] = True
        mx_mask = np.zeros((64, 64), dtype=bool)
        # Shift MX tissue down by +5 px relative to H&E.
        mx_mask[23:51, 18:46] = True

        def _fake_ecc(*args, **kwargs):
            return 1.0, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        monkeypatch.setattr(m._registration.cv2, "findTransformECC", _fake_ecc)

        M = m.register_he_mx_affine(
            he_mask,
            mx_mask,
            ds=1,
            he_h=64,
            he_w=64,
            mx_h=64,
            mx_w=64,
        )

        # Positive ty in a WARP_INVERSE_MAP matrix moves warped content upward.
        # MX was low by ~5 px, so recentering should add roughly +5 px ty.
        assert M[1, 2] == pytest.approx(5.0, abs=1.0)

    def test_no_register_branch_uses_scale_matrix(self, tmp_path):
        """--no-register CLI flag stores a pure scale matrix in index.json."""
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        he = np.full((3, 512, 512), 240, dtype=np.uint8)
        he[:, 128:384, 128:384] = np.array([180, 60, 120])[:, None, None]
        _write_ome(he_path, he, "CYX")

        mx = np.zeros((4, 512, 512), dtype=np.uint16)
        mx[:, 128:384, 128:384] = 1000
        _write_ome(mx_path, mx, "CYX")
        _write_metadata_csv(csv_path, ["CD31", "Ki67", "CD45", "PCNA"])

        cmd = [
            sys.executable,
            "-m",
            "stages.patchify",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--metadata-csv",
            str(csv_path),
            "--out",
            str(out_dir),
            "--patch-size",
            "256",
            "--stride",
            "256",
            "--tissue-min",
            "0.05",
            "--channels",
            "CD31",
            "Ki67",
            "CD45",
            "PCNA",
            "--no-register",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
        assert result.returncode == 0, result.stderr

        data = json.loads((out_dir / "index.json").read_text())
        assert data["registration"] is False, "Expected registration=False"
        M = np.array(data["warp_matrix"])
        assert M.shape == (2, 3)
        # No-register: off-diagonal shear elements must be zero
        assert M[0, 1] == pytest.approx(0.0)
        assert M[1, 0] == pytest.approx(0.0)
        # Translation is zero
        assert M[0, 2] == pytest.approx(0.0)
        assert M[1, 2] == pytest.approx(0.0)

    def test_warp_matrix_stored_in_index_json(self, tmp_path):
        """index.json always contains a 2×3 warp_matrix (both register paths)."""
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        he = np.full((3, 512, 512), 240, dtype=np.uint8)
        he[:, 100:400, 100:400] = np.array([180, 60, 120])[:, None, None]
        _write_ome(he_path, he, "CYX")
        mx = np.zeros((4, 512, 512), dtype=np.uint16)
        mx[:, 100:400, 100:400] = 2000
        _write_ome(mx_path, mx, "CYX")
        _write_metadata_csv(csv_path, ["CD31", "Ki67", "CD45", "PCNA"])

        for flag in ["--register", "--no-register"]:
            out = tmp_path / flag.lstrip("-")
            cmd = [
                sys.executable,
                "-m",
                "stages.patchify",
                "--he-image",
                str(he_path),
                "--multiplex-image",
                str(mx_path),
                "--metadata-csv",
                str(csv_path),
                "--out",
                str(out),
                "--patch-size",
                "256",
                "--stride",
                "256",
                "--tissue-min",
                "0.05",
                "--channels",
                "CD31",
                "Ki67",
                "CD45",
                "PCNA",
                flag,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT
            )
            assert result.returncode == 0, f"Failed with {flag}:\n{result.stderr}"
            data = json.loads((out / "index.json").read_text())
            M = np.array(data["warp_matrix"])
            assert M.shape == (2, 3), f"warp_matrix shape {M.shape} != (2,3) for {flag}"


# ===========================================================================
# build_mx_tissue_mask
# ===========================================================================


class TestBuildMXTissueMask:
    """Tests for the MX DNA-channel tissue mask used as ECC input."""

    def _make_zarr(self, arr: np.ndarray) -> zarr.Array:
        return zarr.array(arr)

    def test_output_shape_matches_downsample(self):
        """Returns bool mask of shape (mx_h // ds, mx_w // ds)."""
        # CYX: 4 channels, 64 rows, 128 cols
        arr = np.zeros((4, 64, 128), dtype=np.uint16)
        arr[0, 16:48, 32:96] = 5000  # DNA channel signal
        store = self._make_zarr(arr)
        mask = m.build_mx_tissue_mask(store, "CYX", mx_h=64, mx_w=128, ds=8)
        assert mask.shape == (64 // 8, 128 // 8)  # (8, 16)
        assert mask.dtype == bool

    def test_zero_dna_channel_returns_all_false(self):
        """All-zero DNA channel → returns all-False mask (no tissue)."""
        arr = np.zeros((4, 64, 64), dtype=np.uint16)
        store = self._make_zarr(arr)
        mask = m.build_mx_tissue_mask(store, "CYX", mx_h=64, mx_w=64, ds=8)
        assert not mask.any(), "Expected no tissue for all-zero DNA channel"

    def test_nonzero_dna_channel_detects_signal(self):
        """Bright DNA signal (with dark background) produces True pixels in the mask."""
        arr = np.zeros((4, 64, 64), dtype=np.uint16)
        # Set a bright region in ch0 — Otsu needs contrast (dark bg vs bright signal)
        arr[0, 16:48, 16:48] = 5000
        store = self._make_zarr(arr)
        mask = m.build_mx_tissue_mask(store, "CYX", mx_h=64, mx_w=64, ds=8)
        # Some pixels should be True (Otsu separates dark bg from bright signal)
        assert mask.any(), "Expected tissue detection in bright DNA channel"

    def test_uses_only_channel_zero(self):
        """Only channel 0 (DNA) matters; other channels don't affect the mask."""
        arr_dna = np.zeros((4, 64, 64), dtype=np.uint16)
        arr_dna[0, 16:48, 16:48] = 3000  # ch0 has signal vs dark background
        arr_nodna = np.zeros((4, 64, 64), dtype=np.uint16)
        arr_nodna[1:, 16:48, 16:48] = 3000  # only non-ch0 channels bright; ch0 zero

        mask_dna = m.build_mx_tissue_mask(zarr.array(arr_dna), "CYX", 64, 64, 8)
        mask_nodna = m.build_mx_tissue_mask(zarr.array(arr_nodna), "CYX", 64, 64, 8)

        assert (
            mask_dna.any()
        ), "Bright ch0 with dark background should produce tissue mask"
        assert not mask_nodna.any(), "All-zero ch0 should produce no-tissue mask"

    def test_cyx_vs_yxc_gives_same_shape(self, tmp_path):
        """CYX and YXC layouts produce the same output shape."""
        h, w = 64, 128
        signal = np.zeros((4, h, w), dtype=np.uint16)
        signal[0, 16:48, 32:96] = 4000

        # CYX via zarr directly
        store_cyx = zarr.array(signal)
        mask_cyx = m.build_mx_tissue_mask(store_cyx, "CYX", mx_h=h, mx_w=w, ds=8)

        # YXC via tifffile
        p = tmp_path / "yxc.ome.tif"
        tifffile.imwrite(
            str(p), np.moveaxis(signal, 0, -1), ome=True, metadata={"axes": "YXC"}
        )
        tif = tifffile.TiffFile(str(p))
        store_yxc = m.open_zarr_store(tif)
        mask_yxc = m.build_mx_tissue_mask(store_yxc, "YXC", mx_h=h, mx_w=w, ds=8)

        assert mask_cyx.shape == mask_yxc.shape == (h // 8, w // 8)


# ===========================================================================
# get_tissue_patches
# ===========================================================================


class TestGetTissuePatches:
    """Tests for tissue-threshold coordinate selection."""

    def test_all_tissue_returns_all_patches(self):
        """Full-tissue mask with tissue_min=0.0 returns every valid patch position."""
        mask = np.ones((8, 8), dtype=bool)
        # img: 512×512, patch=256, stride=256, ds=64 → 2×2 grid = 4 patches
        coords = m.get_tissue_patches(
            mask,
            img_w=512,
            img_h=512,
            patch_size=256,
            stride=256,
            tissue_min=0.0,
            downsample=64,
        )
        assert len(coords) == 4
        assert (0, 0) in coords
        assert (256, 0) in coords
        assert (0, 256) in coords
        assert (256, 256) in coords

    def test_no_tissue_returns_empty(self):
        """All-background mask returns empty list for any threshold > 0."""
        mask = np.zeros((8, 8), dtype=bool)
        coords = m.get_tissue_patches(
            mask,
            img_w=512,
            img_h=512,
            patch_size=256,
            stride=256,
            tissue_min=0.1,
            downsample=64,
        )
        assert coords == []

    def test_threshold_zero_includes_background(self):
        """tissue_min=0.0 includes even background patches."""
        mask = np.zeros((8, 8), dtype=bool)
        coords = m.get_tissue_patches(
            mask,
            img_w=512,
            img_h=512,
            patch_size=256,
            stride=256,
            tissue_min=0.0,
            downsample=64,
        )
        assert len(coords) == 4

    def test_only_top_left_patch_has_tissue(self):
        """Only top-left region of mask is True → only (0,0) patch passes filter."""
        mask = np.zeros((8, 8), dtype=bool)
        # Top-left 4×4 is tissue (corresponds to top-left patch in 512×512 at ds=64)
        mask[:4, :4] = True
        coords = m.get_tissue_patches(
            mask,
            img_w=512,
            img_h=512,
            patch_size=256,
            stride=256,
            tissue_min=0.5,
            downsample=64,
        )
        assert (0, 0) in coords
        assert (256, 0) not in coords
        assert (0, 256) not in coords

    def test_patches_excluded_when_exceeding_image_boundary(self):
        """Patches whose right/bottom edge goes past image boundary are excluded."""
        mask = np.ones((8, 8), dtype=bool)
        # img_w=400, img_h=400, patch=256: only (0,0) fits; (256,0) would end at 512 > 400
        coords = m.get_tissue_patches(
            mask,
            img_w=400,
            img_h=400,
            patch_size=256,
            stride=256,
            tissue_min=0.0,
            downsample=64,
        )
        assert len(coords) == 1
        assert coords[0] == (0, 0)

    def test_coords_are_x0_y0_tuples(self):
        """Returned coordinates are (x0, y0) pixel positions, not (i, j) indices."""
        mask = np.ones((4, 4), dtype=bool)
        coords = m.get_tissue_patches(
            mask,
            img_w=256,
            img_h=256,
            patch_size=256,
            stride=256,
            tissue_min=0.0,
            downsample=64,
        )
        assert len(coords) == 1
        x0, y0 = coords[0]
        assert x0 == 0 and y0 == 0

    def test_stride_smaller_than_patch_gives_overlapping_patches(self):
        """stride < patch_size produces overlapping patches — all within bounds."""
        mask = np.ones((16, 16), dtype=bool)
        # img=512×512, patch=256, stride=128: should get (0,0), (128,0), (256,0),
        #                                                   (0,128), ..., (256,256)
        coords = m.get_tissue_patches(
            mask,
            img_w=512,
            img_h=512,
            patch_size=256,
            stride=128,
            tissue_min=0.0,
            downsample=32,
        )
        # x0 in {0, 128, 256}, y0 in {0, 128, 256} → 3×3 = 9 patches
        assert len(coords) == 9
        assert all(x0 + 256 <= 512 and y0 + 256 <= 512 for x0, y0 in coords)

    def test_high_threshold_excludes_sparse_tissue(self):
        """tissue_min=1.0 requires 100% tissue coverage, so sparse mask fails.

        At img_w/h=512, patch=256, stride=256, ds=64 the (0,0) patch queries
        mask[0:4, 0:4] (4×4 = 16 cells).  Setting only the top-left 2×4 block
        gives mean = 8/16 = 0.5 < 1.0, so no patch should pass.
        """
        mask = np.zeros((8, 8), dtype=bool)
        # Top 2 rows of the 4×4 patch region → mean = 8/16 = 0.5
        mask[:2, :4] = True
        coords = m.get_tissue_patches(
            mask,
            img_w=512,
            img_h=512,
            patch_size=256,
            stride=256,
            tissue_min=1.0,
            downsample=64,
        )
        assert coords == []


# ===========================================================================
# read_he_patch
# ===========================================================================


class TestReadHEPatch:
    """Tests for windowed H&E patch reading with boundary handling."""

    def _make_cyx_store(self, arr: np.ndarray) -> tuple:
        """Return (zarr_array, axes='CYX', img_w, img_h)."""
        store = zarr.array(arr)
        _, img_h, img_w = arr.shape
        return store, "CYX", img_w, img_h

    def test_cyx_returns_correct_shape_and_dtype(self):
        """CYX layout returns (size, size, 3) uint8."""
        arr = np.full((3, 256, 256), 128, dtype=np.uint8)
        store, axes, img_w, img_h = self._make_cyx_store(arr)
        patch = m.read_he_patch(store, axes, img_w, img_h, y0=0, x0=0, size=256)
        assert patch.shape == (256, 256, 3)
        assert patch.dtype == np.uint8

    def test_yxc_layout(self, tmp_path):
        """YXC axis layout is correctly transposed to (H, W, C) output."""
        h, w = 256, 256
        arr = np.full((h, w, 3), 100, dtype=np.uint8)
        p = tmp_path / "yxc.ome.tif"
        tifffile.imwrite(str(p), arr, ome=True, metadata={"axes": "YXC"})
        tif = tifffile.TiffFile(str(p))
        store = m.open_zarr_store(tif)
        _, img_h, img_w = h, h, w  # swap
        img_w2, img_h2, axes = m.get_image_dims(tif)
        patch = m.read_he_patch(store, axes, img_w2, img_h2, y0=0, x0=0, size=128)
        assert patch.shape == (128, 128, 3)
        assert patch.dtype == np.uint8

    def test_patch_at_image_boundary_is_zero_padded(self):
        """Patch extending past image edge is zero-padded to the requested size."""
        h, w = 200, 200
        arr = np.full((3, h, w), 255, dtype=np.uint8)
        store, axes, img_w, img_h = self._make_cyx_store(arr)
        # x0=150, size=128 → only 50px available (150..200), rest zero-padded
        patch = m.read_he_patch(store, axes, img_w, img_h, y0=0, x0=150, size=128)
        assert patch.shape == (128, 128, 3)
        # The right portion (where image ran out) should be zero
        assert (
            patch[:, 50:, :].sum() == 0
        ), "Expected zero-padding in out-of-bounds region"
        # The left portion (valid data) should be non-zero
        assert patch[:, :50, :].sum() > 0, "Expected valid pixel data in-bounds region"

    def test_fully_out_of_bounds_returns_zeros(self):
        """Patch starting completely outside image returns all zeros."""
        arr = np.full((3, 256, 256), 200, dtype=np.uint8)
        store, axes, img_w, img_h = self._make_cyx_store(arr)
        patch = m.read_he_patch(store, axes, img_w, img_h, y0=300, x0=300, size=128)
        assert patch.shape == (128, 128, 3)
        assert patch.sum() == 0

    def test_uint16_input_normalized_to_uint8(self):
        """uint16 input is percentile-normalized to uint8 (no data corruption)."""
        arr = np.zeros((3, 128, 128), dtype=np.uint16)
        arr[:, 20:100, 20:100] = 50000  # bright region
        store = zarr.array(arr)
        patch = m.read_he_patch(
            store, "CYX", img_w=128, img_h=128, y0=0, x0=0, size=128
        )
        assert patch.dtype == np.uint8
        # Bright region should map to high uint8 values
        assert patch[20:100, 20:100].max() > 200

    def test_grayscale_single_channel_expanded_to_rgb(self):
        """Single-channel (1, H, W) input is tiled to 3-channel output."""
        arr = np.full((1, 128, 128), 120, dtype=np.uint8)
        store, axes, img_w, img_h = self._make_cyx_store(arr)
        patch = m.read_he_patch(store, axes, img_w, img_h, y0=0, x0=0, size=128)
        assert patch.shape == (128, 128, 3)
        # All 3 channels should be identical
        np.testing.assert_array_equal(patch[:, :, 0], patch[:, :, 1])
        np.testing.assert_array_equal(patch[:, :, 0], patch[:, :, 2])

    def test_more_than_3_channels_truncated_to_rgb(self):
        """Input with 5 channels is truncated to first 3."""
        arr = np.full((5, 128, 128), 100, dtype=np.uint8)
        arr[3:] = 200  # channels 3,4 are brighter but should be dropped
        store, axes, img_w, img_h = self._make_cyx_store(arr)
        patch = m.read_he_patch(store, axes, img_w, img_h, y0=0, x0=0, size=128)
        assert patch.shape == (128, 128, 3)

    def test_small_patch_from_center_of_large_image(self):
        """Reading a small patch from the center of a larger image returns correct values."""
        h, w = 512, 512
        arr = np.zeros((3, h, w), dtype=np.uint8)
        # Place a bright marker at (200:210, 200:210)
        arr[:, 200:210, 200:210] = 255
        store, axes, img_w, img_h = self._make_cyx_store(arr)
        # Read 64×64 patch centered on the bright marker
        patch = m.read_he_patch(store, axes, img_w, img_h, y0=192, x0=192, size=64)
        assert patch.shape == (64, 64, 3)
        # The marker at relative position (8:18, 8:18) should be bright
        assert patch[8:18, 8:18].max() > 200


# ===========================================================================
# read_multiplex_patch
# ===========================================================================


class TestReadMultiplexPatch:
    """Tests for multiplex windowed patch reading."""

    def test_basic_channel_selection(self):
        """Selected channels are extracted; output shape is (C_sel, size_y, size_x)."""
        arr = np.zeros((6, 128, 128), dtype=np.uint16)
        for c in range(6):
            arr[c] = c * 100  # each channel has a distinct constant value
        store = zarr.array(arr)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=128,
            img_h=128,
            y0=0,
            x0=0,
            size_y=64,
            size_x=64,
            channel_indices=[0, 2, 4],
        )
        assert patch.shape == (3, 64, 64)
        assert patch.dtype == np.uint16
        # Channel 0 maps to arr[0] = 0, channel 2 → 200, channel 4 → 400
        assert patch[0].max() == 0
        assert patch[1].max() == 200
        assert patch[2].max() == 400

    def test_zero_padding_at_right_boundary(self):
        """Patch extending past right image edge is zero-padded."""
        arr = np.full((4, 128, 128), 1000, dtype=np.uint16)
        store = zarr.array(arr)
        # x0=110, size_x=64: only 18 valid columns (110..128)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=128,
            img_h=128,
            y0=0,
            x0=110,
            size_y=32,
            size_x=64,
            channel_indices=[0, 1],
        )
        assert patch.shape == (2, 32, 64)
        # Valid region (first 18 cols) should be 1000
        assert patch[:, :, :18].min() == 1000
        # Padded region (cols 18..64) should be zero
        assert patch[:, :, 18:].sum() == 0

    def test_zero_padding_at_bottom_boundary(self):
        """Patch extending past bottom image edge is zero-padded."""
        arr = np.full((4, 128, 128), 500, dtype=np.uint16)
        store = zarr.array(arr)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=128,
            img_h=128,
            y0=115,
            x0=0,
            size_y=64,
            size_x=32,
            channel_indices=[0],
        )
        assert patch.shape == (1, 64, 32)
        assert patch[:, :13, :].min() == 500
        assert patch[:, 13:, :].sum() == 0

    def test_fully_out_of_bounds_returns_zeros(self):
        """Patch outside image returns all zeros."""
        arr = np.ones((4, 128, 128), dtype=np.uint16) * 999
        store = zarr.array(arr)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=128,
            img_h=128,
            y0=200,
            x0=200,
            size_y=64,
            size_x=64,
            channel_indices=[0, 1],
        )
        assert patch.shape == (2, 64, 64)
        assert patch.sum() == 0

    def test_single_channel_selection(self):
        """Requesting a single channel returns (1, size_y, size_x) output."""
        arr = np.zeros((4, 64, 64), dtype=np.uint16)
        arr[2] = 777
        store = zarr.array(arr)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=64,
            img_h=64,
            y0=0,
            x0=0,
            size_y=32,
            size_x=32,
            channel_indices=[2],
        )
        assert patch.shape == (1, 32, 32)
        assert patch[0].max() == 777

    def test_yxc_layout_channel_selection(self):
        """YXC layout (channel-last zarr): channel selection returns correct values.

        Uses a zarr array directly with shape (H, W, C) to avoid tifffile's
        internal CYX reordering when writing multi-channel images tagged as YXC.
        """
        h, w, c = 128, 128, 4
        arr = np.zeros((h, w, c), dtype=np.uint16)
        arr[:, :, 1] = 300  # channel 1
        arr[:, :, 3] = 700  # channel 3
        store = zarr.array(arr)  # shape (128, 128, 4) — true YXC layout
        patch = m.read_multiplex_patch(
            store,
            "YXC",
            img_w=w,
            img_h=h,
            y0=0,
            x0=0,
            size_y=64,
            size_x=64,
            channel_indices=[1, 3],
        )
        assert patch.shape == (2, 64, 64)
        assert patch[0].max() == 300
        assert patch[1].max() == 700

    def test_rectangular_patch_size(self):
        """size_y != size_x is supported; output shape matches requested dimensions."""
        arr = np.zeros((4, 256, 512), dtype=np.uint16)
        store = zarr.array(arr)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=512,
            img_h=256,
            y0=0,
            x0=0,
            size_y=100,
            size_x=200,
            channel_indices=[0, 1],
        )
        assert patch.shape == (2, 100, 200)

    def test_channel_values_preserved_exactly(self):
        """Pixel values in the valid region are not modified (no normalization for uint16)."""
        arr = np.zeros((2, 64, 64), dtype=np.uint16)
        arr[0, 10:20, 10:20] = 12345
        arr[1, 30:40, 30:40] = 65535
        store = zarr.array(arr)
        patch = m.read_multiplex_patch(
            store,
            "CYX",
            img_w=64,
            img_h=64,
            y0=0,
            x0=0,
            size_y=64,
            size_x=64,
            channel_indices=[0, 1],
        )
        assert patch[0, 10:20, 10:20].min() == 12345
        assert patch[1, 30:40, 30:40].max() == 65535


# ===========================================================================
# End-to-end: H&E coord → MX coord mapping through warp_matrix
# ===========================================================================


class TestEndToEndCoordMapping:
    """Integration test: warp_matrix from index.json correctly maps H&E → MX coords."""

    def test_warp_matrix_maps_patch_into_mx_bounds(self, tmp_path):
        """
        Run patchify on same-size images, then verify that applying the stored
        warp_matrix to each patch (x0, y0) produces MX coordinates inside MX bounds.
        """
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        h = w = 512
        he = np.full((3, h, w), 230, dtype=np.uint8)
        he[:, 100:400, 100:400] = np.array([180, 60, 120])[:, None, None]
        _write_ome(he_path, he, "CYX")

        mx = np.zeros((4, h, w), dtype=np.uint16)
        mx[:, 100:400, 100:400] = 3000
        _write_ome(mx_path, mx, "CYX")
        _write_metadata_csv(csv_path, ["CD31", "Ki67", "CD45", "PCNA"])

        cmd = [
            sys.executable,
            "-m",
            "stages.patchify",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--metadata-csv",
            str(csv_path),
            "--out",
            str(out_dir),
            "--patch-size",
            "256",
            "--stride",
            "256",
            "--tissue-min",
            "0.05",
            "--channels",
            "CD31",
            "Ki67",
            "CD45",
            "PCNA",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
        assert result.returncode == 0, result.stderr

        data = json.loads((out_dir / "index.json").read_text())
        M = np.array(data["warp_matrix"], dtype=np.float32)
        mx_w = data["img_w"]
        mx_h = data["img_h"]
        patch_size = data["patch_size"]

        for patch in data["patches"]:
            x0, y0 = patch["x0"], patch["y0"]
            x_mx, y_mx = m.transform_he_to_mx_point(M, x0, y0)
            # The mapped corner should be within or near MX image bounds
            assert x_mx >= -patch_size, f"x_mx={x_mx} far outside left bound"
            assert y_mx >= -patch_size, f"y_mx={y_mx} far outside top bound"
            assert x_mx < mx_w + patch_size, f"x_mx={x_mx} far outside right bound"
            assert y_mx < mx_h + patch_size, f"y_mx={y_mx} far outside bottom bound"

    def test_scale_only_mapping_matches_mpp_ratio(self, tmp_path, monkeypatch):
        """
        With --no-register, M_full = [[s,0,0],[0,s,0]] where s = he_mpp/mx_mpp.
        Applying transform_he_to_mx_point gives x_mx = round(x0 * s).
        """
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        he = np.full((3, 512, 512), 230, dtype=np.uint8)
        he[:, 100:400, 100:400] = np.array([180, 60, 120])[:, None, None]
        _write_ome(he_path, he, "CYX")
        # MX half the resolution (256×256 same physical area)
        mx = np.zeros((4, 256, 256), dtype=np.uint16)
        mx[:, 50:200, 50:200] = 3000
        _write_ome(mx_path, mx, "CYX")
        _write_metadata_csv(csv_path, ["CD31", "Ki67", "CD45", "PCNA"])

        # Mock mpp so scale = 0.5 (HE 0.5 µm/px, MX 1.0 µm/px)
        import stages.patchify as pm

        calls = []

        def mock_mpp(tif):
            calls.append(1)
            return (0.5, 0.5) if len(calls) == 1 else (1.0, 1.0)

        monkeypatch.setattr(pm, "get_ome_mpp", mock_mpp)

        sys.argv = [
            "stages.patchify",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--metadata-csv",
            str(csv_path),
            "--out",
            str(out_dir),
            "--patch-size",
            "256",
            "--stride",
            "256",
            "--tissue-min",
            "0.05",
            "--channels",
            "CD31",
            "Ki67",
            "CD45",
            "PCNA",
            "--no-register",
        ]
        pm.main()

        data = json.loads((out_dir / "index.json").read_text())
        M = np.array(data["warp_matrix"], dtype=np.float32)
        scale = data["scale_he_to_mx"]

        assert abs(scale - 0.5) < 0.01, f"Expected scale=0.5, got {scale}"
        assert abs(M[0, 0] - scale) < 0.01
        assert abs(M[1, 1] - scale) < 0.01

        for patch in data["patches"]:
            x0, y0 = patch["x0"], patch["y0"]
            x_mx, y_mx = m.transform_he_to_mx_point(M, x0, y0)
            expected_x = int(round(x0 * scale))
            expected_y = int(round(y0 * scale))
            assert x_mx == expected_x, f"x_mx={x_mx}, expected {expected_x}"
            assert y_mx == expected_y, f"y_mx={y_mx}, expected {expected_y}"


# ===========================================================================
# read_multiplex_patch_affine
# ===========================================================================


class TestReadMultiplexPatchAffine:
    """Tests for affine-accurate multiplex patch extraction."""

    def test_identity_transform_matches_direct_crop(self):
        """Identity warp should match direct MX crop at the same coordinates."""
        arr = np.zeros((2, 64, 64), dtype=np.uint16)
        yy, xx = np.indices((64, 64))
        arr[0] = (yy * 100 + xx).astype(np.uint16)
        arr[1] = (yy * 3 + xx * 2).astype(np.uint16)
        store = zarr.array(arr)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        patch, inside = m.read_multiplex_patch_affine(
            store,
            "CYX",
            img_w=64,
            img_h=64,
            he_x0=10,
            he_y0=12,
            patch_size=16,
            m_full=M,
            channel_indices=[0, 1],
        )

        expected = arr[[0, 1], 12:28, 10:26]
        assert inside is True
        np.testing.assert_array_equal(patch, expected)

    def test_translation_transform_matches_shifted_crop(self):
        """Pure translation in M_full should shift the extracted MX crop."""
        arr = np.zeros((1, 80, 80), dtype=np.uint16)
        yy, xx = np.indices((80, 80))
        arr[0] = (yy * 200 + xx).astype(np.uint16)
        store = zarr.array(arr)
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0]], dtype=np.float32)

        patch, inside = m.read_multiplex_patch_affine(
            store,
            "CYX",
            img_w=80,
            img_h=80,
            he_x0=10,
            he_y0=12,
            patch_size=16,
            m_full=M,
            channel_indices=[0],
        )

        expected = arr[[0], 19:35, 15:31]
        assert inside is True
        np.testing.assert_array_equal(patch, expected)

    def test_reports_out_of_bounds_when_affine_footprint_exits_image(self):
        """If mapped patch footprint is outside MX bounds, inside=False and output is padded."""
        arr = np.full((1, 64, 64), 999, dtype=np.uint16)
        store = zarr.array(arr)
        M = np.array([[1.0, 0.0, -30.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        patch, inside = m.read_multiplex_patch_affine(
            store,
            "CYX",
            img_w=64,
            img_h=64,
            he_x0=4,
            he_y0=8,
            patch_size=16,
            m_full=M,
            channel_indices=[0],
        )

        assert inside is False
        assert patch.shape == (1, 16, 16)
        assert patch.sum() == 0

    def test_overlap_fraction_affine_reports_partial_coverage(self):
        """Overlap fraction reports partial coverage for translated affine footprints."""
        M = np.array([[1.0, 0.0, -8.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        frac = m.multiplex_patch_overlap_fraction_affine(
            he_x0=0,
            he_y0=0,
            patch_size=16,
            m_full=M,
            mx_w=64,
            mx_h=64,
        )
        # 8px shift on a 16px patch -> about 50% overlap.
        assert 0.45 <= frac <= 0.55, f"Expected ~0.5 overlap, got {frac:.4f}"


class TestReadMultiplexPatchAffineDeform:
    """Tests for deformable-refined multiplex patch extraction."""

    def test_zero_flow_matches_affine_reader(self):
        arr = np.zeros((1, 64, 64), dtype=np.uint16)
        yy, xx = np.indices((64, 64))
        arr[0] = (yy * 100 + xx).astype(np.uint16)
        store = zarr.array(arr)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        flow_x = np.zeros((64, 64), dtype=np.float32)
        flow_y = np.zeros((64, 64), dtype=np.float32)

        patch_a, inside_a = m.read_multiplex_patch_affine(
            store,
            "CYX",
            img_w=64,
            img_h=64,
            he_x0=10,
            he_y0=12,
            patch_size=16,
            m_full=M,
            channel_indices=[0],
        )
        patch_d, inside_d = m.read_multiplex_patch_affine_deform(
            store,
            "CYX",
            img_w=64,
            img_h=64,
            he_x0=10,
            he_y0=12,
            patch_size=16,
            m_full=M,
            channel_indices=[0],
            flow_dx_ov=flow_x,
            flow_dy_ov=flow_y,
            he_full_w=64,
            he_full_h=64,
        )

        assert inside_a == inside_d
        np.testing.assert_array_equal(patch_d, patch_a)

    def test_constant_flow_applies_expected_shift(self):
        arr = np.zeros((1, 80, 80), dtype=np.uint16)
        yy, xx = np.indices((80, 80))
        arr[0] = (yy * 200 + xx).astype(np.uint16)
        store = zarr.array(arr)
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        # Flow is defined as warped-MX -> HE; inverse application samples from (x-flow, y-flow)
        flow_x = np.full((80, 80), 3.0, dtype=np.float32)
        flow_y = np.full((80, 80), 2.0, dtype=np.float32)
        patch, inside = m.read_multiplex_patch_affine_deform(
            store,
            "CYX",
            img_w=80,
            img_h=80,
            he_x0=10,
            he_y0=12,
            patch_size=16,
            m_full=M,
            channel_indices=[0],
            flow_dx_ov=flow_x,
            flow_dy_ov=flow_y,
            he_full_w=80,
            he_full_h=80,
        )

        expected = arr[[0], 10:26, 7:23]
        assert inside is True
        np.testing.assert_array_equal(patch, expected)


def test_cli_min_multiplex_overlap_allows_partial_save(tmp_path, monkeypatch):
    """CLI saves partially overlapping multiplex patches when threshold is relaxed."""
    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mx.ome.tif"
    csv_path = tmp_path / "meta.csv"
    out_strict = tmp_path / "out_strict"
    out_partial = tmp_path / "out_partial"

    he = np.full((3, 256, 256), 180, dtype=np.uint8)
    mx = np.full((1, 256, 256), 1000, dtype=np.uint16)
    _write_ome(he_path, he, "CYX")
    _write_ome(mx_path, mx, "CYX")
    _write_metadata_csv(csv_path, ["DNA"])

    monkeypatch.setattr(
        m,
        "register_he_mx_affine",
        lambda *args, **kwargs: np.array(
            [[1.0, 0.0, -32.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        m, "decide_registration_path", lambda *args, **kwargs: m.PASS_AFFINE
    )

    base = [
        "patchify.py",
        "--he-image",
        str(he_path),
        "--multiplex-image",
        str(mx_path),
        "--metadata-csv",
        str(csv_path),
        "--patch-size",
        "256",
        "--stride",
        "256",
        "--tissue-min",
        "0.0",
        "--channels",
        "DNA",
        "--overview-downsample",
        "1",
    ]

    sys.argv = [*base, "--out", str(out_strict), "--min-multiplex-overlap", "1.0"]
    m.main()
    strict_index = json.loads((out_strict / "index.json").read_text())
    assert len(strict_index["patches"]) == 1
    strict_patch = strict_index["patches"][0]
    assert strict_patch["has_multiplex"] is False
    assert 0.0 < strict_patch["multiplex_overlap_fraction"] < 1.0
    assert not (out_strict / "multiplex" / "0_0.npy").exists()

    sys.argv = [*base, "--out", str(out_partial), "--min-multiplex-overlap", "0.85"]
    m.main()
    partial_index = json.loads((out_partial / "index.json").read_text())
    assert len(partial_index["patches"]) == 1
    partial_patch = partial_index["patches"][0]
    assert partial_patch["has_multiplex"] is True
    assert 0.0 < partial_patch["multiplex_overlap_fraction"] < 1.0
    assert (out_partial / "multiplex" / "0_0.npy").exists()


def test_force_deformable_cli_sets_registration_mode_deformable(tmp_path, monkeypatch):
    """--force-deformable should select deformable mode when local QC requests it."""
    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mx.ome.tif"
    csv_path = tmp_path / "meta.csv"
    out_dir = tmp_path / "processed"

    he = np.full((3, 256, 256), 180, dtype=np.uint8)
    mx = np.full((1, 256, 256), 1000, dtype=np.uint16)
    _write_ome(he_path, he, "CYX")
    _write_ome(mx_path, mx, "CYX")
    _write_metadata_csv(csv_path, ["DNA"])

    monkeypatch.setattr(
        m,
        "decide_registration_path",
        lambda *args, **kwargs: m.FAIL_LOCAL_NEEDS_DEFORMABLE,
    )

    def _fake_deform_state(he_mask, *args, **kwargs):
        h, w = he_mask.shape
        return {
            "flow_dx_ov": np.zeros((h, w), dtype=np.float32),
            "flow_dy_ov": np.zeros((h, w), dtype=np.float32),
            "iou_affine": 0.90,
            "iou_deformable": 0.90,
            "mean_disp_ov": 0.0,
            "max_disp_ov": 0.0,
        }

    monkeypatch.setattr(m, "estimate_deformable_field", _fake_deform_state)
    monkeypatch.setattr(
        m,
        "compute_deformable_patch_qc_metrics",
        lambda *args, **kwargs: {
            "sample_count": 10.0,
            "improved_fraction": 0.0,
            "median_gain": 0.0,
            "inside_fraction_pass_rate": 1.0,
            "mean_gain": 0.0,
        },
    )

    sys.argv = [
        "patchify.py",
        "--he-image",
        str(he_path),
        "--multiplex-image",
        str(mx_path),
        "--metadata-csv",
        str(csv_path),
        "--out",
        str(out_dir),
        "--channels",
        "DNA",
        "--patch-size",
        "128",
        "--stride",
        "128",
        "--overview-downsample",
        "1",
        "--force-deformable",
    ]
    m.main()

    data = json.loads((out_dir / "index.json").read_text())
    assert data["registration_mode"] == "deformable"
    final_t = json.loads(
        (out_dir / "registration" / "final_transform.json").read_text()
    )
    assert final_t["mode"] == "deformable"
    assert (out_dir / "registration" / "deform_field.npz").exists()


class TestPatchifyDebugLimit:
    """Regression test for the current debug-only extraction cap."""

    def test_patchify_limits_to_first_10_patches(self, tmp_path):
        """Pipeline currently caps extraction to coords[:10] for debug runs."""
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        he = np.zeros((3, 1024, 1024), dtype=np.uint8)
        he[0] = 180
        he[1] = 40
        he[2] = 120
        _write_ome(he_path, he, "CYX")

        mx = np.full((1, 1024, 1024), 2000, dtype=np.uint16)
        _write_ome(mx_path, mx, "CYX")
        _write_metadata_csv(csv_path, ["DNA"])

        cmd = [
            sys.executable,
            "-m",
            "stages.patchify",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--metadata-csv",
            str(csv_path),
            "--out",
            str(out_dir),
            "--patch-size",
            "256",
            "--stride",
            "256",
            "--tissue-min",
            "0.0",
            "--channels",
            "DNA",
            "--no-register",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
        assert result.returncode == 0, result.stderr

        data = json.loads((out_dir / "index.json").read_text())
        assert len(data["patches"]) == 10


class TestChannelDriftQC:
    """Unit tests for CyCIF channel drift metrics and gate thresholds."""

    def test_single_channel_metrics_are_zero(self):
        ch0 = np.zeros((64, 64), dtype=np.float32)
        ch0[16:48, 20:44] = 1.0
        chw = np.stack([ch0], axis=0)

        metrics = m.compute_channel_drift_metrics(chw, ref_channel=0)

        assert metrics["n_channels"] == 1
        assert metrics["median_drift_px"] == pytest.approx(0.0)
        assert metrics["max_drift_px"] == pytest.approx(0.0)
        assert m.channel_drift_passes(metrics)

    def test_phase_correlation_detects_known_shift(self):
        ref = np.zeros((96, 96), dtype=np.float32)
        ref[30:66, 35:60] = 1.0
        dx, dy = 3.0, -2.0
        moving = cv2.warpAffine(
            ref,
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
            (96, 96),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        chw = np.stack([ref, moving], axis=0)

        metrics = m.compute_channel_drift_metrics(chw, ref_channel=0)
        drift = metrics["per_channel"][0]["drift_px"]

        assert drift == pytest.approx((dx * dx + dy * dy) ** 0.5, abs=0.75)
        assert metrics["max_drift_px"] == pytest.approx(drift)

    def test_gate_fails_when_drift_exceeds_thresholds(self):
        metrics = {
            "median_drift_px": 2.0,
            "max_drift_px": 5.1,
        }
        assert not m.channel_drift_passes(metrics, median_thresh=1.5, max_thresh=4.0)


class TestGlobalRegistrationQC:
    """Unit tests for global registration QC metrics and thresholds."""

    def test_identity_masks_pass_global_qc(self):
        he = np.zeros((64, 64), dtype=bool)
        he[16:48, 14:50] = True
        mx = he.copy()
        m_full = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        metrics = m.compute_global_qc_metrics(
            he_mask=he,
            mx_mask=mx,
            m_full=m_full,
            he_h=640,
            he_w=640,
            mx_h=640,
            mx_w=640,
        )

        assert metrics["mask_iou"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["centroid_offset_pct"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["scale_error_pct"] == pytest.approx(0.0, abs=1e-5)
        assert m.global_qc_passes(metrics)

    def test_large_translation_fails_global_qc(self):
        he = np.zeros((64, 64), dtype=bool)
        he[12:52, 10:48] = True
        mx = he.copy()
        m_full = np.array([[1.0, 0.0, 90.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        metrics = m.compute_global_qc_metrics(
            he_mask=he,
            mx_mask=mx,
            m_full=m_full,
            he_h=640,
            he_w=640,
            mx_h=640,
            mx_w=640,
        )

        assert metrics["mask_iou"] < 0.75
        assert not m.global_qc_passes(metrics)

    def test_scale_sanity_detects_mismatch(self):
        he = np.zeros((48, 48), dtype=bool)
        he[8:40, 8:40] = True
        mx = he.copy()
        # Expected H&E->MX scale is 0.5, but this matrix says 0.8
        m_full = np.array([[0.8, 0.0, 0.0], [0.0, 0.8, 0.0]], dtype=np.float32)

        metrics = m.compute_global_qc_metrics(
            he_mask=he,
            mx_mask=mx,
            m_full=m_full,
            he_h=960,
            he_w=960,
            mx_h=480,
            mx_w=480,
        )

        assert metrics["scale_error_pct"] > 10.0
        assert not m.global_qc_passes(metrics)


class TestPatchLevelQC:
    """Unit tests for patch-level local overlap quality checks."""

    def test_correct_translation_improves_local_iou(self):
        he = np.zeros((128, 128), dtype=bool)
        he[28:108, 20:100] = True
        mx = np.zeros_like(he)
        mx[28:108, 32:112] = True  # shifted +12 in x
        m_full = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        metrics = m.compute_patch_qc_metrics(
            he_mask=he,
            mx_mask=mx,
            m_full=m_full,
            he_h=128,
            he_w=128,
            mx_h=128,
            mx_w=128,
            sample_count=60,
            patch_size_ov=24,
            seed=0,
        )

        assert metrics["improved_fraction"] >= 0.8
        assert metrics["median_gain"] > 0.0
        assert m.patch_qc_passes(metrics)

    def test_no_gain_fails_patch_gate(self):
        he = np.zeros((96, 96), dtype=bool)
        he[20:76, 18:74] = True
        mx = np.zeros_like(he)
        mx[20:76, 30:86] = True  # shifted +12 in x
        m_full = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        metrics = m.compute_patch_qc_metrics(
            he_mask=he,
            mx_mask=mx,
            m_full=m_full,
            he_h=96,
            he_w=96,
            mx_h=96,
            mx_w=96,
            sample_count=30,
            patch_size_ov=20,
            seed=1,
        )

        assert metrics["improved_fraction"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["median_gain"] == pytest.approx(0.0, abs=1e-6)
        assert not m.patch_qc_passes(metrics)


class TestRegistrationDecisionGate:
    """Decision routing between pass/global-fail/local-fail paths."""

    def test_global_fail_routes_to_landmarks(self):
        global_metrics = {
            "mask_iou": 0.4,
            "centroid_offset_pct": 20.0,
            "scale_error_pct": 5.0,
        }
        patch_metrics = {
            "improved_fraction": 1.0,
            "median_gain": 0.2,
            "inside_fraction_pass_rate": 1.0,
        }
        decision = m.decide_registration_path(global_metrics, patch_metrics)
        assert decision == "FAIL_GLOBAL_NEEDS_LANDMARKS"

    def test_local_fail_routes_to_deformable(self):
        global_metrics = {
            "mask_iou": 0.92,
            "centroid_offset_pct": 0.8,
            "scale_error_pct": 1.2,
        }
        patch_metrics = {
            "improved_fraction": 0.25,
            "median_gain": 0.0,
            "inside_fraction_pass_rate": 0.96,
        }
        decision = m.decide_registration_path(global_metrics, patch_metrics)
        assert decision == "FAIL_LOCAL_NEEDS_DEFORMABLE"

    def test_all_pass_routes_to_affine_accept(self):
        global_metrics = {
            "mask_iou": 0.90,
            "centroid_offset_pct": 1.2,
            "scale_error_pct": 2.0,
        }
        patch_metrics = {
            "improved_fraction": 0.95,
            "median_gain": 0.03,
            "inside_fraction_pass_rate": 0.99,
        }
        decision = m.decide_registration_path(global_metrics, patch_metrics)
        assert decision == "PASS_AFFINE"


class TestRegistrationArtifacts:
    """Integration checks for registration artifacts written by patchify."""

    def test_patchify_writes_qc_artifacts(self, tmp_path):
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        he = np.full((3, 512, 512), 230, dtype=np.uint8)
        he[:, 80:420, 90:430] = np.array([180, 60, 120])[:, None, None]
        _write_ome(he_path, he, "CYX")

        mx = np.zeros((2, 512, 512), dtype=np.uint16)
        mx[0, 80:420, 100:440] = 3000  # DNA with a mild shift
        mx[1, 80:420, 100:440] = 1500
        _write_ome(mx_path, mx, "CYX")
        _write_metadata_csv(csv_path, ["DNA", "CD45"])

        cmd = [
            sys.executable,
            "-m",
            "stages.patchify",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--metadata-csv",
            str(csv_path),
            "--out",
            str(out_dir),
            "--patch-size",
            "256",
            "--stride",
            "256",
            "--tissue-min",
            "0.05",
            "--channels",
            "DNA",
            "--register",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
        assert result.returncode == 0, result.stderr

        reg_dir = out_dir / "registration"
        assert (reg_dir / "affine.json").exists()
        assert (reg_dir / "qc_metrics.json").exists()
        assert (reg_dir / "final_transform.json").exists()

        qc = json.loads((reg_dir / "qc_metrics.json").read_text())
        assert "channel_drift" in qc
        assert "global_qc" in qc
        assert "patch_qc" in qc
        assert "decision" in qc
        assert "registration_mode" in qc
        assert "deformable" in qc

        final_t = json.loads((reg_dir / "final_transform.json").read_text())
        assert final_t["mode"] in {"affine", "deformable"}
        assert "warp_matrix" in final_t

        data = json.loads((out_dir / "index.json").read_text())
        assert data["registration_mode"] in {"affine", "deformable"}
