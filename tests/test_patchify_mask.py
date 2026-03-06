"""
Tests for cell segmentation mask patch extraction in stages/patchify.py.

Covers
------
read_mask_patch  -- windowed mask reading: dtype casting, zero-padding, axes layouts
CLI --mask-image -- end-to-end: masks/ directory created, .npy shape, index.json flags
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stages.patchify as m

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ome(path: Path, arr: np.ndarray, axes: str) -> None:
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": axes})


def _write_metadata_csv(path: Path, names: list[str]) -> None:
    lines = ["Channel ID,Target Name\n"]
    for i, n in enumerate(names):
        lines.append(f"Channel:0:{i},{n}\n")
    path.write_text("".join(lines))


def _he_array(h: int = 512, w: int = 512) -> np.ndarray:
    arr = np.full((3, h, w), 230, dtype=np.uint8)
    arr[:, 100:400, 100:400] = np.array([180, 60, 120])[:, None, None]
    return arr


def _mx_array(h: int = 512, w: int = 512, c: int = 4) -> np.ndarray:
    arr = np.zeros((c, h, w), dtype=np.uint16)
    arr[:, 100:400, 100:400] = 3000
    return arr


# ===========================================================================
# read_mask_patch — unit tests
# ===========================================================================

class TestReadMaskPatch:
    """Unit tests for read_mask_patch."""

    def test_yx_layout_returns_correct_shape(self):
        """YX mask (no channel axis) returns (size, size) uint32."""
        arr = np.zeros((256, 256), dtype=np.uint32)
        arr[50:100, 50:100] = 42
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=256, img_h=256,
                                  y0=0, x0=0, size=128)
        assert patch.shape == (128, 128)
        assert patch.dtype == np.uint32
        assert patch[50:100, 50:100].min() == 42

    def test_cyx_single_channel_squeezed(self):
        """CYX mask with C=1 is squeezed to (size, size)."""
        arr = np.zeros((1, 256, 256), dtype=np.uint16)
        arr[0, 10:20, 10:20] = 7
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "CYX", img_w=256, img_h=256,
                                  y0=0, x0=0, size=128)
        assert patch.shape == (128, 128)
        assert patch.dtype == np.uint32
        assert patch[10:20, 10:20].min() == 7

    def test_uint8_label_upcast_to_uint32(self):
        """uint8 mask is safely upcast to uint32 without value loss."""
        arr = np.zeros((128, 128), dtype=np.uint8)
        arr[30:60, 30:60] = 255
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=0, x0=0, size=128)
        assert patch.dtype == np.uint32
        assert patch[30:60, 30:60].min() == 255

    def test_uint16_label_upcast_to_uint32(self):
        """uint16 mask is safely upcast to uint32 without value loss."""
        arr = np.zeros((128, 128), dtype=np.uint16)
        arr[10:20, 10:20] = 1000
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=0, x0=0, size=64)
        assert patch.dtype == np.uint32
        assert patch[10:20, 10:20].min() == 1000

    def test_float_mask_rounded_and_cast(self):
        """Float mask values are rounded before casting to uint32."""
        arr = np.zeros((128, 128), dtype=np.float32)
        arr[5:10, 5:10] = 3.7   # should round to 4
        arr[20:30, 20:30] = 1.4  # should round to 1
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=0, x0=0, size=64)
        assert patch.dtype == np.uint32
        assert patch[5:10, 5:10].min() == 4
        assert patch[20:30, 20:30].min() == 1

    def test_label_values_preserved_exactly(self):
        """Label IDs in valid region are not modified (no normalisation)."""
        arr = np.zeros((256, 256), dtype=np.uint32)
        # Use distinct, non-overlapping rows for each label
        arr[10:15, :5] = 1
        arr[20:25, :5] = 100
        arr[30:35, :5] = 50000
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=256, img_h=256,
                                  y0=0, x0=0, size=256)
        assert patch[10:15, :5].min() == 1
        assert patch[20:25, :5].min() == 100
        assert patch[30:35, :5].min() == 50000

    def test_zero_padding_at_right_boundary(self):
        """Patch extending past right edge is zero-padded."""
        arr = np.full((128, 128), 5, dtype=np.uint32)
        store = zarr.array(arr)
        # x0=110, size=64: only 18 valid columns
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=0, x0=110, size=64)
        assert patch.shape == (64, 64)
        assert patch[:, :18].min() == 5
        assert patch[:, 18:].sum() == 0

    def test_zero_padding_at_bottom_boundary(self):
        """Patch extending past bottom edge is zero-padded."""
        arr = np.full((128, 128), 3, dtype=np.uint32)
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=115, x0=0, size=64)
        assert patch.shape == (64, 64)
        assert patch[:13, :].min() == 3
        assert patch[13:, :].sum() == 0

    def test_fully_out_of_bounds_returns_zeros(self):
        """Patch completely outside image returns all-zero mask."""
        arr = np.full((128, 128), 99, dtype=np.uint32)
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=200, x0=200, size=64)
        assert patch.shape == (64, 64)
        assert patch.sum() == 0

    def test_background_zero_preserved(self):
        """Background (label 0) pixels remain 0 in the patch."""
        arr = np.zeros((128, 128), dtype=np.uint32)
        arr[30:60, 30:60] = 1  # only a small region is labelled
        store = zarr.array(arr)
        patch = m.read_mask_patch(store, "YX", img_w=128, img_h=128,
                                  y0=0, x0=0, size=128)
        assert patch[:30, :30].sum() == 0  # background region
        assert patch[30:60, 30:60].min() == 1

    def test_output_is_always_2d(self):
        """Output is always a 2-D array regardless of input axes."""
        for axes, shape in [("YX", (64, 64)), ("CYX", (1, 64, 64))]:
            arr = np.zeros(shape, dtype=np.uint16)
            store = zarr.array(arr)
            patch = m.read_mask_patch(store, axes, img_w=64, img_h=64,
                                      y0=0, x0=0, size=32)
            assert patch.ndim == 2, f"Expected 2-D for axes={axes}, got {patch.ndim}-D"


# ===========================================================================
# CLI --mask-image  end-to-end tests
# ===========================================================================

class TestMaskImageCLI:
    """End-to-end tests for the --mask-image CLI flag."""

    def _run_patchify(self, tmp_path: Path, extra_args: list[str]) -> subprocess.CompletedProcess:
        he_path = tmp_path / "he.ome.tif"
        mx_path = tmp_path / "mx.ome.tif"
        csv_path = tmp_path / "meta.csv"
        out_dir = tmp_path / "processed"

        _write_ome(he_path, _he_array(), "CYX")
        _write_ome(mx_path, _mx_array(), "CYX")
        _write_metadata_csv(csv_path, ["CD31", "Ki67", "CD45", "PCNA"])

        cmd = [
            sys.executable, "-m", "stages.patchify",
            "--he-image", str(he_path),
            "--multiplex-image", str(mx_path),
            "--metadata-csv", str(csv_path),
            "--out", str(out_dir),
            "--patch-size", "256", "--stride", "256",
            "--tissue-min", "0.05",
            "--channels", "CD31", "Ki67", "CD45", "PCNA",
        ] + extra_args
        return subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT), out_dir

    def test_without_mask_image_no_masks_dir(self, tmp_path):
        """Without --mask-image, no masks/ directory is created."""
        result, out_dir = self._run_patchify(tmp_path, [])
        assert result.returncode == 0, result.stderr
        assert not (out_dir / "masks").exists()

    def test_without_mask_image_no_has_mask_in_index(self, tmp_path):
        """Without --mask-image, index.json entries have no has_mask key."""
        result, out_dir = self._run_patchify(tmp_path, [])
        assert result.returncode == 0, result.stderr
        data = json.loads((out_dir / "index.json").read_text())
        for patch in data["patches"]:
            assert "has_mask" not in patch

    def test_with_mask_image_creates_masks_dir(self, tmp_path):
        """--mask-image creates processed/masks/ directory."""
        mask_path = tmp_path / "mask.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint32)
        seg[100:400, 100:400] = 1
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr
        assert (out_dir / "masks").is_dir()

    def test_with_mask_image_npy_shape_and_dtype(self, tmp_path):
        """Mask .npy files have shape (patch_size, patch_size) and dtype uint32."""
        mask_path = tmp_path / "mask.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint32)
        seg[100:400, 100:400] = 1
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr

        data = json.loads((out_dir / "index.json").read_text())
        p0 = data["patches"][0]
        npy_path = out_dir / "masks" / f"{p0['x0']}_{p0['y0']}.npy"
        assert npy_path.exists(), f"Expected {npy_path} to exist"
        arr = np.load(npy_path)
        assert arr.shape == (256, 256)
        assert arr.dtype == np.uint32

    def test_with_mask_image_has_mask_in_index(self, tmp_path):
        """--mask-image adds has_mask key to every index.json entry."""
        mask_path = tmp_path / "mask.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint32)
        seg[100:400, 100:400] = 1
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr
        data = json.loads((out_dir / "index.json").read_text())
        for patch in data["patches"]:
            assert "has_mask" in patch, "Expected has_mask key in each patch entry"
            assert isinstance(patch["has_mask"], bool)

    def test_with_mask_image_records_mask_path_in_index(self, tmp_path):
        """index.json records mask_image path when --mask-image is given."""
        mask_path = tmp_path / "mask.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint32)
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr
        data = json.loads((out_dir / "index.json").read_text())
        assert "mask_image" in data
        assert str(mask_path) in data["mask_image"]

    def test_mask_patch_matches_he_patch_coordinates(self, tmp_path):
        """Mask .npy files are named with same (x0, y0) as H&E PNG patches."""
        mask_path = tmp_path / "mask.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint32)
        seg[100:400, 100:400] = 1
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr
        data = json.loads((out_dir / "index.json").read_text())

        for patch in data["patches"]:
            x0, y0 = patch["x0"], patch["y0"]
            he_png = out_dir / "he" / f"{x0}_{y0}.png"
            mask_npy = out_dir / "masks" / f"{x0}_{y0}.npy"
            assert he_png.exists(), f"Missing H&E patch {he_png}"
            assert mask_npy.exists(), f"Missing mask patch {mask_npy}"

    def test_mask_patch_uint16_input(self, tmp_path):
        """Mask stored as uint16 is read correctly and saved as uint32."""
        mask_path = tmp_path / "mask_u16.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint16)
        seg[100:400, 100:400] = 500
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr

        data = json.loads((out_dir / "index.json").read_text())
        p0 = data["patches"][0]
        arr = np.load(out_dir / "masks" / f"{p0['x0']}_{p0['y0']}.npy")
        assert arr.dtype == np.uint32

    def test_one_npy_per_patch(self, tmp_path):
        """Exactly one .npy file per patch entry in index.json."""
        mask_path = tmp_path / "mask.ome.tif"
        seg = np.zeros((512, 512), dtype=np.uint32)
        seg[100:400, 100:400] = 1
        _write_ome(mask_path, seg, "YX")

        result, out_dir = self._run_patchify(tmp_path, ["--mask-image", str(mask_path)])
        assert result.returncode == 0, result.stderr
        data = json.loads((out_dir / "index.json").read_text())

        mask_files = list((out_dir / "masks").glob("*.npy"))
        assert len(mask_files) == len(data["patches"]), (
            f"Expected {len(data['patches'])} .npy files, found {len(mask_files)}"
        )
