"""
Unit tests for patchify.py (no real data).
Tests get_ome_mpp, load_channel_indices, patch readers with synthetic/mock inputs.
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

# ---------------------------------------------------------------------------
# get_ome_mpp
# ---------------------------------------------------------------------------

OME_XML_WITH_MPP = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeZ="1.0"
            PhysicalSizeXUnit="µm" PhysicalSizeYUnit="µm" DimensionOrder="XYZCT"
            ID="Pixels:0" SizeX="100" SizeY="200" SizeZ="1" SizeC="3" SizeT="1"/>
  </Image>
</OME>"""

OME_XML_NO_MPP = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" ID="Pixels:0" SizeX="100" SizeY="200" SizeZ="1" SizeC="3" SizeT="1"/>
  </Image>
</OME>"""


def test_get_ome_mpp_returns_mpp_when_present():
    """get_ome_mpp parses PhysicalSizeX/Y from OME-XML."""
    tif = type("TiffFile", (), {"ome_metadata": OME_XML_WITH_MPP})()
    mpp_x, mpp_y = m.get_ome_mpp(tif)
    assert mpp_x == 0.325
    assert mpp_y == 0.325


def test_get_ome_mpp_returns_none_when_missing():
    """get_ome_mpp returns (None, None) when PhysicalSize not in OME-XML."""
    tif = type("TiffFile", (), {"ome_metadata": OME_XML_NO_MPP})()
    mpp_x, mpp_y = m.get_ome_mpp(tif)
    assert mpp_x is None
    assert mpp_y is None


def test_get_ome_mpp_returns_none_when_no_metadata():
    """get_ome_mpp returns (None, None) when ome_metadata is missing."""
    tif = type("TiffFile", (), {})()
    mpp_x, mpp_y = m.get_ome_mpp(tif)
    assert mpp_x is None
    assert mpp_y is None


# ---------------------------------------------------------------------------
# load_channel_indices
# ---------------------------------------------------------------------------


def test_load_channel_indices_resolves_names(tmp_path):
    """load_channel_indices finds indices from metadata CSV (case-insensitive)."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "Channel ID,Target Name\n"
        "Channel:0:0,CD31\n"
        "Channel:0:1,Ki67\n"
        "Channel:0:2,CD45\n"
        "Channel:0:3,PCNA\n"
    )
    indices, names = m.load_channel_indices(
        str(csv_path), ["CD31", "ki67", "CD45", "PCNA"]
    )
    assert indices == [0, 1, 2, 3]
    assert names == ["CD31", "Ki67", "CD45", "PCNA"]


def test_load_channel_indices_raises_on_missing_channel(tmp_path):
    """load_channel_indices raises ValueError when a requested channel is not in CSV."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("Channel ID,Target Name\nChannel:0:0,CD31\n")
    with pytest.raises(ValueError, match="Channel\\(s\\) not found"):
        m.load_channel_indices(str(csv_path), ["CD31", "MissingChannel"])


# ---------------------------------------------------------------------------
# Patch grid and readers (with synthetic zarr-like data)
# ---------------------------------------------------------------------------


def test_get_patch_grid():
    """get_patch_grid returns (i,j) coords fully within image."""
    grid = m.get_patch_grid(img_w=512, img_h=512, patch_size=256, stride=256)
    assert grid == [(0, 0), (0, 1), (1, 0), (1, 1)]
    grid2 = m.get_patch_grid(img_w=256, img_h=256, patch_size=256, stride=256)
    assert grid2 == [(0, 0)]


def test_read_multiplex_patch_rectangular_size():
    """read_multiplex_patch accepts size_y != size_x and returns (C, size_y, size_x)."""
    # Synthetic store: 4 channels, 128 height, 64 width (CYX)
    arr = np.zeros((4, 128, 64), dtype=np.uint16)
    arr[:, 10:20, 5:15] = 100
    store = zarr.array(arr)

    out = m.read_multiplex_patch(
        store,
        "CYX",
        img_w=64,
        img_h=128,
        y0=0,
        x0=0,
        size_y=32,
        size_x=24,
        channel_indices=[0, 1, 2],
    )
    assert out.shape == (3, 32, 24)
    assert out.dtype == np.uint16


def test_tissue_fraction_accepts_rgb():
    """tissue_fraction runs without error on a small RGB patch."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[10:50, 10:50] = [180, 60, 120]  # tissue-like
    f = m.tissue_fraction(rgb)
    assert 0 <= f <= 1


# ---------------------------------------------------------------------------
# CLI with synthetic data (same-size HE + multiplex)
# ---------------------------------------------------------------------------


def _write_he_ome(path, h: int, w: int, tissue_frac: float = 0.3):
    """Write a small H&E OME-TIFF with a central tissue-like region."""
    he = np.full((3, h, w), 240, dtype=np.uint8)
    r = int(min(h, w) * (tissue_frac**0.5) / 2)
    cy, cx = h // 2, w // 2
    he[:, cy - r : cy + r, cx - r : cx + r] = np.array([180, 60, 120], dtype=np.uint8)[
        :, None, None
    ]
    tifffile.imwrite(str(path), he, ome=True, metadata={"axes": "CYX"})


def _write_multiplex_ome(path, h: int, w: int, n_channels: int = 4):
    """Write a small multiplex OME-TIFF (CYX) with uint16."""
    mux = np.zeros((n_channels, h, w), dtype=np.uint16)
    mux[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 100
    tifffile.imwrite(str(path), mux, ome=True, metadata={"axes": "CYX"})


def _write_metadata_csv(path, channel_names: list[str]):
    """Write minimal channel metadata CSV."""
    lines = ["Channel ID,Target Name\n"]
    for i, name in enumerate(channel_names):
        lines.append(f"Channel:0:{i},{name}\n")
    path.write_text("".join(lines))


def test_patchify_cli_same_size_he_multiplex(tmp_path):
    """
    Run patchify CLI with synthetic same-size HE and multiplex (no real data).
    Asserts index.json exists and at least one patch is kept; one multiplex .npy has shape (C, 256, 256).
    """
    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mux.ome.tif"
    csv_path = tmp_path / "channels.csv"
    out_dir = tmp_path / "processed"

    # 512x512 so we get a 2x2 grid of 256 patches; tissue in center so several patches have tissue
    _write_he_ome(he_path, 512, 512, tissue_frac=0.5)
    _write_multiplex_ome(mx_path, 512, 512, n_channels=4)
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
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    index_path = out_dir / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert "patches" in data
    assert len(data["patches"]) >= 1
    assert data["patch_size"] == 256
    assert data["channels"] == ["CD31", "Ki67", "CD45", "PCNA"]

    # At least one HE PNG and one multiplex .npy
    p0 = data["patches"][0]
    patch_id = f"{p0['x0']}_{p0['y0']}"
    assert (out_dir / "he" / f"{patch_id}.png").exists()
    npy_path = out_dir / "multiplex" / f"{patch_id}.npy"
    assert npy_path.exists()
    patch_arr = np.load(npy_path)
    assert patch_arr.shape == (4, 256, 256)
    assert patch_arr.dtype == np.uint16


def test_patchify_cli_different_size_with_mock_mpp(tmp_path, monkeypatch):
    """
    When HE and multiplex have different pixel dimensions, physical alignment is used
    if mpp is available. Mock get_ome_mpp so HE=0.5 µm/px, MX=1.0 µm/px (same physical extent).
    Run main() in-process so the mock is applied.
    """
    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mux.ome.tif"
    csv_path = tmp_path / "channels.csv"
    out_dir = tmp_path / "processed"

    _write_he_ome(he_path, 512, 512, tissue_frac=0.5)
    _write_multiplex_ome(mx_path, 256, 256, n_channels=4)
    _write_metadata_csv(csv_path, ["CD31", "Ki67", "CD45", "PCNA"])

    calls = []

    def mock_get_ome_mpp(tif):
        calls.append(1)
        return (0.5, 0.5) if len(calls) == 1 else (1.0, 1.0)

    monkeypatch.setattr(m, "get_ome_mpp", mock_get_ome_mpp)

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
    m.main()

    data = json.loads((out_dir / "index.json").read_text())
    assert len(data["patches"]) >= 1
    p0 = data["patches"][0]
    patch_id = f"{p0['x0']}_{p0['y0']}"
    patch_arr = np.load(out_dir / "multiplex" / f"{patch_id}.npy")
    # HE mpp=0.5, MX mpp=1.0 => mpp_scale=0.5 => MX patch size = 256*0.5 = 128
    assert patch_arr.shape == (4, 128, 128)


# ---------------------------------------------------------------------------
# build_tissue_mask
# ---------------------------------------------------------------------------


def _make_cyx_tif(tmp_path, arr, name="test.ome.tif"):
    """Write a CYX OME-TIFF and return (store, axes, img_w, img_h)."""
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, ome=True, metadata={"axes": "CYX"})
    tif = tifffile.TiffFile(str(p))
    store = m.open_zarr_store(tif)
    img_w, img_h, axes = m.get_image_dims(tif)
    return store, axes, img_w, img_h


def test_build_tissue_mask_shape(tmp_path):
    """Mask shape should be (img_h//downsample, img_w//downsample)."""
    arr = np.zeros((3, 128, 192), dtype=np.uint8)
    store, axes, img_w, img_h = _make_cyx_tif(tmp_path, arr)
    mask = m.build_tissue_mask(store, axes, img_w, img_h, downsample=16)
    assert mask.shape == (128 // 16, 192 // 16)  # (8, 12)
    assert mask.dtype == bool


def test_build_tissue_mask_detects_tissue(tmp_path):
    """High-saturation pixels (pink tissue) should be detected as tissue."""
    arr = np.zeros((3, 128, 128), dtype=np.uint8)
    # Paint a tissue-like region: high R, low G, medium B -> high HSV saturation
    arr[0, 40:80, 40:80] = 220  # R
    arr[1, 40:80, 40:80] = 60  # G
    arr[2, 40:80, 40:80] = 100  # B
    store, axes, img_w, img_h = _make_cyx_tif(tmp_path, arr)
    mask = m.build_tissue_mask(store, axes, img_w, img_h, downsample=8)
    # Tissue region is rows 5-9, cols 5-9 in mask (40//8=5, 80//8=10)
    assert mask[5:10, 5:10].any(), "Expected tissue in tissue region"
    assert not mask[:3, :3].any(), "Expected no tissue in blank corner"


def test_build_tissue_mask_yxc(tmp_path):
    """build_tissue_mask works correctly with a YXC layout."""
    h, w = 128, 192
    # Shape (H, W, 3) -- YXC layout
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    p = tmp_path / "yxc.ome.tif"
    tifffile.imwrite(str(p), arr, ome=True, metadata={"axes": "YXC"})
    tif = tifffile.TiffFile(str(p))
    store = m.open_zarr_store(tif)
    img_w, img_h, axes = m.get_image_dims(tif)

    mask = m.build_tissue_mask(store, axes, img_w, img_h, downsample=16)
    assert mask.shape == (h // 16, w // 16)  # (8, 12)
    assert mask.dtype == bool


def test_build_tissue_mask_uint16(tmp_path):
    """build_tissue_mask accepts uint16 input and returns a bool ndarray."""
    h, w = 128, 128
    # All-zero uint16 array -> no tissue; dtype is preserved through tifffile
    arr = np.zeros((3, h, w), dtype=np.uint16)
    p = tmp_path / "uint16.ome.tif"
    tifffile.imwrite(str(p), arr, ome=True, metadata={"axes": "CYX"})
    tif = tifffile.TiffFile(str(p))
    store = m.open_zarr_store(tif)
    img_w, img_h, axes = m.get_image_dims(tif)

    mask = m.build_tissue_mask(store, axes, img_w, img_h, downsample=16)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool


def test_build_tissue_mask_non_multiple_dims(tmp_path):
    """Output shape is exactly (img_h // downsample, img_w // downsample) for non-multiple dimensions."""
    # 130 and 197 are not multiples of 16
    h, w = 130, 197
    arr = np.zeros((3, h, w), dtype=np.uint8)
    store, axes, img_w, img_h = _make_cyx_tif(tmp_path, arr)

    downsample = 16
    mask = m.build_tissue_mask(store, axes, img_w, img_h, downsample=downsample)
    expected_rows = h // downsample  # 130 // 16 = 8
    expected_cols = w // downsample  # 197 // 16 = 12
    assert mask.shape == (
        expected_rows,
        expected_cols,
    ), f"Expected ({expected_rows}, {expected_cols}), got {mask.shape}"
    assert mask.dtype == bool


def test_build_tissue_mask_invalid_axes(tmp_path):
    """build_tissue_mask raises ValueError when axes string is missing 'Y' or 'X'."""
    arr = np.zeros((3, 64, 64), dtype=np.uint8)
    store, _, img_w, img_h = _make_cyx_tif(tmp_path, arr)

    with pytest.raises(ValueError, match="axes must contain both"):
        m.build_tissue_mask(store, "CZT", img_w, img_h, downsample=8)

    with pytest.raises(ValueError, match="axes must contain both"):
        m.build_tissue_mask(store, "CX", img_w, img_h, downsample=8)
