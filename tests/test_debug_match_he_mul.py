import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

from tools.debug_match_he_mul import _read_channel_overview, _read_window


def test_read_channel_overview_supports_iyx_axis():
    """_read_channel_overview must index I-axis channels, not always plane 0."""
    data = np.zeros((3, 8, 8), dtype=np.uint16)  # IYX
    data[2, 2:6, 2:6] = 1234
    out = _read_channel_overview(data, axes="IYX", img_h=8, img_w=8, ds=2, c_idx=2)
    assert out.shape == (4, 4)
    assert int(out.max()) == 1234


def test_read_window_supports_yxs_rgb_axis():
    """_read_window should preserve S-axis RGB data as CYX when reading YXS."""
    data = np.zeros((10, 12, 3), dtype=np.uint8)  # YXS
    data[3:7, 4:8, 0] = 11
    data[3:7, 4:8, 1] = 22
    data[3:7, 4:8, 2] = 33

    crop, axes = _read_window(
        data,
        axes="YXS",
        y0=3,
        x0=4,
        height=4,
        width=4,
        step=1,
        channel_index=None,
    )
    assert axes == "CYX"
    assert crop.shape == (3, 4, 4)
    assert int(crop[0].mean()) == 11
    assert int(crop[1].mean()) == 22
    assert int(crop[2].mean()) == 33


def test_debug_match_he_mul_save_png_smoke(tmp_path: Path):
    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mx.ome.tif"
    metadata_csv = tmp_path / "metadata.csv"
    out_png = tmp_path / "debug_match.png"

    he = np.zeros((3, 96, 96), dtype=np.uint8)
    he[0, 20:80, 20:80] = 180
    he[1, 30:70, 30:70] = 140
    he[2, 40:60, 40:60] = 220

    mx = np.zeros((3, 96, 96), dtype=np.uint16)
    mx[0] = np.arange(96 * 96, dtype=np.uint16).reshape(96, 96)  # DNA-like
    mx[1, 24:72, 24:72] = 1200
    mx[2, 8:40, 8:40] = 1600

    tifffile.imwrite(str(he_path), he, metadata={"axes": "CYX"})
    tifffile.imwrite(str(mx_path), mx, metadata={"axes": "CYX"})

    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Channel_Number", "Marker_Name"])
        writer.writerow([1, "DNA1"])
        writer.writerow([2, "CD45"])
        writer.writerow([3, "PanCK"])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.debug_match_he_mul",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--metadata-csv",
            str(metadata_csv),
            "--downsample",
            "2",
            "--save-png",
            str(out_png),
        ],
        check=True,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_debug_match_he_mul_save_png_with_seg_row(tmp_path: Path):
    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mx.ome.tif"
    seg_path = tmp_path / "mx.ome.seg.tif"
    metadata_csv = tmp_path / "metadata.csv"
    out_png = tmp_path / "debug_match_seg.png"

    he = np.zeros((3, 128, 128), dtype=np.uint8)
    he[0, 20:110, 20:110] = 180
    he[1, 30:100, 30:100] = 140
    he[2, 40:90, 40:90] = 220

    mx = np.zeros((3, 128, 128), dtype=np.uint16)
    mx[0, 24:104, 24:104] = 1200  # DNA-like
    mx[1, 30:100, 30:100] = 600
    mx[2, 16:64, 16:64] = 900

    seg = np.zeros((128, 128), dtype=np.uint32)
    seg[40:56, 40:56] = 101
    seg[60:78, 70:88] = 202

    tifffile.imwrite(str(he_path), he, metadata={"axes": "CYX"})
    tifffile.imwrite(str(mx_path), mx, metadata={"axes": "CYX"})
    tifffile.imwrite(str(seg_path), seg, metadata={"axes": "YX"})

    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Channel_Number", "Marker_Name"])
        writer.writerow([1, "DNA1"])
        writer.writerow([2, "CD45"])
        writer.writerow([3, "PanCK"])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.debug_match_he_mul",
            "--he-image",
            str(he_path),
            "--multiplex-image",
            str(mx_path),
            "--seg-image",
            str(seg_path),
            "--metadata-csv",
            str(metadata_csv),
            "--downsample",
            "4",
            "--zoom-size",
            "32",
            "--zoom-downsample",
            "1",
            "--save-png",
            str(out_png),
        ],
        check=True,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 0
