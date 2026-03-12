from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from tools.viz_mask import (
    crop_tiff_pair,
    parse_crop_region,
    visualize_crop_dna,
    visualize_tiff_pair,
)


def test_parse_crop_region_supports_short_and_full_forms():
    assert parse_crop_region(None) is None
    assert parse_crop_region("10,20") == (10, 20, 1024, 1024)
    assert parse_crop_region("10,20,30") == (10, 20, 30, 30)
    assert parse_crop_region("10,20,30,40") == (10, 20, 30, 40)


def test_crop_tiff_pair_with_explicit_region_on_same_grid(tmp_path: Path):
    image1 = tmp_path / "image1.ome.tif"
    image2 = tmp_path / "image2.ome.tif"
    out_prefix = tmp_path / "paired_crop"

    arr1 = np.arange(3 * 64 * 64, dtype=np.uint8).reshape(3, 64, 64)
    arr2 = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
    tifffile.imwrite(str(image1), arr1, metadata={"axes": "CYX"})
    tifffile.imwrite(str(image2), arr2, metadata={"axes": "CYX"})

    out1, out2 = crop_tiff_pair(
        image1,
        image2,
        crop_region="8,12,16",
        save_path=out_prefix,
    )

    assert out1 == tmp_path / "paired_crop_image1.ome.tif"
    assert out2 == tmp_path / "paired_crop_image2.ome.tif"

    with tifffile.TiffFile(out1) as tif:
        assert tif.series[0].shape == (3, 16, 16)
        np.testing.assert_array_equal(tif.asarray(), arr1[:, 12:28, 8:24])

    with tifffile.TiffFile(out2) as tif:
        assert tif.series[0].shape == (5, 16, 16)
        np.testing.assert_array_equal(tif.asarray(), arr2[:, 12:28, 8:24])


def test_visualize_tiff_pair_saves_png(tmp_path: Path):
    image1 = tmp_path / "image1.ome.tif"
    image2 = tmp_path / "image2.ome.tif"
    out_png = tmp_path / "pair.png"

    arr1 = np.zeros((3, 32, 32), dtype=np.uint8)
    arr1[0, 8:24, 8:24] = 255
    arr2 = np.zeros((1, 32, 32), dtype=np.uint16)
    arr2[0, 4:28, 4:28] = 2000
    tifffile.imwrite(str(image1), arr1, metadata={"axes": "CYX"})
    tifffile.imwrite(str(image2), arr2, metadata={"axes": "CYX"})

    result = visualize_tiff_pair(image1, image2, save_path=out_png, downsample=2)

    assert result == out_png
    assert out_png.exists()


def test_visualize_crop_dna_saves_three_panel_png(tmp_path: Path):
    mask_path = tmp_path / "cells.mask.tif"
    mx_path = tmp_path / "mx.ome.tif"
    out_png = tmp_path / "mask_dna_overlap.png"

    mask = np.zeros((64, 64), dtype=np.uint32)
    mask[20:44, 18:40] = 7
    mx = np.zeros((2, 64, 64), dtype=np.uint16)
    mx[1] = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)

    tifffile.imwrite(str(mask_path), mask, metadata={"axes": "YX"})
    tifffile.imwrite(str(mx_path), mx, metadata={"axes": "CYX"})

    visualize_crop_dna(
        mask_path,
        mx_path,
        out_png,
        row=32,
        col=32,
        crop_size=32,
        dna_channel=1,
    )

    assert out_png.exists()
    img = np.asarray(Image.open(out_png))
    # 3 panels of width 32 with 4px gaps.
    assert img.shape[0] == 32
    assert img.shape[1] == (32 * 3 + 4 * 2)

    # Middle panel (DNA) should carry non-zero signal when dna_channel=1.
    dna_panel = img[:, 32 + 4 : 32 + 4 + 32]
    assert float(dna_panel.mean()) > 0.0


def test_visualize_crop_dna_rejects_missing_channel(tmp_path: Path):
    mask_path = tmp_path / "cells.mask.tif"
    mx_path = tmp_path / "mx.ome.tif"
    out_png = tmp_path / "mask_dna_overlap.png"

    mask = np.zeros((32, 32), dtype=np.uint32)
    mx = np.zeros((1, 32, 32), dtype=np.uint16)

    tifffile.imwrite(str(mask_path), mask, metadata={"axes": "YX"})
    tifffile.imwrite(str(mx_path), mx, metadata={"axes": "CYX"})

    with pytest.raises(ValueError, match="Requested"):
        visualize_crop_dna(
            mask_path,
            mx_path,
            out_png,
            row=16,
            col=16,
            crop_size=16,
            dna_channel=3,
        )
