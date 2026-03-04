import numpy as np
import tifffile


def _write_synthetic_ome_tiff(path, arr, axes: str):
    # tifffile writes OME metadata based on axes string
    tifffile.imwrite(path, arr, ome=True, metadata={"axes": axes})


def test_read_region_rgb_from_ome_tiff(tmp_path):
    """
    Contract:
    - `read_region_rgb(path, y0, x0, h, w)` returns (h,w,3) uint8
    - Works on a simple CYX OME-TIFF
    """
    img_h, img_w = 40, 50
    # CYX
    c = np.zeros((3, img_h, img_w), dtype=np.uint16)
    c[0] = 1000  # R
    c[1] = 2000  # G
    c[2] = 3000  # B
    p = tmp_path / "he.ome.tif"
    _write_synthetic_ome_tiff(str(p), c, axes="CYX")

    from ome_reader import read_region_rgb  # noqa: WPS433 (import inside test ensures RED first)

    out = read_region_rgb(str(p), y0=5, x0=7, h=10, w=11)
    assert out.shape == (10, 11, 3)
    assert out.dtype == np.uint8


def test_read_region_channels_from_multiplex(tmp_path):
    """
    Contract:
    - `read_region_channels(path, y0, x0, h, w, channels=[...])` returns (h,w,len(channels))
    - Supports 36-channel multiplex in CYX layout
    """
    img_h, img_w = 30, 32
    mux = np.zeros((36, img_h, img_w), dtype=np.uint16)
    mux[0] = 1
    mux[10] = 11
    mux[35] = 36

    p = tmp_path / "mux.ome.tif"
    _write_synthetic_ome_tiff(str(p), mux, axes="CYX")

    from ome_reader import read_region_channels  # noqa: WPS433

    out = read_region_channels(str(p), y0=0, x0=0, h=8, w=9, channels=[0, 10, 35])
    assert out.shape == (8, 9, 3)
    assert out.dtype == np.uint16
    assert int(out[0, 0, 0]) == 1
    assert int(out[0, 0, 1]) == 11
    assert int(out[0, 0, 2]) == 36

