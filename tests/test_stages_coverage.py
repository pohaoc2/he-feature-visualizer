"""
Additional tests targeting uncovered branches identified by the comprehensive
review of stages/, utils/channels.py, and utils/ome.py.

Coverage targets (lines previously at 0%):
  stages/assign_cells.py  — assign_type exception guards, assign_state
                             (apoptotic/quiescent/healthy/exception branches),
                             rasterize_cells short-contour skip,
                             compute_thresholds state-marker and Ecadherin-missing paths,
                             _resolve_coord_cols X/Y fallback + missing columns error.
  utils/channels.py       — new-format CSV (Channel_Number/Marker_Name),
                             duplicate-key skipping, bad Channel_Number error,
                             unrecognised format error.
  utils/ome.py            — _safe_float non-float value,
                             get_ome_mpp ParseError + pixels-is-None branches,
                             read_overview_chw transposition.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# stages/assign_cells — assign_type exception guards
# ---------------------------------------------------------------------------


def test_assign_type_exception_guard_in_get():
    """assign_type._get silently returns 0 when a value cannot be converted to float."""
    from stages.assign_cells import assign_type

    class BadVal:
        """A value that raises on float()."""

        def __float__(self):
            raise RuntimeError("boom")

    bad_row = pd.Series({"Keratin": BadVal()})
    thresholds = {"Keratin": 500.0}
    # Bad value → treated as 0 → no marker exceeds threshold → "other"
    assert assign_type(bad_row, thresholds) == "other"


def test_assign_type_outer_exception_guard():
    """assign_type outer exception guard returns 'other' if iteration raises."""
    from stages.assign_cells import assign_type

    # Pass a non-iterable as thresholds to trigger the outer except branch.
    bad_thresholds = None  # .get() will raise AttributeError
    row = pd.Series({"Keratin": 1000.0})
    assert assign_type(row, bad_thresholds) == "other"


# ---------------------------------------------------------------------------
# stages/assign_cells — assign_state missing branches
# ---------------------------------------------------------------------------


def test_assign_state_apoptotic_type4():
    """assign_state returns 'apoptotic' for CellViT Dead class (type_cellvit=4)."""
    from stages.assign_cells import assign_state

    row = pd.Series({"Ki67": 0.0, "PCNA": 0.0, "Vimentin": 0.0, "Ecadherin": 500.0})
    thresholds = {
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
        "Ecadherin_high": 400.0,
        "Keratin": 500.0,
    }
    assert assign_state(row, thresholds, type_cellvit=4) == "apoptotic"


def test_assign_state_quiescent():
    """assign_state returns 'quiescent' when Keratin is high and E-cad is high."""
    from stages.assign_cells import assign_state

    thresholds = {
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
        "Ecadherin_high": 400.0,
        "Keratin": 500.0,
    }
    # High Keratin + E-cad above 'high' threshold → quiescent (resting tumor)
    row = pd.Series(
        {
            "Ki67": 0.0,
            "PCNA": 0.0,
            "Vimentin": 0.0,
            "Ecadherin": 600.0,  # above Ecadherin_high=400
            "Keratin": 1000.0,
        }
    )
    assert assign_state(row, thresholds) == "quiescent"


def test_assign_state_healthy():
    """assign_state returns 'healthy' when E-cad is high but Keratin is low."""
    from stages.assign_cells import assign_state

    thresholds = {
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
        "Ecadherin_high": 400.0,
        "Keratin": 500.0,
    }
    # E-cad above high threshold, Keratin below threshold → healthy epithelial
    row = pd.Series(
        {
            "Ki67": 0.0,
            "PCNA": 0.0,
            "Vimentin": 0.0,
            "Ecadherin": 600.0,
            "Keratin": 0.0,
        }
    )
    assert assign_state(row, thresholds) == "healthy"


def test_assign_state_inner_exception_guard():
    """assign_state outer try/except returns 'other' if comparison raises."""
    from stages.assign_cells import assign_state

    # Pass None thresholds so dict.get raises AttributeError inside try block.
    row = pd.Series({"Ki67": 0.0, "PCNA": 0.0, "Vimentin": 0.0, "Ecadherin": 0.0})
    result = assign_state(row, None)  # type: ignore[arg-type]
    assert result == "other"


def test_assign_state_get_exception_branch():
    """assign_state._get silently returns 0 when row.get raises."""
    from stages.assign_cells import assign_state

    class UnreadableRow:
        """A row whose .get always raises."""

        def get(self, key, default=0):
            raise RuntimeError("unreadable")

        def __getattr__(self, name):
            raise RuntimeError("unreadable")

    thresholds = {
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
        "Ecadherin_high": 400.0,
        "Keratin": 500.0,
    }
    # All marker reads raise → treated as 0 → no condition met → "other"
    result = assign_state(UnreadableRow(), thresholds)
    assert result == "other"


# ---------------------------------------------------------------------------
# stages/assign_cells — rasterize_cells short-contour skip
# ---------------------------------------------------------------------------


def test_rasterize_cells_skips_short_contour():
    """rasterize_cells skips cells whose contour has fewer than 3 points."""
    from stages.assign_cells import rasterize_cells, CELL_TYPE_COLORS

    short_cell = {
        "contour": [[64, 64], [80, 64]],  # only 2 points → must be skipped
        "cell_type": "tumor",
        "cell_state": "other",
    }
    valid_cell = {
        "contour": [[100, 100], [120, 100], [120, 120], [100, 120]],
        "cell_type": "immune",
        "cell_state": "other",
    }

    canvas = rasterize_cells(
        cells=[short_cell, valid_cell],
        patch_size=256,
        color_key="cell_type",
        color_map=CELL_TYPE_COLORS,
    )

    # Short-contour region must remain transparent
    assert canvas[64, 64, 3] == 0, "Short-contour cell must not be drawn"
    # Valid cell region must be painted
    assert canvas[110, 110, 3] > 0, "Valid 4-point cell must be drawn"


def test_rasterize_cells_empty_list_returns_zeros():
    """rasterize_cells returns an all-zero canvas when given an empty cell list."""
    from stages.assign_cells import rasterize_cells, CELL_TYPE_COLORS

    canvas = rasterize_cells(
        cells=[],
        patch_size=64,
        color_key="cell_type",
        color_map=CELL_TYPE_COLORS,
    )
    assert canvas.shape == (64, 64, 4)
    assert np.all(canvas == 0)


# ---------------------------------------------------------------------------
# stages/assign_cells — compute_thresholds state-marker and Ecadherin paths
# ---------------------------------------------------------------------------


def test_compute_thresholds_state_markers_present():
    """compute_thresholds computes Ki67/PCNA/Vimentin thresholds when columns exist."""
    from stages.assign_cells import compute_thresholds

    df = pd.DataFrame(
        {
            "Ki67": [0.0, 50.0, 100.0],
            "PCNA": [0.0, 50.0, 100.0],
            "Vimentin": [0.0, 50.0, 100.0],
            "Ecadherin": [0.0, 50.0, 100.0],
        }
    )
    thresholds = compute_thresholds(df, default_state_percentile=50)
    # p50 of [0, 50, 100] == 50.0
    for marker in ["Ki67", "PCNA", "Vimentin"]:
        assert thresholds[marker] == pytest.approx(
            50.0
        ), f"Expected p50=50 for {marker}"


def test_compute_thresholds_ecadherin_missing():
    """compute_thresholds sets Ecadherin=-inf and Ecadherin_high=+inf when column absent."""
    from stages.assign_cells import compute_thresholds

    df = pd.DataFrame({"Keratin": [0.0, 50.0, 100.0]})
    thresholds = compute_thresholds(df)
    assert thresholds["Ecadherin"] == float("-inf")
    assert thresholds["Ecadherin_high"] == float("inf")


def test_compute_thresholds_ecadherin_present():
    """compute_thresholds computes Ecadherin low/high thresholds when column present."""
    from stages.assign_cells import compute_thresholds

    df = pd.DataFrame({"Ecadherin": [0.0, 25.0, 50.0, 75.0, 100.0]})
    thresholds = compute_thresholds(df)
    # Default Ecadherin low = p25, Ecadherin high = p50
    assert thresholds["Ecadherin"] == pytest.approx(25.0)
    assert thresholds["Ecadherin_high"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# stages/assign_cells — _resolve_coord_cols
# ---------------------------------------------------------------------------


def test_resolve_coord_cols_prefers_xt_yt():
    """_resolve_coord_cols returns Xt/Yt when both are present."""
    from stages.assign_cells import _resolve_coord_cols

    df = pd.DataFrame({"Xt": [1], "Yt": [1], "X": [1], "Y": [1]})
    x_col, y_col = _resolve_coord_cols(df)
    assert x_col == "Xt"
    assert y_col == "Yt"


def test_resolve_coord_cols_falls_back_to_x_y():
    """_resolve_coord_cols falls back to X/Y when Xt/Yt are absent."""
    from stages.assign_cells import _resolve_coord_cols

    df = pd.DataFrame({"X": [1], "Y": [1], "Keratin": [0]})
    x_col, y_col = _resolve_coord_cols(df)
    assert x_col == "X"
    assert y_col == "Y"


def test_resolve_coord_cols_raises_when_no_coords():
    """_resolve_coord_cols raises ValueError when no valid coordinate columns exist."""
    from stages.assign_cells import _resolve_coord_cols

    df = pd.DataFrame({"Keratin": [0], "CD45": [0]})
    with pytest.raises(ValueError, match="coordinate columns"):
        _resolve_coord_cols(df)


# ---------------------------------------------------------------------------
# utils/channels — new-format CSV (Channel_Number / Marker_Name)
# ---------------------------------------------------------------------------


def test_load_channel_metadata_new_format(tmp_path):
    """load_channel_metadata parses Channel_Number + Marker_Name format correctly."""
    from utils.channels import load_channel_metadata

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("Channel_Number,Marker_Name\n1,CD31\n2,Ki67\n3,CD45\n")

    result = load_channel_metadata(str(csv_path))

    assert "cd31" in result
    assert result["cd31"] == (0, "CD31")  # 1-based → 0-based
    assert result["ki67"] == (1, "Ki67")
    assert result["cd45"] == (2, "CD45")


def test_load_channel_metadata_duplicate_key_skipped(tmp_path):
    """load_channel_metadata keeps the first occurrence of a duplicate marker name."""
    from utils.channels import load_channel_metadata

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("Channel_Number,Marker_Name\n1,CD31\n2,CD31\n")

    result = load_channel_metadata(str(csv_path))
    # First CD31 entry at channel 1 (index 0) should win.
    assert result["cd31"] == (0, "CD31")


def test_load_channel_metadata_bad_channel_number(tmp_path):
    """load_channel_metadata raises ValueError when Channel_Number is not an integer."""
    from utils.channels import load_channel_metadata

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("Channel_Number,Marker_Name\nnot_an_int,CD31\n")

    with pytest.raises(ValueError, match="Channel_Number"):
        load_channel_metadata(str(csv_path))


def test_load_channel_metadata_unrecognised_format(tmp_path):
    """load_channel_metadata raises ValueError for an unrecognised CSV layout."""
    from utils.channels import load_channel_metadata

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("ColA,ColB\n1,CD31\n")

    with pytest.raises(ValueError, match="Unrecognised"):
        load_channel_metadata(str(csv_path))


def test_resolve_channel_indices_new_format(tmp_path):
    """resolve_channel_indices works with Channel_Number/Marker_Name format."""
    from utils.channels import resolve_channel_indices

    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("Channel_Number,Marker_Name\n1,CD31\n2,Ki67\n3,CD45\n")

    indices, names = resolve_channel_indices(str(csv_path), ["CD31", "CD45"])
    assert indices == [0, 2]
    assert names == ["CD31", "CD45"]


# ---------------------------------------------------------------------------
# utils/ome — _safe_float, ParseError, pixels-None, transposition
# ---------------------------------------------------------------------------


def test_safe_float_non_numeric():
    """_safe_float returns None for non-numeric strings."""
    from utils.ome import _safe_float

    assert _safe_float("not_a_number") is None
    assert _safe_float(None) is None
    assert _safe_float("1.23") == pytest.approx(1.23)


def test_get_ome_mpp_parse_error():
    """get_ome_mpp returns (None, None) when ome_metadata is malformed XML."""
    from utils.ome import get_ome_mpp

    tif = type("TiffFile", (), {"ome_metadata": "<<<not valid xml>>>"})()
    mpp_x, mpp_y = get_ome_mpp(tif)
    assert mpp_x is None
    assert mpp_y is None


def test_get_ome_mpp_pixels_none():
    """get_ome_mpp returns (None, None) when the Pixels element is absent."""
    from utils.ome import get_ome_mpp

    xml_no_pixels = """<?xml version="1.0"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0"/>
</OME>"""
    tif = type("TiffFile", (), {"ome_metadata": xml_no_pixels})()
    mpp_x, mpp_y = get_ome_mpp(tif)
    assert mpp_x is None
    assert mpp_y is None


def test_get_ome_mpp_from_pages_description():
    """get_ome_mpp falls back to pages[0].description when ome_metadata is absent."""
    from utils.ome import get_ome_mpp

    xml_with_mpp = """<?xml version="1.0"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels PhysicalSizeX="0.5" PhysicalSizeY="0.5"
            DimensionOrder="XYZCT" ID="Pixels:0"
            SizeX="100" SizeY="100" SizeZ="1" SizeC="1" SizeT="1"/>
  </Image>
</OME>"""

    page = type("Page", (), {"description": xml_with_mpp})()
    tif = type("TiffFile", (), {"ome_metadata": None, "pages": [page]})()

    mpp_x, mpp_y = get_ome_mpp(tif)
    assert mpp_x == pytest.approx(0.5)
    assert mpp_y == pytest.approx(0.5)


def test_read_overview_chw_yxc_axes():
    """read_overview_chw transposes YXC→CYX and returns (C, H//ds, W//ds)."""
    import zarr
    from utils.ome import read_overview_chw

    # Build an in-memory zarr array with YXC layout: (H=64, W=64, C=3)
    h, w, c, ds = 64, 64, 3, 8
    data = np.zeros((h, w, c), dtype=np.uint16)
    data[0:32, :, 0] = 1000  # top-half of channel 0
    store = zarr.array(data)

    overview = read_overview_chw(store, "YXC", img_h=h, img_w=w, ds=ds)

    # Result must be canonical (C, H//ds, W//ds)
    assert overview.shape == (
        c,
        h // ds,
        w // ds,
    ), f"Expected (3,8,8), got {overview.shape}"
    assert overview.dtype == np.uint16


def test_read_overview_chw_cyx_no_transpose(tmp_path):
    """read_overview_chw returns (C, H//ds, W//ds) for standard CYX layout."""
    import tifffile
    from utils.ome import read_overview_chw, open_zarr_store, get_image_dims

    h, w, c = 64, 64, 4
    arr = np.zeros((c, h, w), dtype=np.uint16)
    arr[0, 10:20, 10:20] = 5000

    p = tmp_path / "cyx.ome.tif"
    tifffile.imwrite(str(p), arr, ome=True, metadata={"axes": "CYX"})

    with tifffile.TiffFile(str(p)) as tif:
        store = open_zarr_store(tif)
        img_w, img_h, axes = get_image_dims(tif)
        overview = read_overview_chw(store, axes, img_h, img_w, ds=8)

    assert overview.shape == (
        c,
        h // 8,
        w // 8,
    ), f"Expected (4,8,8), got {overview.shape}"
