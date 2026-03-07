import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

import server_minerva


@pytest.fixture(autouse=True)
def _clear_server_state():
    """Reset global server state between tests."""
    server_minerva.G.clear()
    yield
    server_minerva.G.clear()


def test_resolve_coord_cols_prefers_xt_yt():
    df = pd.DataFrame({"Xt": [1.0], "Yt": [2.0], "X": [3.0], "Y": [4.0]})
    assert server_minerva.resolve_coord_cols(df) == ("Xt", "Yt")


def test_resolve_coord_cols_falls_back_to_xy():
    df = pd.DataFrame({"X": [3.0], "Y": [4.0]})
    assert server_minerva.resolve_coord_cols(df) == ("X", "Y")


def test_resolve_coord_cols_raises_without_coords():
    df = pd.DataFrame({"col": [1.0]})
    with pytest.raises(ValueError, match="Xt/Yt or X/Y"):
        server_minerva.resolve_coord_cols(df)


def test_dzi_level_info_contains_all_levels():
    levels, max_level = server_minerva.dzi_level_info(img_w=1000, img_h=300)
    assert max_level == 10
    assert levels[0]["w"] == 1
    assert levels[max_level]["w"] == 1000
    assert levels[max_level]["h"] == 300


class _FakeLevel:
    def __init__(self, w: int, h: int):
        self.axes = "YX"
        self.shape = (h, w)


class _FakeSeries:
    def __init__(self, sizes: list[tuple[int, int]]):
        self.levels = [_FakeLevel(w, h) for w, h in sizes]


def test_choose_render_level_skips_level0_by_default():
    series = _FakeSeries([(20000, 15000), (10000, 7500), (5000, 3750)])
    level = server_minerva.choose_render_level(
        series=series,
        render_level_arg=None,
        max_render_dim=12000,
        min_render_level=1,
    )
    assert level == 1


def test_choose_render_level_falls_back_to_coarsest_if_needed():
    series = _FakeSeries([(30000, 20000), (15000, 10000), (9000, 7000)])
    level = server_minerva.choose_render_level(
        series=series,
        render_level_arg=None,
        max_render_dim=4000,
        min_render_level=1,
    )
    assert level == 2


def test_choose_render_level_clamps_explicit_to_min_level():
    series = _FakeSeries([(30000, 20000), (15000, 10000), (9000, 7000)])
    level = server_minerva.choose_render_level(
        series=series,
        render_level_arg=0,
        max_render_dim=7000,
        min_render_level=1,
    )
    assert level == 1


def test_cells_returns_empty_for_he_group():
    response = asyncio.run(
        server_minerva.cells(group="he", x0=0, y0=0, x1=100, y1=100, max_cells=500)
    )
    payload = json.loads(response.body)
    assert payload == {"cells": [], "total_in_view": 0, "sampled": False}


def test_cells_filters_by_group_and_viewport():
    server_minerva.G["df"] = pd.DataFrame(
        {
            "Xt": [10.0, 20.0, 500.0],
            "Yt": [10.0, 20.0, 500.0],
            "grp_immune": [True, False, True],
        }
    )
    server_minerva.G["coord_cols"] = ("Xt", "Yt")
    server_minerva.G["group_ids"] = {"immune"}

    response = asyncio.run(
        server_minerva.cells(group="immune", x0=0, y0=0, x1=100, y1=100, max_cells=500)
    )
    payload = json.loads(response.body)

    assert payload["total_in_view"] == 1
    assert payload["sampled"] is False
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["x"] == 10.0
    assert payload["cells"][0]["y"] == 10.0


def test_cells_applies_sampling_flag():
    n = 20
    server_minerva.G["df"] = pd.DataFrame(
        {
            "Xt": [float(i) for i in range(n)],
            "Yt": [float(i) for i in range(n)],
            "grp_immune": [True] * n,
        }
    )
    server_minerva.G["coord_cols"] = ("Xt", "Yt")
    server_minerva.G["group_ids"] = {"immune"}

    response = asyncio.run(
        server_minerva.cells(group="immune", x0=-1, y0=-1, x1=100, y1=100, max_cells=5)
    )
    payload = json.loads(response.body)

    assert payload["total_in_view"] == n
    assert payload["sampled"] is True
    assert len(payload["cells"]) == 5


def test_cells_raises_for_unknown_group():
    server_minerva.G["df"] = pd.DataFrame({"Xt": [0.0], "Yt": [0.0], "grp_immune": [True]})
    server_minerva.G["coord_cols"] = ("Xt", "Yt")
    server_minerva.G["group_ids"] = {"immune"}

    with pytest.raises(HTTPException, match="Unknown group"):
        asyncio.run(
            server_minerva.cells(group="unknown", x0=0, y0=0, x1=100, y1=100, max_cells=500)
        )


def test_he_tile_uses_thumbnail_for_low_levels(monkeypatch):
    server_minerva.G["dzi_levels"] = {0: {"w": 16, "h": 16, "cols": 1, "rows": 1}}
    server_minerva.G["thumb_rgb"] = np.full((8, 8, 3), 120, dtype=np.uint8)
    server_minerva.G["lowres_max_level"] = 0
    server_minerva.G["img_w"] = 4096

    def _should_not_read_region(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("read_region should not be called for low-res level")

    monkeypatch.setattr(server_minerva, "read_region", _should_not_read_region)

    response = asyncio.run(server_minerva.he_tile(level=0, col_row="0_0"))
    assert response.media_type == "image/jpeg"
    assert len(response.body) > 0


def test_cells_uses_precomputed_group_points():
    server_minerva.G["group_points"] = {
        "immune": np.array([[10.0, 10.0], [500.0, 500.0]], dtype=np.float32)
    }
    server_minerva.G["group_ids"] = {"immune"}

    response = asyncio.run(
        server_minerva.cells(group="immune", x0=0, y0=0, x1=100, y1=100, max_cells=500)
    )
    payload = json.loads(response.body)
    assert payload["total_in_view"] == 1
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["x"] == 10.0


def test_cells_returns_marker_colored_components():
    server_minerva.G["group_ids"] = {"tissue"}
    server_minerva.G["group_component_points"] = {
        "tissue": {
            "dna1": np.array([[10.0, 10.0], [40.0, 40.0]], dtype=np.float32),
            "panck": np.array([[20.0, 20.0]], dtype=np.float32),
        }
    }
    server_minerva.G["group_components_meta"] = {
        "tissue": [
            {
                "id": "dna1",
                "label": "DNA1",
                "color": [10, 20, 30, 200],
                "available": True,
                "source_marker": "Hoechst0",
            },
            {
                "id": "panck",
                "label": "PanCk",
                "color": [200, 100, 50, 220],
                "available": True,
                "source_marker": "Keratin",
            },
        ]
    }

    response = asyncio.run(
        server_minerva.cells(group="tissue", x0=0, y0=0, x1=25, y1=25, max_cells=50)
    )
    payload = json.loads(response.body)

    assert payload["sampled"] is False
    assert payload["total_in_view"] == 2
    assert len(payload["cells"]) == 2
    colors = {(cell["r"], cell["g"], cell["b"], cell["a"]) for cell in payload["cells"]}
    assert (10, 20, 30, 200) in colors
    assert (200, 100, 50, 220) in colors


def test_cells_component_sampling_respects_max_cells():
    server_minerva.G["group_ids"] = {"tissue"}
    server_minerva.G["group_component_points"] = {
        "tissue": {
            "dna1": np.array([[float(i), float(i)] for i in range(20)], dtype=np.float32),
            "panck": np.array([[float(i), float(i)] for i in range(20, 40)], dtype=np.float32),
        }
    }
    server_minerva.G["group_components_meta"] = {
        "tissue": [
            {
                "id": "dna1",
                "label": "DNA1",
                "color": [10, 20, 30, 200],
                "available": True,
                "source_marker": "Hoechst0",
            },
            {
                "id": "panck",
                "label": "PanCk",
                "color": [200, 100, 50, 220],
                "available": True,
                "source_marker": "Keratin",
            },
        ]
    }

    response = asyncio.run(
        server_minerva.cells(group="tissue", x0=0, y0=0, x1=100, y1=100, max_cells=7)
    )
    payload = json.loads(response.body)
    assert payload["sampled"] is True
    assert payload["total_in_view"] == 40
    assert len(payload["cells"]) <= 7


def test_load_mx_to_he_affine_inverts_warp_matrix(tmp_path):
    index_json = tmp_path / "index.json"
    payload = {
        "warp_matrix": [[2.0, 0.0, 10.0], [0.0, 4.0, -20.0]],
        "registration_mode": "affine",
    }
    index_json.write_text(json.dumps(payload))

    mx_to_he, meta = server_minerva.load_mx_to_he_affine(Path(index_json))
    assert meta["enabled"] is True
    assert mx_to_he is not None

    x_he = np.array([100.0])
    y_he = np.array([200.0])
    # forward HE->MX: x'=2x+10, y'=4y-20
    x_mx = 2.0 * x_he + 10.0
    y_mx = 4.0 * y_he - 20.0
    x_back, y_back = server_minerva.apply_affine_points(x_mx, y_mx, mx_to_he)
    assert x_back[0] == pytest.approx(x_he[0])
    assert y_back[0] == pytest.approx(y_he[0])
