from __future__ import annotations

import argparse

import numpy as np
from fastapi import HTTPException
from fastapi.routing import APIRoute

import tools.view_groups_web as web


class _DummyFrame:
    def __init__(self, group_id: str):
        self.group_id = group_id
        self.label = group_id.upper()
        self.image_rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        self.used_markers = ("CD45",) if group_id != "he" else ()
        self.missing_markers = ()
        self.colormap = "tab20"
        self.marker_colors = (("CD45", (12, 34, 56)),) if group_id != "he" else ()


class _DummyCore:
    init_kwargs: list[dict] = []
    render_calls: list[tuple[int, str, str]] = []

    def __init__(
        self,
        he_image: str,
        multiplex_image: str,
        metadata_csv: str,
        he_level: int | None = None,
        multiplex_level: int | None = None,
        auto_target_max_dim: int = 2400,
        min_level: int = 1,
        default_colormap: str = "turbo",
        index_json: str | None = None,
        preload_multiplex: bool = False,
        contrast_gamma: float = 0.65,
        contrast_gain: float = 1.35,
    ):
        self.he_rgb = np.full((8, 12, 3), 120, dtype=np.uint8)
        self._dim = int(auto_target_max_dim)
        _DummyCore.init_kwargs.append(
            {
                "he_image": he_image,
                "multiplex_image": multiplex_image,
                "metadata_csv": metadata_csv,
                "he_level": he_level,
                "multiplex_level": multiplex_level,
                "auto_target_max_dim": auto_target_max_dim,
                "min_level": min_level,
                "default_colormap": default_colormap,
                "index_json": index_json,
                "preload_multiplex": preload_multiplex,
                "contrast_gamma": contrast_gamma,
                "contrast_gain": contrast_gain,
            }
        )

    def close(self) -> None:
        return None

    def summary(self) -> dict:
        return {
            "he_level": 1,
            "multiplex_level": 2,
            "he_shape": [8, 12],
            "multiplex_shape": [8, 12],
            "default_colormap": "turbo",
            "registered": True,
            "registration_index_json": "processed_wd/index.json",
            "preload_multiplex": True,
            "contrast_gamma": 0.65,
            "contrast_gain": 1.35,
            "groups": [
                {
                    "id": "he",
                    "label": "H&E",
                    "available": True,
                    "used_markers": [],
                    "missing_markers": [],
                },
                {
                    "id": "immune",
                    "label": "Immune",
                    "available": True,
                    "colormap": "tab20",
                    "used_markers": ["CD45"],
                    "missing_markers": [],
                },
            ],
        }

    def render_group(self, group_id: str, combine_mode: str = "max") -> _DummyFrame:
        if group_id == "bad":
            raise ValueError("unknown group")
        _DummyCore.render_calls.append((self._dim, group_id, combine_mode))
        return _DummyFrame(group_id.strip().lower())


def _make_args(min_level: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        he_image="data/WD-76845-096.ome.tif",
        multiplex_image="data/WD-76845-097.ome.tif",
        metadata_csv="data/WD-76845-097-metadata.csv",
        index_json="processed_wd/index.json",
        he_level=None,
        multiplex_level=None,
        min_level=min_level,
        auto_max_dim=1200,
        detail_auto_max_dim=3200,
        high_auto_max_dim=7000,
        default_colormap="turbo",
        contrast_gamma=0.65,
        contrast_gain=1.35,
        preload_multiplex=True,
        host="127.0.0.1",
        port=8010,
    )


def _route_endpoint(app, path: str):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route.endpoint
    raise AssertionError(f"route not found: {path}")


def test_index_html_contains_viewer_controls() -> None:
    html = web._make_index_html()  # noqa: SLF001
    assert 'id="group-buttons"' in html
    assert 'id="alpha"' in html
    assert "two-finger scroll pans" in html
    assert 'id="legend-overlay"' in html


def test_build_app_enforces_min_level_ge_1(monkeypatch) -> None:
    _DummyCore.init_kwargs.clear()
    monkeypatch.setattr(web, "GroupVisualizerCore", _DummyCore)
    app = web.build_app(_make_args(min_level=0))
    payload = _route_endpoint(app, "/api/meta")()
    assert payload["available_tiers"] == ["base", "detail", "high"]
    assert _DummyCore.init_kwargs
    assert _DummyCore.init_kwargs[0]["min_level"] == 1


def test_api_meta_reports_available_tiers(monkeypatch) -> None:
    _DummyCore.init_kwargs.clear()
    monkeypatch.setattr(web, "GroupVisualizerCore", _DummyCore)
    app = web.build_app(_make_args(min_level=1))
    payload = _route_endpoint(app, "/api/meta")()
    assert payload["available_tiers"] == ["base", "detail", "high"]
    assert set(payload["tiers"].keys()) == {"base", "detail", "high"}


def test_api_group_png_cache_key_is_tier_aware(monkeypatch) -> None:
    _DummyCore.render_calls.clear()
    monkeypatch.setattr(web, "GroupVisualizerCore", _DummyCore)
    app = web.build_app(_make_args(min_level=1))
    api_group_png = _route_endpoint(app, "/api/group.png")
    res1 = api_group_png(group_id="immune", combine_mode="max", tier="base")
    res2 = api_group_png(group_id="immune", combine_mode="max", tier="base")
    res3 = api_group_png(group_id="immune", combine_mode="max", tier="detail")
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res3.status_code == 200
    assert len(_DummyCore.render_calls) == 2


def test_api_group_meta_returns_tier_shape_and_levels(monkeypatch) -> None:
    monkeypatch.setattr(web, "GroupVisualizerCore", _DummyCore)
    app = web.build_app(_make_args(min_level=1))
    api_group_meta = _route_endpoint(app, "/api/group-meta")
    payload = api_group_meta(group_id="immune", combine_mode="max", tier="base")
    assert payload["tier"] == "base"
    assert payload["he_level"] == 1
    assert payload["multiplex_level"] == 2
    assert payload["he_shape"] == [8, 12]
    assert payload["multiplex_shape"] == [8, 12]
    assert payload["registered"] is True
    assert payload["marker_colors"] == [["CD45", [12, 34, 56]]]


def test_api_rejects_invalid_tier(monkeypatch) -> None:
    monkeypatch.setattr(web, "GroupVisualizerCore", _DummyCore)
    app = web.build_app(_make_args(min_level=1))
    api_he_png = _route_endpoint(app, "/api/he.png")
    try:
        api_he_png(tier="nope")
        assert False, "expected HTTPException for invalid tier"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "invalid tier" in str(exc.detail)
