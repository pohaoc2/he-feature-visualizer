from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.paper_figures.build_marker_zoom_figure import _format_distance_label
from tools.paper_figures.select_marker_patch_examples import (
	_header_label_centers,
	_select_top,
	export_metabolic_tiles,
	make_metabolic_gradient_tiles,
)
from tools.paper_figures.vis_he_mx_side_by_side import _draw_crop_boxes
from tools.paper_figures.vis_cell_feature_tiles import (
	export_rendered_tiles,
	_load_tile,
	_remap_legacy_state_palette,
)
from utils.colormaps import CELL_STATE_COLORS


def test_header_label_centers_fill_band_evenly() -> None:
	centers = _header_label_centers(10, 120, 3)
	assert centers == [30.0, 70.0, 110.0]


def test_format_distance_label_uses_mm_for_large_wsi_bar() -> None:
	assert _format_distance_label(5000.0) == "5 mm"
	assert _format_distance_label(20.0) == "20 µm"


def test_remap_legacy_state_palette_recolors_legacy_state_rgb() -> None:
	arr = np.array(
		[
			[[240, 190, 0], [120, 120, 120]],
			[[110, 60, 20], [0, 0, 0]],
		],
		dtype=np.uint8,
	)

	remapped = _remap_legacy_state_palette(arr)

	assert tuple(remapped[0, 0]) == CELL_STATE_COLORS["proliferative"][:3]
	assert tuple(remapped[0, 1]) == CELL_STATE_COLORS["nonprolif"][:3]
	assert tuple(remapped[1, 0]) == CELL_STATE_COLORS["dead"][:3]
	assert tuple(remapped[1, 1]) == (0, 0, 0)


def test_load_tile_renders_vasculature_rgba_on_black(tmp_path: Path) -> None:
	processed = tmp_path / "processed"
	(processed / "vasculature").mkdir(parents=True)

	overlay = np.zeros((8, 8, 4), dtype=np.uint8)
	overlay[2:6, 2:6, :3] = np.array([255, 40, 40], dtype=np.uint8)
	overlay[2:6, 2:6, 3] = 200
	Image.fromarray(overlay, mode="RGBA").save(processed / "vasculature" / "0_0.png")

	tile = _load_tile(
		processed,
		"0_0",
		{"subdir": "vasculature", "mode": "rgba_on_black"},
		tile_size=8,
	)
	arr = np.asarray(tile)

	assert tuple(arr[0, 0]) == (0, 0, 0)
	assert arr[3, 3, 0] > 0
	assert arr[3, 3, 0] > arr[3, 3, 1]
	assert arr[3, 3, 0] > arr[3, 3, 2]


def test_export_rendered_tiles_writes_expected_stage1_tile_files(tmp_path: Path) -> None:
	processed = tmp_path / "processed"
	for subdir, mode in [
		("cell_types/union", "RGB"),
		("cell_states/union", "RGB"),
		("vasculature", "RGBA"),
	]:
		(processed / subdir).mkdir(parents=True)

	Image.new("RGB", (8, 8), (10, 20, 30)).save(processed / "cell_types/union" / "0_0.png")
	Image.new("RGB", (8, 8), (240, 190, 0)).save(processed / "cell_states/union" / "0_0.png")
	overlay = np.zeros((8, 8, 4), dtype=np.uint8)
	overlay[2:6, 2:6, :3] = np.array([255, 40, 40], dtype=np.uint8)
	overlay[2:6, 2:6, 3] = 255
	Image.fromarray(overlay, mode="RGBA").save(processed / "vasculature" / "0_0.png")

	exported = export_rendered_tiles(
		processed,
		["0_0"],
		tmp_path / "paper/figures/stage1/tiles",
		tile_size=8,
		group_id="g1",
	)

	assert set(exported["0_0"]) == {"cell_type", "cell_state", "vasculature"}
	for path in exported["0_0"].values():
		assert Path(path).exists()

	vasculature = np.asarray(Image.open(exported["0_0"]["vasculature"]))
	assert tuple(vasculature[0, 0]) == (0, 0, 0)


def test_make_metabolic_gradient_tiles_renders_processed_oxygen_and_glucose(
	tmp_path: Path,
) -> None:
	processed = tmp_path / "processed"
	(processed / "oxygen").mkdir(parents=True)
	(processed / "glucose").mkdir(parents=True)

	oxygen = np.zeros((8, 8, 4), dtype=np.uint8)
	oxygen[:, :, :3] = np.array([0, 255, 255], dtype=np.uint8)
	oxygen[:, :, 3] = 255
	glucose = np.zeros((8, 8, 4), dtype=np.uint8)
	glucose[:, :, :3] = np.array([255, 242, 30], dtype=np.uint8)
	glucose[:, :, 3] = 255
	Image.fromarray(oxygen, mode="RGBA").save(processed / "oxygen" / "0_0.png")
	Image.fromarray(glucose, mode="RGBA").save(processed / "glucose" / "0_0.png")

	Image.fromarray(oxygen, mode="RGBA").save(processed / "oxygen" / "1_0.png")
	Image.fromarray(glucose, mode="RGBA").save(processed / "glucose" / "1_0.png")

	out_path = tmp_path / "metabolic.png"
	canvas = make_metabolic_gradient_tiles(
		processed,
		["0_0", "1_0"],
		out_path,
		tile_size=8,
		panel_gap=2,
		header_height=6,
		mpp=0.325,
		scale_bar_um=0.0,
	)
	arr = np.asarray(canvas)

	assert out_path.exists()
	assert canvas.size == (70, 24)
	assert tuple(arr[11, 3]) == (0, 255, 255)
	assert tuple(arr[11, 39]) == (255, 242, 30)
	assert arr[12, 20, 1] > 0
	assert arr[12, 56, 0] > 0


def test_export_metabolic_tiles_writes_expected_stage1_tile_files(tmp_path: Path) -> None:
	processed = tmp_path / "processed"
	(processed / "oxygen").mkdir(parents=True)
	(processed / "glucose").mkdir(parents=True)

	oxygen = np.zeros((8, 8, 4), dtype=np.uint8)
	oxygen[:, :, :3] = np.array([0, 255, 255], dtype=np.uint8)
	oxygen[:, :, 3] = 255
	glucose = np.zeros((8, 8, 4), dtype=np.uint8)
	glucose[:, :, :3] = np.array([255, 242, 30], dtype=np.uint8)
	glucose[:, :, 3] = 255
	Image.fromarray(oxygen, mode="RGBA").save(processed / "oxygen" / "0_0.png")
	Image.fromarray(glucose, mode="RGBA").save(processed / "glucose" / "0_0.png")

	exported = export_metabolic_tiles(
		processed,
		["0_0"],
		tmp_path / "paper/figures/stage1/tiles",
		group_id="g1",
	)

	assert set(exported["0_0"]) == {"oxygen", "glucose"}
	for path in exported["0_0"].values():
		assert Path(path).exists()

	assert tuple(np.asarray(Image.open(exported["0_0"]["oxygen"]))[0, 0]) == (0, 255, 255)
	assert tuple(np.asarray(Image.open(exported["0_0"]["glucose"]))[0, 0]) == (255, 242, 30)


def test_select_top_can_require_paired_generated_he() -> None:
	records = [
		{
			"patch_id": "a",
			"g1_score": 0.99,
			"g1_min_rank": 0.99,
			"has_paired_generated_he": False,
		},
		{
			"patch_id": "b",
			"g1_score": 0.95,
			"g1_min_rank": 0.95,
			"has_paired_generated_he": True,
		},
	]

	selected = _select_top(
		records,
		{"id": "g1"},
		n=1,
		used=set(),
		require_generated_he=True,
	)

	assert [rec["patch_id"] for rec in selected] == ["b"]


def test_draw_crop_boxes_adds_red_and_white_callouts() -> None:
	rgb = np.zeros((64, 64, 3), dtype=np.uint8)
	boxed = _draw_crop_boxes(
		rgb,
		[
			{
				"he_x0": 24,
				"he_y0": 24,
				"he_size": 12,
			}
		],
		base_w=64,
		base_h=64,
		x_key="he_x0",
		y_key="he_y0",
		size_key="he_size",
		min_box_px=12,
		line_width=3,
	)

	assert np.any(np.all(boxed == np.array([255, 0, 0], dtype=np.uint8), axis=-1))
	assert np.any(np.all(boxed == np.array([255, 255, 255], dtype=np.uint8), axis=-1))
