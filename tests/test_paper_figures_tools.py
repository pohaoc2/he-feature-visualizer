from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.paper_figures.build_marker_zoom_figure import _format_distance_label
from tools.paper_figures.select_marker_patch_examples import (
	_header_label_centers,
	make_metabolic_gradient_tiles,
)
from tools.paper_figures.vis_he_mx_side_by_side import _draw_crop_boxes
from tools.paper_figures.vis_cell_feature_tiles import (
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
