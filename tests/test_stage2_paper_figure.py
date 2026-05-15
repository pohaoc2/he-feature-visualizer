from pathlib import Path

import json

import numpy as np
import pytest
from PIL import Image

from tools.paper.figures.stage2.assets import (
    Stage2FigureAssets,
    pick_representative_patch,
    resolve_stage2_assets,
    selected_patch_ids,
)
from tools.paper.figures.stage2.build_training_figure import (
    build_stage2_training_figure,
    export_stage2_tiles,
    export_selected_stage2_tiles,
)
from tools.paper.figures.stage2.render import render_stage2_training_figure
from tools.paper.figures.stage2.render import _compute_layout


def _make_stage2_assets(tmp_path: Path) -> Stage2FigureAssets:
    def save_rgb(path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(path)

    assets = Stage2FigureAssets(
        patch_id="23296_21760",
        group_id="g1",
        cell_mask_path=tmp_path / "cell_mask.png",
        reference_he_path=tmp_path / "he.png",
        cell_type_path=tmp_path / "type.png",
        cell_state_path=tmp_path / "state.png",
        vasculature_path=tmp_path / "vas.png",
        oxygen_path=tmp_path / "oxygen.png",
        glucose_path=tmp_path / "glucose.png",
        generated_he_path=tmp_path / "generated.png",
    )
    save_rgb(assets.cell_mask_path, (255, 255, 255))
    save_rgb(assets.reference_he_path, (240, 200, 210))
    save_rgb(assets.cell_type_path, (210, 235, 245))
    save_rgb(assets.cell_state_path, (225, 245, 230))
    save_rgb(assets.vasculature_path, (180, 40, 40))
    save_rgb(assets.oxygen_path, (0, 255, 255))
    save_rgb(assets.glucose_path, (255, 240, 40))
    save_rgb(assets.generated_he_path, (235, 190, 205))
    return assets


def _write_stage2_input_tree(
    tmp_path: Path,
    patch_ids: list[str],
) -> tuple[Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    pixcell_root = tmp_path / "pixcell"
    selection_json = tmp_path / "selections.json"

    groups: dict[str, dict[str, list[dict[str, str]]]] = {"g1": {"selections": []}}
    for patch_id in patch_ids:
        groups["g1"]["selections"].append({"patch_id": patch_id})
        for subdir, color in [
            ("cell_masks", (255, 255, 255)),
            ("he", (240, 200, 210)),
            ("cell_types/union", (210, 235, 245)),
            ("cell_states/union", (225, 245, 230)),
            ("vasculature", (180, 40, 40)),
            ("oxygen", (0, 255, 255)),
            ("glucose", (255, 240, 40)),
        ]:
            path = processed_dir / subdir / f"{patch_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color).save(path)

    selection_json.write_text(
        json.dumps({"groups": groups}),
        encoding="utf-8",
    )
    return processed_dir, pixcell_root, selection_json


def _count_components(mask: np.ndarray) -> int:
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    height, width = mask.shape

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            components += 1
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
    return components


def test_stage2_package_and_mockups_exist() -> None:
    repo = Path(__file__).resolve().parents[1]

    assert (repo / "tools/paper/figures/stage2/__init__.py").exists()
    assert (repo / "paper/figures/stage2/stage2_pipeline_designs.html").exists()
    assert (repo / "paper/figures/stage2/stage2_pipeline_design_A_mod.html").exists()


def test_stage2_pipeline_redesign_svg_structure() -> None:
    """Validate the Stage 2 HTML matches the documented training architecture."""
    repo = Path(__file__).resolve().parents[1]
    html = (repo / "paper/figures/stage2/stage2_pipeline_design_A_mod.html").read_text()

    assert 'viewBox="0 0 660 360"' in html, "viewBox must match the updated training layout"

    assert "Stage 2 — Training Architecture" in html
    assert "Multi-Group TME Module" in html
    assert "PixCell ControlNet" in html
    assert "Base PixArt-256" in html
    assert "SD3.5 VAE enc" in html
    assert "cell mask" in html
    assert "zero_mask_latent" in html
    assert "Q tokens" in html
    assert "residuals ΣΔgroup" in html
    assert "fused = tme(vae_mask) - vae_mask = Σ(Δgroup)" in html
    assert "Diffusion loss: MSE(eps_pred, eps)" in html

    assert "trainable, 27 blocks" in html
    assert "frozen, 28 blocks" in html
    assert "UNI latent [B,1536]" in html
    assert "Stage 1 TME example tiles" in html

    assert "generated H&amp;E" not in html
    assert "v-loss" not in html
    assert "PixCell denoiser" not in html


def test_pick_representative_patch_uses_first_matching_selection(tmp_path: Path) -> None:
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    pixcell_root.mkdir()

    payload = {
        "groups": {
            "g1": {
                "selections": [
                    {"patch_id": "100_100"},
                    {"patch_id": "200_200"},
                ]
            },
            "g2": {
                "selections": [
                    {"patch_id": "300_300"},
                ]
            },
        }
    }
    selection_json.write_text(json.dumps(payload), encoding="utf-8")

    match = pixcell_root / "200_200" / "all"
    match.mkdir(parents=True)
    (match / "generated_he.png").write_bytes(b"fake")

    result = pick_representative_patch(selection_json, pixcell_root)

    assert result.patch_id == "200_200"
    assert result.group_id == "g1"


def test_selected_patch_ids_can_filter_by_group_and_limit(tmp_path: Path) -> None:
    selection_json = tmp_path / "selections.json"
    selection_json.write_text(
        json.dumps(
            {
                "groups": {
                    "g1": {"selections": [{"patch_id": "100_100"}, {"patch_id": "200_200"}]},
                    "g2": {"selections": [{"patch_id": "300_300"}]},
                }
            }
        ),
        encoding="utf-8",
    )

    assert selected_patch_ids(selection_json, group_id="g1", n=1) == [("g1", "100_100")]


def test_pick_representative_patch_raises_with_checked_ids(tmp_path: Path) -> None:
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    pixcell_root.mkdir()
    selection_json.write_text(
        json.dumps(
            {
                "groups": {
                    "g1": {
                        "selections": [
                            {"patch_id": "100_100"},
                            {"patch_id": "200_200"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as exc:
        pick_representative_patch(selection_json, pixcell_root)

    message = str(exc.value)
    assert "100_100" in message
    assert "200_200" in message


def test_pick_representative_patch_raises_for_malformed_manifest(tmp_path: Path) -> None:
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    pixcell_root.mkdir()
    selection_json.write_text('{"groups": {"g1": {"selections": [{}]}}}', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        pick_representative_patch(selection_json, pixcell_root)

    message = str(exc.value)
    assert str(selection_json) in message
    assert "patch_id" in message


def test_resolve_stage2_assets_returns_expected_paths(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    patch_id = "23296_21760"

    selection_json.write_text(
        json.dumps({"groups": {"g1": {"selections": [{"patch_id": patch_id}]}}}),
        encoding="utf-8",
    )

    required_paths = [
        processed_dir / "cell_masks" / f"{patch_id}.png",
        processed_dir / "he" / f"{patch_id}.png",
        processed_dir / "cell_types/union" / f"{patch_id}.png",
        processed_dir / "cell_states/union" / f"{patch_id}.png",
        processed_dir / "vasculature" / f"{patch_id}.png",
        processed_dir / "oxygen" / f"{patch_id}.png",
        processed_dir / "glucose" / f"{patch_id}.png",
        pixcell_root / patch_id / "all" / "generated_he.png",
    ]
    for path in required_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assets = resolve_stage2_assets(processed_dir, selection_json, pixcell_root)

    assert assets.patch_id == patch_id
    assert assets.group_id == "g1"
    assert assets.cell_mask_path == processed_dir / "cell_masks" / f"{patch_id}.png"
    assert assets.reference_he_path == processed_dir / "he" / f"{patch_id}.png"
    assert assets.generated_he_path == pixcell_root / patch_id / "all" / "generated_he.png"


def test_resolve_stage2_assets_raises_for_missing_local_asset(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    patch_id = "23296_21760"

    selection_json.write_text(
        json.dumps({"groups": {"g1": {"selections": [{"patch_id": patch_id}]}}}),
        encoding="utf-8",
    )

    existing_paths = [
        processed_dir / "cell_masks" / f"{patch_id}.png",
        processed_dir / "he" / f"{patch_id}.png",
        processed_dir / "cell_types/union" / f"{patch_id}.png",
        processed_dir / "cell_states/union" / f"{patch_id}.png",
        processed_dir / "vasculature" / f"{patch_id}.png",
        processed_dir / "oxygen" / f"{patch_id}.png",
        pixcell_root / patch_id / "all" / "generated_he.png",
    ]
    for path in existing_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    with pytest.raises(FileNotFoundError) as exc:
        resolve_stage2_assets(processed_dir, selection_json, pixcell_root)

    assert "glucose" in str(exc.value)


def test_export_stage2_tiles_writes_requested_pngs(tmp_path: Path) -> None:
    assets = _make_stage2_assets(tmp_path)

    exported = export_stage2_tiles(assets, tmp_path / "paper/figures/stage2/tiles")

    assert set(exported) == {"cell_mask", "reference_he", "generated_he"}
    for path in exported.values():
        assert Path(path).exists()


def test_export_selected_stage2_tiles_writes_all_group_tiles(tmp_path: Path) -> None:
    processed_dir, pixcell_root, selection_json = _write_stage2_input_tree(
        tmp_path,
        ["100_100", "200_200"],
    )
    for patch_id in ["100_100", "200_200"]:
        generated_dir = pixcell_root / patch_id / "all"
        generated_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), (235, 190, 205)).save(
            generated_dir / "generated_he.png"
        )

    exported, missing = export_selected_stage2_tiles(
        processed_dir,
        selection_json,
        pixcell_root,
        tmp_path / "paper/figures/stage2/tiles",
        group_id="g1",
    )

    assert missing == {}
    assert set(exported) == {"100_100", "200_200"}
    for patch_exports in exported.values():
        assert set(patch_exports) == {"cell_mask", "reference_he", "generated_he"}
        for path in patch_exports.values():
            assert Path(path).exists()


def test_render_stage2_training_figure_returns_nonempty_canvas(tmp_path: Path) -> None:
    assets = _make_stage2_assets(tmp_path)

    canvas = render_stage2_training_figure(assets, tile_size=64)

    arr = np.asarray(canvas)
    assert 350 < canvas.width < 650
    assert canvas.height > 250
    assert arr.sum() > 0
    assert np.any(np.all(arr == np.array([0, 255, 255], dtype=np.uint8), axis=-1))
    assert np.any(np.all(arr == np.array([255, 240, 40], dtype=np.uint8), axis=-1))


def test_render_stage2_training_figure_draws_trainable_and_frozen_regions(
    tmp_path: Path,
) -> None:
    assets = _make_stage2_assets(tmp_path)

    arr = np.asarray(render_stage2_training_figure(assets, tile_size=64))
    layout = _compute_layout(tile_size=64, panel_gap=16, header_height=34)

    greenish = (arr[:, :, 1] > arr[:, :, 0] + 10) & (arr[:, :, 1] > arr[:, :, 2] + 5)
    striped = (
        (np.abs(arr[:, :, 0] - arr[:, :, 1]) < 8)
        & (np.abs(arr[:, :, 1] - arr[:, :, 2]) < 8)
        & (arr[:, :, 0] > 180)
        & (arr[:, :, 0] < 245)
    )

    assert greenish.any()
    assert striped.any()

    trainable_fill = np.all(arr == np.array([223, 244, 227], dtype=np.uint8), axis=-1)
    assert _count_components(trainable_fill) >= 5

    for box in [*layout["cnn_boxes"], layout["attention_box"]]:
        x0, y0, x1, y1 = box
        region = arr[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4]
        assert np.any(np.all(region == np.array([223, 244, 227], dtype=np.uint8), axis=-1))

    for box in [layout["uni_box"], layout["denoiser_box"], layout["vae_box"]]:
        x0, y0, x1, y1 = box
        region = arr[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4]
        assert np.any(np.all(region == np.array([247, 244, 239], dtype=np.uint8), axis=-1))
        assert np.any(np.all(region == np.array([198, 192, 183], dtype=np.uint8), axis=-1))


def test_render_stage2_training_figure_places_key_visual_elements(tmp_path: Path) -> None:
    assets = _make_stage2_assets(tmp_path)

    arr = np.asarray(render_stage2_training_figure(assets, tile_size=64))
    layout = _compute_layout(tile_size=64, panel_gap=16, header_height=34)
    height, width = arr.shape[:2]

    upper_left = arr[: height // 3, : width // 4]
    lower_right = arr[height // 2 :, width * 3 // 4 :]
    right_half = arr[:, width // 2 :]

    he_ref = np.array([240, 200, 210], dtype=np.uint8)
    generated_he = np.array([235, 190, 205], dtype=np.uint8)
    loss_color = np.array([179, 59, 52], dtype=np.uint8)

    assert np.any(np.all(upper_left == he_ref, axis=-1))
    assert np.any(np.all(lower_right == generated_he, axis=-1))
    assert np.any(np.all(arr == loss_color, axis=-1))

    latent_fill = np.array([238, 242, 249], dtype=np.uint8)
    assert not np.any(np.all(right_half == latent_fill, axis=-1))

    for box in [layout["noisy_box"], layout["latent_box"]]:
        x0, y0, x1, y1 = box
        region = arr[y0:y1, x0:x1]
        assert np.any(np.any(region != np.array([255, 253, 249], dtype=np.uint8), axis=-1))

    noisy_box = layout["noisy_box"]
    denoiser_box = layout["denoiser_box"]
    assert noisy_box[3] <= denoiser_box[1]
    assert noisy_box[0] > width // 2
    tme_boxes = layout["tme_boxes"]
    assert len({box[0] for box in tme_boxes}) == 1
    assert all(tme_boxes[index][1] < tme_boxes[index + 1][1] for index in range(3))
    latent_box = layout["latent_box"]
    vae_box = layout["vae_box"]
    assert vae_box[1] > latent_box[3]

    loss_points = layout["loss_points"]
    loss_y = loss_points[1][1]
    x_start = loss_points[2][0]
    x_end = loss_points[1][0]
    horizontal_loss = arr[loss_y - 1 : loss_y + 2, x_start + 8 : x_end - 8]
    loss_mask = np.all(horizontal_loss == loss_color, axis=-1)
    assert loss_mask.any()
    assert (~loss_mask).any()
    assert height > layout["cnn_boxes"][-1][3] + 4

    generated_box = layout["generated_box"]
    label_band = arr[
        generated_box[3] : generated_box[3] + 18,
        max(0, generated_box[0] - 8) : min(width, generated_box[2] + 8),
    ]
    assert not np.any(np.all(label_band == loss_color, axis=-1))


def test_build_stage2_training_figure_writes_png_and_json(tmp_path: Path) -> None:
    patch_id = "23296_21760"
    processed_dir, pixcell_root, selection_json = _write_stage2_input_tree(
        tmp_path,
        [patch_id],
    )
    output_dir = tmp_path / "paper/figures/stage2"

    generated_dir = pixcell_root / patch_id / "all"
    generated_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (235, 190, 205)).save(generated_dir / "generated_he.png")

    png_path, json_path = build_stage2_training_figure(
        processed_dir=processed_dir,
        selection_json=selection_json,
        pixcell_root=pixcell_root,
        out_dir=output_dir,
    )

    assert png_path == output_dir / "crc33_stage2_training_pipeline.png"
    assert json_path == output_dir / "crc33_stage2_training_pipeline.json"
    assert png_path.exists()
    assert json_path.exists()

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["patch_id"] == patch_id
    assert metadata["group_id"] == "g1"
    assert metadata["manifest_path"] == str(selection_json)
    assert metadata["pixcell_root"] == str(pixcell_root)
    assert metadata["generated_he_path"].endswith(f"{patch_id}/all/generated_he.png")
    assert metadata["source_asset_paths"]["cell_mask_path"] == str(
        processed_dir / "cell_masks" / f"{patch_id}.png"
    )
    assert metadata["tiles_dir"] == str(output_dir / "tiles")
    assert metadata["selection_rule"] == (
        "first stage1 selected patch with matching PixCell all/generated_he.png"
    )
    assert metadata["source_asset_paths"]["reference_he_path"] == str(
        processed_dir / "he" / f"{patch_id}.png"
    )
    assert metadata["exported_tiles"]["cell_mask"] == str(
        output_dir / "tiles" / f"{patch_id}_cell_mask.png"
    )
    assert metadata["selected_group_patch_ids"] == [patch_id]
    assert metadata["selected_group_exported_tiles"][patch_id]["generated_he"] == str(
        output_dir / "tiles" / f"{patch_id}_generated_he.png"
    )
    assert metadata["missing_selected_group_tiles"] == {}
    assert Path(metadata["exported_tiles"]["reference_he"]).exists()
    assert Path(metadata["exported_tiles"]["generated_he"]).exists()
    with Image.open(png_path) as image:
        canvas_size = list(image.size)
    from tools.paper.figures.stage2.build_training_figure import (
        DEFAULT_TILE_SIZE,
        DEFAULT_PANEL_GAP,
        DEFAULT_HEADER_HEIGHT,
    )
    assert metadata["layout_parameters"] == {
        "tile_size_px": DEFAULT_TILE_SIZE,
        "panel_gap_px": DEFAULT_PANEL_GAP,
        "header_height_px": DEFAULT_HEADER_HEIGHT,
        "canvas_size_px": canvas_size,
    }


def test_build_stage2_training_figure_honors_patch_id_override(tmp_path: Path) -> None:
    processed_dir, pixcell_root, selection_json = _write_stage2_input_tree(
        tmp_path,
        ["100_100", "200_200"],
    )
    forced_patch_id = "200_200"

    first_generated_dir = pixcell_root / "100_100" / "all"
    first_generated_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (235, 190, 205)).save(
        first_generated_dir / "generated_he.png"
    )

    forced_generated_dir = pixcell_root / forced_patch_id / "all"
    forced_generated_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (235, 190, 205)).save(
        forced_generated_dir / "generated_he.png"
    )

    _, json_path = build_stage2_training_figure(
        processed_dir=processed_dir,
        selection_json=selection_json,
        pixcell_root=pixcell_root,
        out_dir=tmp_path / "paper/figures/stage2",
        patch_id=forced_patch_id,
    )

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["patch_id"] == forced_patch_id
    assert metadata["group_id"] == "g1"
    assert metadata["selection_rule"] == f"forced patch_id override: {forced_patch_id}"