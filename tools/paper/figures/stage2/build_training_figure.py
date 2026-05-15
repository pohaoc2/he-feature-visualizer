"""Build the stage-2 training paper figure and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .assets import Stage2FigureAssets, resolve_stage2_assets, selected_patch_ids
from .render import render_stage2_training_figure


DEFAULT_PROCESSED_DIR = Path("processed_crc33")
DEFAULT_SELECTION_JSON = Path(
    "paper/figures/stage1/crc33_marker_high_patch_examples_selections.json"
)
DEFAULT_PIXCELL_ROOT = Path(
    "/home/pohaoc2/UW/bagherilab/PixCell/inference_output/paired_ablation/ablation_results"
)
DEFAULT_OUT_DIR = Path("paper/figures/stage2")
DEFAULT_TILE_SIZE = 200
DEFAULT_PANEL_GAP = 22
DEFAULT_HEADER_HEIGHT = 56
DEFAULT_SCALE_BAR_LABEL = "67 µm"
OUTPUT_PNG_NAME = "crc33_stage2_training_pipeline.png"
OUTPUT_JSON_NAME = "crc33_stage2_training_pipeline.json"


def _selection_rule(patch_id: str | None) -> str:
    if patch_id is not None:
        return f"forced patch_id override: {patch_id}"
    return "first stage1 selected patch with matching PixCell all/generated_he.png"


def _source_asset_paths(assets: Stage2FigureAssets) -> dict[str, str]:
    return {
        "cell_mask_path": str(assets.cell_mask_path),
        "reference_he_path": str(assets.reference_he_path),
        "cell_type_path": str(assets.cell_type_path),
        "cell_state_path": str(assets.cell_state_path),
        "vasculature_path": str(assets.vasculature_path),
        "oxygen_path": str(assets.oxygen_path),
        "glucose_path": str(assets.glucose_path),
        "generated_he_path": str(assets.generated_he_path),
    }


def export_stage2_tiles(assets: Stage2FigureAssets, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_map = {
        "cell_mask": assets.cell_mask_path,
        "reference_he": assets.reference_he_path,
        "generated_he": assets.generated_he_path,
    }
    exported: dict[str, str] = {}
    for label, source_path in export_map.items():
        out_path = out_dir / f"{assets.patch_id}_{label}.png"
        with Image.open(source_path) as image:
            image.save(out_path)
        exported[label] = str(out_path)
    return exported


def export_selected_stage2_tiles(
    processed_dir: Path,
    selection_json: Path,
    pixcell_root: Path,
    out_dir: Path,
    group_id: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    exported: dict[str, dict[str, str]] = {}
    missing: dict[str, str] = {}
    for _, patch_id in selected_patch_ids(selection_json, group_id=group_id):
        try:
            assets = resolve_stage2_assets(
                processed_dir,
                selection_json,
                pixcell_root,
                patch_id=patch_id,
            )
        except FileNotFoundError as exc:
            missing[patch_id] = str(exc)
            continue
        exported[patch_id] = export_stage2_tiles(assets, out_dir)
    return exported, missing


def build_stage2_training_figure(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    selection_json: Path = DEFAULT_SELECTION_JSON,
    pixcell_root: Path = DEFAULT_PIXCELL_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    patch_id: str | None = None,
    group_id: str | None = None,
    scale_bar_label: str = DEFAULT_SCALE_BAR_LABEL,
) -> tuple[Path, Path]:
    assets = resolve_stage2_assets(
        Path(processed_dir),
        Path(selection_json),
        Path(pixcell_root),
        patch_id=patch_id,
        group_id=group_id,
    )
    canvas = render_stage2_training_figure(
        assets,
        tile_size=DEFAULT_TILE_SIZE,
        panel_gap=DEFAULT_PANEL_GAP,
        header_height=DEFAULT_HEADER_HEIGHT,
        scale_bar_label=scale_bar_label,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / OUTPUT_PNG_NAME
    out_json = out_dir / OUTPUT_JSON_NAME
    canvas.save(out_png)
    tiles_dir = out_dir / "tiles"
    exported_tiles = export_stage2_tiles(assets, tiles_dir)
    selected_group_id = str(group_id or assets.group_id)
    selected_group_tiles, missing_selected_group_tiles = export_selected_stage2_tiles(
        Path(processed_dir),
        Path(selection_json),
        Path(pixcell_root),
        tiles_dir,
        group_id=selected_group_id,
    )

    metadata = {
        "patch_id": assets.patch_id,
        "group_id": assets.group_id,
        "selected_group_id": selected_group_id,
        "selected_group_patch_ids": [
            patch_id for _, patch_id in selected_patch_ids(selection_json, group_id=selected_group_id)
        ],
        "manifest_path": str(selection_json),
        "pixcell_root": str(pixcell_root),
        "generated_he_path": str(assets.generated_he_path),
        "source_asset_paths": _source_asset_paths(assets),
        "exported_tiles": exported_tiles,
        "selected_group_exported_tiles": selected_group_tiles,
        "missing_selected_group_tiles": missing_selected_group_tiles,
        "tiles_dir": str(tiles_dir),
        "selection_rule": _selection_rule(patch_id),
        "layout_parameters": {
            "tile_size_px": DEFAULT_TILE_SIZE,
            "panel_gap_px": DEFAULT_PANEL_GAP,
            "header_height_px": DEFAULT_HEADER_HEIGHT,
            "canvas_size_px": list(canvas.size),
        },
        "output_png": str(out_png),
        "output_json": str(out_json),
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_png, out_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    parser.add_argument("--pixcell-root", type=Path, default=DEFAULT_PIXCELL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--patch-id", default=None)
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--scale-bar-label", default=DEFAULT_SCALE_BAR_LABEL)
    args = parser.parse_args(argv)

    out_png, out_json = build_stage2_training_figure(
        processed_dir=args.processed_dir,
        selection_json=args.selection_json,
        pixcell_root=args.pixcell_root,
        out_dir=args.out_dir,
        patch_id=args.patch_id,
        group_id=args.group_id,
        scale_bar_label=args.scale_bar_label,
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()