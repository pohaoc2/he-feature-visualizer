"""Build a stage-3 figure overlaying generated H&E with CellViT contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_RESULT_DIR = Path(
    "/home/pohaoc2/UW/bagherilab/PixCell/inference_output/unpaired_ablation/ablation_results/14592_5632"
)
DEFAULT_OUT_DIR = Path("paper/figures/stage3")
DEFAULT_CONTOUR_OUTLINE_RGB = (0, 0, 0)
DEFAULT_CONTOUR_INNER_RGB = (255, 255, 255)
DEFAULT_OUTLINE_WIDTH = 2
DEFAULT_INNER_WIDTH = 1
DEFAULT_SOURCE_SIZE = 256.0


def _generated_image_path(result_dir: Path) -> Path:
    return result_dir / "all" / "generated_he.png"


def _cellvit_json_path(result_dir: Path) -> Path:
    return result_dir / "all" / "generated_he_cellvit_instances.json"


def _result_label(result_dir: Path) -> str:
    for parent in result_dir.parents:
        name = parent.name.lower()
        if name in {"paired_ablation", "unpaired_ablation"}:
            return name.replace("_ablation", "")
    return result_dir.parent.name or "result"


def _default_out_path(result_dir: Path, out_dir: Path, label: str | None = None) -> Path:
    prefix = label or _result_label(result_dir)
    return out_dir / f"{prefix}_{result_dir.name}_generated_he_cellvit_overlay.png"


def _load_cells(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        for key in ("cells", "nuclei", "detections", "instances"):
            value = data.get(key)
            if isinstance(value, list):
                return [cell for cell in value if isinstance(cell, dict)]
        values = list(data.values())
        if values and all(isinstance(cell, dict) for cell in values):
            return values
        return []

    if isinstance(data, list):
        return [cell for cell in data if isinstance(cell, dict)]

    return []


def overlay_cellvit_contours(
    generated_he_path: Path,
    cellvit_json_path: Path,
    out_path: Path,
    *,
    outline_rgb: tuple[int, int, int] = DEFAULT_CONTOUR_OUTLINE_RGB,
    inner_rgb: tuple[int, int, int] = DEFAULT_CONTOUR_INNER_RGB,
    outline_width: int = DEFAULT_OUTLINE_WIDTH,
    inner_width: int = DEFAULT_INNER_WIDTH,
    source_size: float = DEFAULT_SOURCE_SIZE,
) -> Path:
    if not generated_he_path.exists():
        raise FileNotFoundError(f"generated H&E image not found: {generated_he_path}")
    if not cellvit_json_path.exists():
        raise FileNotFoundError(f"CellViT JSON not found: {cellvit_json_path}")

    cells = _load_cells(cellvit_json_path)
    image = Image.open(generated_he_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x = image.width / float(source_size) if image.width > 0 else 1.0
    scale_y = image.height / float(source_size) if image.height > 0 else 1.0

    for cell in cells:
        contour = cell.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        points = [
            (float(point[0]) * scale_x, float(point[1]) * scale_y)
            for point in contour
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(points) < 3:
            continue

        closed = points + [points[0]]
        if outline_width > 0:
            draw.line(closed, fill=outline_rgb, width=outline_width)
        draw.line(closed, fill=inner_rgb, width=inner_width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def build_unpaired_generated_overlay(
    result_dir: Path = DEFAULT_RESULT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    label: str | None = None,
) -> Path:
    return overlay_cellvit_contours(
        _generated_image_path(result_dir),
        _cellvit_json_path(result_dir),
        _default_out_path(result_dir, out_dir, label=label),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    out_path = build_unpaired_generated_overlay(
        result_dir=args.result_dir,
        out_dir=args.out_dir,
        label=args.label,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()