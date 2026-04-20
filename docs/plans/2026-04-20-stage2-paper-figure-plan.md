# Stage 2 Paper Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-rendered stage-2 training figure that auto-picks one representative stage-1-selected patch with a matching PixCell generated H&E, writes final PNG/JSON outputs to `paper/figures/stage2`, and relocates the existing HTML mockups into the same stage-2 output directory.

**Architecture:** Create a focused `tools/paper/figures/stage2` package with separate modules for asset resolution, rendering, and CLI orchestration. Reuse stage-1 tile-loading and visual idioms where practical, but keep stage-2 layout logic independent so it can evolve without coupling to the HTML mockups.

**Tech Stack:** Python 3.13, Pillow, pathlib, json, pytest

---

## File Structure

- Create: `tools/paper/figures/stage2/__init__.py`
- Create: `tools/paper/figures/stage2/assets.py`
- Create: `tools/paper/figures/stage2/render.py`
- Create: `tools/paper/figures/stage2/build_training_figure.py`
- Create: `tests/test_stage2_paper_figure.py`
- Modify: `paper/figures/stage2_pipeline_designs.html` (move to `paper/figures/stage2/stage2_pipeline_designs.html`)
- Modify: `paper/figures/stage2_pipeline_design_A_mod.html` (move to `paper/figures/stage2/stage2_pipeline_design_A_mod.html`)

Implementation notes:

- `assets.py` owns patch selection, PixCell output discovery, and local asset-path resolution.
- `render.py` owns the publication layout drawing using PIL only.
- `build_training_figure.py` owns CLI defaults, output directory creation, PNG saving, and JSON metadata emission.
- `tests/test_stage2_paper_figure.py` covers resolver determinism, missing-match errors, metadata content, and end-to-end synthetic rendering.

### Task 1: Scaffold Stage-2 Package And Relocate Mockups

**Files:**
- Create: `tools/paper/figures/stage2/__init__.py`
- Modify: `paper/figures/stage2_pipeline_designs.html`
- Modify: `paper/figures/stage2_pipeline_design_A_mod.html`

- [ ] **Step 1: Write the failing relocation/package test**

```python
from pathlib import Path


def test_stage2_package_and_mockups_exist() -> None:
    repo = Path(__file__).resolve().parents[1]

    assert (repo / "tools/paper/figures/stage2/__init__.py").exists()
    assert (repo / "paper/figures/stage2/stage2_pipeline_designs.html").exists()
    assert (repo / "paper/figures/stage2/stage2_pipeline_design_A_mod.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_paper_figure.py::test_stage2_package_and_mockups_exist -v`
Expected: FAIL because the stage-2 package and relocated HTML files do not exist yet.

- [ ] **Step 3: Add the stage-2 package and move the mockup files**

```python
# tools/paper/figures/stage2/__init__.py
"""Stage-2 paper figure generation tools."""
```

```text
Move:
paper/figures/stage2_pipeline_designs.html
-> paper/figures/stage2/stage2_pipeline_designs.html

Move:
paper/figures/stage2_pipeline_design_A_mod.html
-> paper/figures/stage2/stage2_pipeline_design_A_mod.html
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2_paper_figure.py::test_stage2_package_and_mockups_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/paper/figures/stage2/__init__.py \
        paper/figures/stage2/stage2_pipeline_designs.html \
        paper/figures/stage2/stage2_pipeline_design_A_mod.html \
        tests/test_stage2_paper_figure.py
git commit -m "feat: scaffold stage2 figure package"
```

### Task 2: Implement Representative Patch Resolution And Asset Discovery

**Files:**
- Create: `tools/paper/figures/stage2/assets.py`
- Test: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Write the failing resolver tests**

```python
from pathlib import Path

import json
import pytest

from tools.paper.figures.stage2.assets import pick_representative_patch


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


def test_pick_representative_patch_raises_with_checked_ids(tmp_path: Path) -> None:
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    pixcell_root.mkdir()
    selection_json.write_text(
        json.dumps({
            "groups": {
                "g1": {"selections": [{"patch_id": "100_100"}, {"patch_id": "200_200"}]}
            }
        }),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as exc:
        pick_representative_patch(selection_json, pixcell_root)

    message = str(exc.value)
    assert "100_100" in message
    assert "200_200" in message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stage2_paper_figure.py::test_pick_representative_patch_uses_first_matching_selection tests/test_stage2_paper_figure.py::test_pick_representative_patch_raises_with_checked_ids -v`
Expected: FAIL with import or missing symbol errors because `assets.py` does not exist yet.

- [ ] **Step 3: Write minimal asset-resolution implementation**

```python
# tools/paper/figures/stage2/assets.py
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RepresentativePatch:
    group_id: str
    patch_id: str
    generated_he_path: Path


def _iter_selected_patch_ids(selection_json: Path) -> list[tuple[str, str]]:
    with selection_json.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    ordered: list[tuple[str, str]] = []
    for group_id, group in payload.get("groups", {}).items():
        for selection in group.get("selections", []):
            ordered.append((str(group_id), str(selection["patch_id"])))
    return ordered


def _generated_he_path(pixcell_root: Path, patch_id: str) -> Path:
    return pixcell_root / patch_id / "all" / "generated_he.png"


def pick_representative_patch(selection_json: Path, pixcell_root: Path) -> RepresentativePatch:
    checked: list[str] = []
    for group_id, patch_id in _iter_selected_patch_ids(selection_json):
        checked.append(patch_id)
        generated = _generated_he_path(pixcell_root, patch_id)
        if generated.exists():
            return RepresentativePatch(
                group_id=group_id,
                patch_id=patch_id,
                generated_he_path=generated,
            )

    raise FileNotFoundError(
        "No selected patch has PixCell generated output. Checked patch IDs: "
        + ", ".join(checked)
    )
```

- [ ] **Step 4: Extend implementation to resolve local stage-2 asset paths**

```python
@dataclass(frozen=True)
class Stage2FigureAssets:
    patch_id: str
    group_id: str
    reference_he_path: Path
    cell_type_path: Path
    cell_state_path: Path
    vasculature_path: Path
    oxygen_path: Path
    glucose_path: Path
    generated_he_path: Path


def resolve_stage2_assets(
    processed_dir: Path,
    selection_json: Path,
    pixcell_root: Path,
    patch_id: str | None = None,
) -> Stage2FigureAssets:
    match = (
        RepresentativePatch("forced", patch_id, _generated_he_path(pixcell_root, patch_id))
        if patch_id is not None
        else pick_representative_patch(selection_json, pixcell_root)
    )
    asset_paths = Stage2FigureAssets(
        patch_id=match.patch_id,
        group_id=match.group_id,
        reference_he_path=processed_dir / "he" / f"{match.patch_id}.png",
        cell_type_path=processed_dir / "cell_types/union" / f"{match.patch_id}.png",
        cell_state_path=processed_dir / "cell_states/union" / f"{match.patch_id}.png",
        vasculature_path=processed_dir / "vasculature" / f"{match.patch_id}.png",
        oxygen_path=processed_dir / "oxygen" / f"{match.patch_id}.png",
        glucose_path=processed_dir / "glucose" / f"{match.patch_id}.png",
        generated_he_path=match.generated_he_path,
    )
    for path in [
        asset_paths.reference_he_path,
        asset_paths.cell_type_path,
        asset_paths.cell_state_path,
        asset_paths.vasculature_path,
        asset_paths.oxygen_path,
        asset_paths.glucose_path,
        asset_paths.generated_he_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required stage2 asset: {path}")
    return asset_paths
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_stage2_paper_figure.py::test_pick_representative_patch_uses_first_matching_selection tests/test_stage2_paper_figure.py::test_pick_representative_patch_raises_with_checked_ids -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tools/paper/figures/stage2/assets.py tests/test_stage2_paper_figure.py
git commit -m "feat: resolve stage2 representative patch assets"
```

### Task 3: Implement Stage-2 Figure Rendering

**Files:**
- Create: `tools/paper/figures/stage2/render.py`
- Test: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Write the failing render smoke test**

```python
import numpy as np
from pathlib import Path
from PIL import Image

from tools.paper.figures.stage2.assets import Stage2FigureAssets
from tools.paper.figures.stage2.render import render_stage2_training_figure


def test_render_stage2_training_figure_returns_nonempty_canvas(tmp_path: Path) -> None:
    def save_rgb(path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(path)

    assets = Stage2FigureAssets(
        patch_id="23296_21760",
        group_id="g1",
        reference_he_path=tmp_path / "he.png",
        cell_type_path=tmp_path / "type.png",
        cell_state_path=tmp_path / "state.png",
        vasculature_path=tmp_path / "vas.png",
        oxygen_path=tmp_path / "oxygen.png",
        glucose_path=tmp_path / "glucose.png",
        generated_he_path=tmp_path / "generated.png",
    )
    save_rgb(assets.reference_he_path, (240, 200, 210))
    save_rgb(assets.cell_type_path, (210, 235, 245))
    save_rgb(assets.cell_state_path, (225, 245, 230))
    save_rgb(assets.vasculature_path, (40, 0, 0))
    save_rgb(assets.oxygen_path, (0, 255, 255))
    save_rgb(assets.glucose_path, (255, 240, 40))
    save_rgb(assets.generated_he_path, (235, 190, 205))

    canvas = render_stage2_training_figure(assets, tile_size=64)

    arr = np.asarray(canvas)
    assert canvas.width > 300
    assert canvas.height > 180
    assert arr.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_paper_figure.py::test_render_stage2_training_figure_returns_nonempty_canvas -v`
Expected: FAIL with import error because `render.py` and `render_stage2_training_figure` do not exist yet.

- [ ] **Step 3: Implement the rendering module with reusable helpers**

```python
# tools/paper/figures/stage2/render.py
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .assets import Stage2FigureAssets


def _open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _fit_tile(path: Path, tile_size: int) -> Image.Image:
    image = _open_rgb(path)
    if image.size == (tile_size, tile_size):
        return image
    return image.resize((tile_size, tile_size), Image.Resampling.NEAREST)


def render_stage2_training_figure(
    assets: Stage2FigureAssets,
    tile_size: int = 128,
    panel_gap: int = 10,
    header_height: int = 28,
) -> Image.Image:
    width = tile_size * 6 + panel_gap * 7
    height = tile_size * 2 + header_height + panel_gap * 5
    canvas = Image.new("RGB", (width, height), (255, 253, 249))
    draw = ImageDraw.Draw(canvas)

    x0 = panel_gap
    y_top = header_height + panel_gap
    y_bottom = y_top + tile_size + panel_gap * 2

    reference_he = _fit_tile(assets.reference_he_path, tile_size)
    generated_he = _fit_tile(assets.generated_he_path, tile_size)
    cell_type = _fit_tile(assets.cell_type_path, tile_size // 2)
    cell_state = _fit_tile(assets.cell_state_path, tile_size // 2)
    vasculature = _fit_tile(assets.vasculature_path, tile_size // 2)
    oxygen = _fit_tile(assets.oxygen_path, tile_size // 2)
    glucose = _fit_tile(assets.glucose_path, tile_size // 2)

    canvas.paste(reference_he, (x0, y_top))
    canvas.paste(cell_type, (x0, y_bottom))
    canvas.paste(cell_state, (x0 + tile_size // 2 + panel_gap, y_bottom))
    canvas.paste(vasculature, (x0 + tile_size + panel_gap * 2, y_bottom))
    canvas.paste(oxygen, (x0 + tile_size + tile_size // 2 + panel_gap * 3, y_bottom))
    canvas.paste(glucose, (x0 + tile_size * 2 + panel_gap * 4, y_bottom))

    denoiser_x = x0 + tile_size * 3 + panel_gap * 4
    denoiser_y = y_top + 4
    draw.rounded_rectangle(
        [denoiser_x, denoiser_y, denoiser_x + tile_size + 20, denoiser_y + 78],
        radius=10,
        fill=(245, 241, 233),
        outline=(142, 136, 128),
        width=2,
    )
    draw.text((denoiser_x + 20, denoiser_y + 18), "PixCell denoiser", fill=(36, 33, 29))
    draw.text((denoiser_x + 8, denoiser_y - 18), "noisy latent Z_t", fill=(51, 67, 93))
    draw.text((denoiser_x + tile_size + 40, denoiser_y + 8), "Denoised", fill=(51, 67, 93), anchor="mm")
    draw.text((denoiser_x + tile_size + 40, denoiser_y + 24), "latent Z", fill=(51, 67, 93), anchor="mm")

    vae_x = denoiser_x + tile_size + 56
    vae_y = denoiser_y + 96
    draw.rounded_rectangle(
        [vae_x, vae_y, vae_x + 60, vae_y + 54],
        radius=8,
        fill=(245, 241, 233),
        outline=(142, 136, 128),
        width=2,
    )
    draw.text((vae_x + 30, vae_y + 20), "SD3.5", fill=(36, 33, 29), anchor="mm")
    draw.text((vae_x + 30, vae_y + 34), "VAE", fill=(36, 33, 29), anchor="mm")
    canvas.paste(generated_he, (vae_x - 2, vae_y + 72))

    draw.line((x0 + tile_size, y_top + tile_size // 2, denoiser_x, y_top + tile_size // 2), fill=(89, 84, 79), width=2)
    draw.line((denoiser_x + tile_size + 20, denoiser_y + 55, vae_x, denoiser_y + 55), fill=(89, 84, 79), width=2)
    draw.line((vae_x + 30, vae_y + 54, vae_x + 30, vae_y + 72), fill=(89, 84, 79), width=2)
    draw.line((x0 + tile_size + 10, y_bottom + 12, denoiser_x, y_bottom + 12), fill=(45, 140, 79), width=2)
    draw.line((denoiser_x, y_bottom + 12, denoiser_x, denoiser_y + 70), fill=(45, 140, 79), width=2)

    return canvas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2_paper_figure.py::test_render_stage2_training_figure_returns_nonempty_canvas -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/paper/figures/stage2/render.py tests/test_stage2_paper_figure.py
git commit -m "feat: render stage2 training figure"
```

### Task 4: Implement Builder CLI, Metadata Output, And End-To-End Tests

**Files:**
- Create: `tools/paper/figures/stage2/build_training_figure.py`
- Test: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Write the failing end-to-end metadata test**

```python
import json
from pathlib import Path
from PIL import Image

from tools.paper.figures.stage2.build_training_figure import build_stage2_training_figure


def test_build_stage2_training_figure_writes_png_and_json(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    pixcell_root = tmp_path / "pixcell"
    output_dir = tmp_path / "paper/figures/stage2"
    selection_json = tmp_path / "selections.json"

    payload = {"groups": {"g1": {"selections": [{"patch_id": "23296_21760"}]}}}
    selection_json.write_text(json.dumps(payload), encoding="utf-8")

    for rel in [
        "he/23296_21760.png",
        "cell_types/union/23296_21760.png",
        "cell_states/union/23296_21760.png",
        "vasculature/23296_21760.png",
        "oxygen/23296_21760.png",
        "glucose/23296_21760.png",
    ]:
        path = processed / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (200, 200, 200)).save(path)

    generated = pixcell_root / "23296_21760" / "all"
    generated.mkdir(parents=True)
    Image.new("RGB", (64, 64), (220, 180, 190)).save(generated / "generated_he.png")

    png_path, json_path = build_stage2_training_figure(
        processed_dir=processed,
        selection_json=selection_json,
        pixcell_root=pixcell_root,
        out_dir=output_dir,
    )

    assert png_path.exists()
    assert json_path.exists()
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["patch_id"] == "23296_21760"
    assert metadata["generated_he_path"].endswith("23296_21760/all/generated_he.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_paper_figure.py::test_build_stage2_training_figure_writes_png_and_json -v`
Expected: FAIL with import error because the builder CLI does not exist yet.

- [ ] **Step 3: Implement the orchestration module and CLI**

```python
# tools/paper/figures/stage2/build_training_figure.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import resolve_stage2_assets
from .render import render_stage2_training_figure


DEFAULT_SELECTION_JSON = Path("paper/figures/stage1/crc33_marker_high_patch_examples_selections.json")
DEFAULT_PROCESSED_DIR = Path("processed_crc33")
DEFAULT_PIXCELL_ROOT = Path(
    "/home/pohaoc2/UW/bagherilab/PixCell/inference_output/paired_ablation/ablation_results"
)
DEFAULT_OUT_DIR = Path("paper/figures/stage2")


def build_stage2_training_figure(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    selection_json: Path = DEFAULT_SELECTION_JSON,
    pixcell_root: Path = DEFAULT_PIXCELL_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    patch_id: str | None = None,
) -> tuple[Path, Path]:
    assets = resolve_stage2_assets(processed_dir, selection_json, pixcell_root, patch_id=patch_id)
    canvas = render_stage2_training_figure(assets)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "crc33_stage2_training_pipeline.png"
    out_json = out_dir / "crc33_stage2_training_pipeline.json"
    canvas.save(out_png)

    metadata = {
        "patch_id": assets.patch_id,
        "group_id": assets.group_id,
        "selection_json": str(selection_json),
        "processed_dir": str(processed_dir),
        "pixcell_root": str(pixcell_root),
        "reference_he_path": str(assets.reference_he_path),
        "cell_type_path": str(assets.cell_type_path),
        "cell_state_path": str(assets.cell_state_path),
        "vasculature_path": str(assets.vasculature_path),
        "oxygen_path": str(assets.oxygen_path),
        "glucose_path": str(assets.glucose_path),
        "generated_he_path": str(assets.generated_he_path),
        "selection_rule": "first stage1 selected patch with matching PixCell all/generated_he.png",
        "output_png": str(out_png),
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_png, out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    parser.add_argument("--pixcell-root", type=Path, default=DEFAULT_PIXCELL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--patch-id", default=None)
    args = parser.parse_args()

    out_png, out_json = build_stage2_training_figure(
        processed_dir=args.processed_dir,
        selection_json=args.selection_json,
        pixcell_root=args.pixcell_root,
        out_dir=args.out_dir,
        patch_id=args.patch_id,
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `pytest tests/test_stage2_paper_figure.py -v`
Expected: PASS for package, resolver, render, and builder coverage.

- [ ] **Step 5: Run the builder with real defaults to produce the stage-2 figure**

Run: `python3 -m tools.paper.figures.stage2.build_training_figure`
Expected: writes `paper/figures/stage2/crc33_stage2_training_pipeline.png` and `paper/figures/stage2/crc33_stage2_training_pipeline.json`

- [ ] **Step 6: Commit**

```bash
git add tools/paper/figures/stage2/build_training_figure.py \
        tools/paper/figures/stage2/assets.py \
        tools/paper/figures/stage2/render.py \
        tests/test_stage2_paper_figure.py \
        paper/figures/stage2
git commit -m "feat: add stage2 training paper figure builder"
```

## Self-Review Checklist

- Spec coverage: this plan covers stage-2 package creation, output relocation, deterministic representative-patch selection, Python rendering, JSON metadata, failure cases, and synthetic tests.
- Placeholder scan: no `TBD`, `TODO`, or deferred implementation language remains.
- Type consistency: `RepresentativePatch`, `Stage2FigureAssets`, `pick_representative_patch`, `resolve_stage2_assets`, `render_stage2_training_figure`, and `build_stage2_training_figure` are named consistently across tasks and tests.

## Execution Handoff

The user already chose subagent-driven execution. Implement this plan with `superpowers:subagent-driven-development`, using one fresh implementation subagent per task and review gates between tasks.# Stage 2 Paper Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-rendered stage-2 training figure that auto-picks one representative stage-1-selected patch with a matching PixCell generated H&E, writes final PNG/JSON outputs to `paper/figures/stage2`, and relocates the existing HTML mockups into the same stage-2 output directory.

**Architecture:** Create a focused `tools/paper/figures/stage2` package with separate modules for asset resolution, rendering, and CLI orchestration. Reuse stage-1 tile-loading and visual idioms where practical, but keep stage-2 layout logic independent so it can evolve without coupling to the HTML mockups.

**Tech Stack:** Python 3.13, Pillow, pathlib, json, pytest

---

## File Structure

- Create: `tools/paper/figures/stage2/__init__.py`
- Create: `tools/paper/figures/stage2/assets.py`
- Create: `tools/paper/figures/stage2/render.py`
- Create: `tools/paper/figures/stage2/build_training_figure.py`
- Create: `tests/test_stage2_paper_figure.py`
- Modify: `paper/figures/stage2_pipeline_designs.html` (move to `paper/figures/stage2/stage2_pipeline_designs.html`)
- Modify: `paper/figures/stage2_pipeline_design_A_mod.html` (move to `paper/figures/stage2/stage2_pipeline_design_A_mod.html`)

Implementation notes:

- `assets.py` owns patch selection, PixCell output discovery, and local asset-path resolution.
- `render.py` owns the publication layout drawing using PIL only.
- `build_training_figure.py` owns CLI defaults, output directory creation, PNG saving, and JSON metadata emission.
- `tests/test_stage2_paper_figure.py` covers resolver determinism, missing-match errors, metadata content, and end-to-end synthetic rendering.

### Task 1: Scaffold Stage-2 Package And Relocate Mockups

**Files:**
- Create: `tools/paper/figures/stage2/__init__.py`
- Modify: `paper/figures/stage2_pipeline_designs.html`
- Modify: `paper/figures/stage2_pipeline_design_A_mod.html`

- [ ] **Step 1: Write the failing relocation/package test**

```python
from pathlib import Path


def test_stage2_package_and_mockups_exist() -> None:
    repo = Path(__file__).resolve().parents[1]

    assert (repo / "tools/paper/figures/stage2/__init__.py").exists()
    assert (repo / "paper/figures/stage2/stage2_pipeline_designs.html").exists()
    assert (repo / "paper/figures/stage2/stage2_pipeline_design_A_mod.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_paper_figure.py::test_stage2_package_and_mockups_exist -v`
Expected: FAIL because the stage-2 package and relocated HTML files do not exist yet.

- [ ] **Step 3: Add the stage-2 package and move the mockup files**

```python
# tools/paper/figures/stage2/__init__.py
"""Stage-2 paper figure generation tools."""
```

```text
Move:
paper/figures/stage2_pipeline_designs.html
-> paper/figures/stage2/stage2_pipeline_designs.html

Move:
paper/figures/stage2_pipeline_design_A_mod.html
-> paper/figures/stage2/stage2_pipeline_design_A_mod.html
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2_paper_figure.py::test_stage2_package_and_mockups_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/paper/figures/stage2/__init__.py \
        paper/figures/stage2/stage2_pipeline_designs.html \
        paper/figures/stage2/stage2_pipeline_design_A_mod.html \
        tests/test_stage2_paper_figure.py
git commit -m "feat: scaffold stage2 figure package"
```

### Task 2: Implement Representative Patch Resolution And Asset Discovery

**Files:**
- Create: `tools/paper/figures/stage2/assets.py`
- Test: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Write the failing resolver tests**

```python
from pathlib import Path

import json
import pytest

from tools.paper.figures.stage2.assets import pick_representative_patch


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


def test_pick_representative_patch_raises_with_checked_ids(tmp_path: Path) -> None:
    selection_json = tmp_path / "selections.json"
    pixcell_root = tmp_path / "pixcell"
    pixcell_root.mkdir()
    selection_json.write_text(
        json.dumps({
            "groups": {
                "g1": {"selections": [{"patch_id": "100_100"}, {"patch_id": "200_200"}]}
            }
        }),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as exc:
        pick_representative_patch(selection_json, pixcell_root)

    message = str(exc.value)
    assert "100_100" in message
    assert "200_200" in message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stage2_paper_figure.py::test_pick_representative_patch_uses_first_matching_selection tests/test_stage2_paper_figure.py::test_pick_representative_patch_raises_with_checked_ids -v`
Expected: FAIL with import or missing symbol errors because `assets.py` does not exist yet.

- [ ] **Step 3: Write minimal asset-resolution implementation**

```python
# tools/paper/figures/stage2/assets.py
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RepresentativePatch:
    group_id: str
    patch_id: str
    generated_he_path: Path


def _iter_selected_patch_ids(selection_json: Path) -> list[tuple[str, str]]:
    with selection_json.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    ordered: list[tuple[str, str]] = []
    for group_id, group in payload.get("groups", {}).items():
        for selection in group.get("selections", []):
            ordered.append((str(group_id), str(selection["patch_id"])))
    return ordered


def _generated_he_path(pixcell_root: Path, patch_id: str) -> Path:
    return pixcell_root / patch_id / "all" / "generated_he.png"


def pick_representative_patch(selection_json: Path, pixcell_root: Path) -> RepresentativePatch:
    checked: list[str] = []
    for group_id, patch_id in _iter_selected_patch_ids(selection_json):
        checked.append(patch_id)
        generated = _generated_he_path(pixcell_root, patch_id)
        if generated.exists():
            return RepresentativePatch(
                group_id=group_id,
                patch_id=patch_id,
                generated_he_path=generated,
            )

    raise FileNotFoundError(
        "No selected patch has PixCell generated output. Checked patch IDs: "
        + ", ".join(checked)
    )
```

- [ ] **Step 4: Extend implementation to resolve local stage-2 asset paths**

```python
@dataclass(frozen=True)
class Stage2FigureAssets:
    patch_id: str
    group_id: str
    reference_he_path: Path
    cell_type_path: Path
    cell_state_path: Path
    vasculature_path: Path
    oxygen_path: Path
    glucose_path: Path
    generated_he_path: Path


def resolve_stage2_assets(
    processed_dir: Path,
    selection_json: Path,
    pixcell_root: Path,
    patch_id: str | None = None,
) -> Stage2FigureAssets:
    match = (
        RepresentativePatch("forced", patch_id, _generated_he_path(pixcell_root, patch_id))
        if patch_id is not None
        else pick_representative_patch(selection_json, pixcell_root)
    )
    asset_paths = Stage2FigureAssets(
        patch_id=match.patch_id,
        group_id=match.group_id,
        reference_he_path=processed_dir / "he" / f"{match.patch_id}.png",
        cell_type_path=processed_dir / "cell_types/union" / f"{match.patch_id}.png",
        cell_state_path=processed_dir / "cell_states/union" / f"{match.patch_id}.png",
        vasculature_path=processed_dir / "vasculature" / f"{match.patch_id}.png",
        oxygen_path=processed_dir / "oxygen" / f"{match.patch_id}.png",
        glucose_path=processed_dir / "glucose" / f"{match.patch_id}.png",
        generated_he_path=match.generated_he_path,
    )
    for path in [
        asset_paths.reference_he_path,
        asset_paths.cell_type_path,
        asset_paths.cell_state_path,
        asset_paths.vasculature_path,
        asset_paths.oxygen_path,
        asset_paths.glucose_path,
        asset_paths.generated_he_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required stage2 asset: {path}")
    return asset_paths
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_stage2_paper_figure.py::test_pick_representative_patch_uses_first_matching_selection tests/test_stage2_paper_figure.py::test_pick_representative_patch_raises_with_checked_ids -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tools/paper/figures/stage2/assets.py tests/test_stage2_paper_figure.py
git commit -m "feat: resolve stage2 representative patch assets"
```

### Task 3: Implement Stage-2 Figure Rendering

**Files:**
- Create: `tools/paper/figures/stage2/render.py`
- Test: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Write the failing render smoke test**

```python
import numpy as np
from pathlib import Path
from PIL import Image

from tools.paper.figures.stage2.assets import Stage2FigureAssets
from tools.paper.figures.stage2.render import render_stage2_training_figure


def test_render_stage2_training_figure_returns_nonempty_canvas(tmp_path: Path) -> None:
    def save_rgb(path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color).save(path)

    assets = Stage2FigureAssets(
        patch_id="23296_21760",
        group_id="g1",
        reference_he_path=tmp_path / "he.png",
        cell_type_path=tmp_path / "type.png",
        cell_state_path=tmp_path / "state.png",
        vasculature_path=tmp_path / "vas.png",
        oxygen_path=tmp_path / "oxygen.png",
        glucose_path=tmp_path / "glucose.png",
        generated_he_path=tmp_path / "generated.png",
    )
    save_rgb(assets.reference_he_path, (240, 200, 210))
    save_rgb(assets.cell_type_path, (210, 235, 245))
    save_rgb(assets.cell_state_path, (225, 245, 230))
    save_rgb(assets.vasculature_path, (40, 0, 0))
    save_rgb(assets.oxygen_path, (0, 255, 255))
    save_rgb(assets.glucose_path, (255, 240, 40))
    save_rgb(assets.generated_he_path, (235, 190, 205))

    canvas = render_stage2_training_figure(assets, tile_size=64)

    arr = np.asarray(canvas)
    assert canvas.width > 300
    assert canvas.height > 180
    assert arr.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_paper_figure.py::test_render_stage2_training_figure_returns_nonempty_canvas -v`
Expected: FAIL with import error because `render.py` and `render_stage2_training_figure` do not exist yet.

- [ ] **Step 3: Implement the rendering module with reusable helpers**

```python
# tools/paper/figures/stage2/render.py
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .assets import Stage2FigureAssets


def _open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _fit_tile(path: Path, tile_size: int) -> Image.Image:
    image = _open_rgb(path)
    if image.size == (tile_size, tile_size):
        return image
    return image.resize((tile_size, tile_size), Image.Resampling.NEAREST)


def render_stage2_training_figure(
    assets: Stage2FigureAssets,
    tile_size: int = 128,
    panel_gap: int = 10,
    header_height: int = 28,
) -> Image.Image:
    width = tile_size * 6 + panel_gap * 7
    height = tile_size * 2 + header_height + panel_gap * 5
    canvas = Image.new("RGB", (width, height), (255, 253, 249))
    draw = ImageDraw.Draw(canvas)

    x0 = panel_gap
    y_top = header_height + panel_gap
    y_bottom = y_top + tile_size + panel_gap * 2

    reference_he = _fit_tile(assets.reference_he_path, tile_size)
    generated_he = _fit_tile(assets.generated_he_path, tile_size)
    cell_type = _fit_tile(assets.cell_type_path, tile_size // 2)
    cell_state = _fit_tile(assets.cell_state_path, tile_size // 2)
    vasculature = _fit_tile(assets.vasculature_path, tile_size // 2)
    oxygen = _fit_tile(assets.oxygen_path, tile_size // 2)
    glucose = _fit_tile(assets.glucose_path, tile_size // 2)

    canvas.paste(reference_he, (x0, y_top))
    canvas.paste(cell_type, (x0, y_bottom))
    canvas.paste(cell_state, (x0 + tile_size // 2 + panel_gap, y_bottom))
    canvas.paste(vasculature, (x0 + tile_size + panel_gap * 2, y_bottom))
    canvas.paste(oxygen, (x0 + tile_size + tile_size // 2 + panel_gap * 3, y_bottom))
    canvas.paste(glucose, (x0 + tile_size * 2 + panel_gap * 4, y_bottom))

    denoiser_x = x0 + tile_size * 3 + panel_gap * 4
    denoiser_y = y_top + 4
    draw.rounded_rectangle(
        [denoiser_x, denoiser_y, denoiser_x + tile_size + 20, denoiser_y + 78],
        radius=10,
        fill=(245, 241, 233),
        outline=(142, 136, 128),
        width=2,
    )
    draw.text((denoiser_x + 20, denoiser_y + 18), "PixCell denoiser", fill=(36, 33, 29))
    draw.text((denoiser_x + 8, denoiser_y - 18), "noisy latent Z_t", fill=(51, 67, 93))
    draw.text((denoiser_x + tile_size + 40, denoiser_y + 8), "Denoised", fill=(51, 67, 93), anchor="mm")
    draw.text((denoiser_x + tile_size + 40, denoiser_y + 24), "latent Z", fill=(51, 67, 93), anchor="mm")

    vae_x = denoiser_x + tile_size + 56
    vae_y = denoiser_y + 96
    draw.rounded_rectangle(
        [vae_x, vae_y, vae_x + 60, vae_y + 54],
        radius=8,
        fill=(245, 241, 233),
        outline=(142, 136, 128),
        width=2,
    )
    draw.text((vae_x + 30, vae_y + 20), "SD3.5", fill=(36, 33, 29), anchor="mm")
    draw.text((vae_x + 30, vae_y + 34), "VAE", fill=(36, 33, 29), anchor="mm")
    canvas.paste(generated_he, (vae_x - 2, vae_y + 72))

    draw.line((x0 + tile_size, y_top + tile_size // 2, denoiser_x, y_top + tile_size // 2), fill=(89, 84, 79), width=2)
    draw.line((denoiser_x + tile_size + 20, denoiser_y + 55, vae_x, denoiser_y + 55), fill=(89, 84, 79), width=2)
    draw.line((vae_x + 30, vae_y + 54, vae_x + 30, vae_y + 72), fill=(89, 84, 79), width=2)
    draw.line((x0 + tile_size + 10, y_bottom + 12, denoiser_x, y_bottom + 12), fill=(45, 140, 79), width=2)
    draw.line((denoiser_x, y_bottom + 12, denoiser_x, denoiser_y + 70), fill=(45, 140, 79), width=2)

    return canvas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage2_paper_figure.py::test_render_stage2_training_figure_returns_nonempty_canvas -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/paper/figures/stage2/render.py tests/test_stage2_paper_figure.py
git commit -m "feat: render stage2 training figure"
```

### Task 4: Implement Builder CLI, Metadata Output, And End-To-End Tests

**Files:**
- Create: `tools/paper/figures/stage2/build_training_figure.py`
- Test: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Write the failing end-to-end metadata test**

```python
import json
from pathlib import Path
from PIL import Image

from tools.paper.figures.stage2.build_training_figure import build_stage2_training_figure


def test_build_stage2_training_figure_writes_png_and_json(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    pixcell_root = tmp_path / "pixcell"
    output_dir = tmp_path / "paper/figures/stage2"
    selection_json = tmp_path / "selections.json"

    payload = {"groups": {"g1": {"selections": [{"patch_id": "23296_21760"}]}}}
    selection_json.write_text(json.dumps(payload), encoding="utf-8")

    for rel in [
        "he/23296_21760.png",
        "cell_types/union/23296_21760.png",
        "cell_states/union/23296_21760.png",
        "vasculature/23296_21760.png",
        "oxygen/23296_21760.png",
        "glucose/23296_21760.png",
    ]:
        path = processed / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (200, 200, 200)).save(path)

    generated = pixcell_root / "23296_21760" / "all"
    generated.mkdir(parents=True)
    Image.new("RGB", (64, 64), (220, 180, 190)).save(generated / "generated_he.png")

    png_path, json_path = build_stage2_training_figure(
        processed_dir=processed,
        selection_json=selection_json,
        pixcell_root=pixcell_root,
        out_dir=output_dir,
    )

    assert png_path.exists()
    assert json_path.exists()
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["patch_id"] == "23296_21760"
    assert metadata["generated_he_path"].endswith("23296_21760/all/generated_he.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage2_paper_figure.py::test_build_stage2_training_figure_writes_png_and_json -v`
Expected: FAIL with import error because the builder CLI does not exist yet.

- [ ] **Step 3: Implement the orchestration module and CLI**

```python
# tools/paper/figures/stage2/build_training_figure.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import resolve_stage2_assets
from .render import render_stage2_training_figure


DEFAULT_SELECTION_JSON = Path("paper/figures/stage1/crc33_marker_high_patch_examples_selections.json")
DEFAULT_PROCESSED_DIR = Path("processed_crc33")
DEFAULT_PIXCELL_ROOT = Path(
    "/home/pohaoc2/UW/bagherilab/PixCell/inference_output/paired_ablation/ablation_results"
)
DEFAULT_OUT_DIR = Path("paper/figures/stage2")


def build_stage2_training_figure(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    selection_json: Path = DEFAULT_SELECTION_JSON,
    pixcell_root: Path = DEFAULT_PIXCELL_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    patch_id: str | None = None,
) -> tuple[Path, Path]:
    assets = resolve_stage2_assets(processed_dir, selection_json, pixcell_root, patch_id=patch_id)
    canvas = render_stage2_training_figure(assets)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "crc33_stage2_training_pipeline.png"
    out_json = out_dir / "crc33_stage2_training_pipeline.json"
    canvas.save(out_png)

    metadata = {
        "patch_id": assets.patch_id,
        "group_id": assets.group_id,
        "selection_json": str(selection_json),
        "processed_dir": str(processed_dir),
        "pixcell_root": str(pixcell_root),
        "reference_he_path": str(assets.reference_he_path),
        "cell_type_path": str(assets.cell_type_path),
        "cell_state_path": str(assets.cell_state_path),
        "vasculature_path": str(assets.vasculature_path),
        "oxygen_path": str(assets.oxygen_path),
        "glucose_path": str(assets.glucose_path),
        "generated_he_path": str(assets.generated_he_path),
        "selection_rule": "first stage1 selected patch with matching PixCell all/generated_he.png",
        "output_png": str(out_png),
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_png, out_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--selection-json", type=Path, default=DEFAULT_SELECTION_JSON)
    parser.add_argument("--pixcell-root", type=Path, default=DEFAULT_PIXCELL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--patch-id", default=None)
    args = parser.parse_args()

    out_png, out_json = build_stage2_training_figure(
        processed_dir=args.processed_dir,
        selection_json=args.selection_json,
        pixcell_root=args.pixcell_root,
        out_dir=args.out_dir,
        patch_id=args.patch_id,
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run targeted tests to verify they pass**

Run: `pytest tests/test_stage2_paper_figure.py -v`
Expected: PASS for package, resolver, render, and builder coverage.

- [ ] **Step 5: Run the builder with real defaults to produce the stage-2 figure**

Run: `python3 -m tools.paper.figures.stage2.build_training_figure`
Expected: writes `paper/figures/stage2/crc33_stage2_training_pipeline.png` and `paper/figures/stage2/crc33_stage2_training_pipeline.json`

- [ ] **Step 6: Commit**

```bash
git add tools/paper/figures/stage2/build_training_figure.py \
        tools/paper/figures/stage2/assets.py \
        tools/paper/figures/stage2/render.py \
        tests/test_stage2_paper_figure.py \
        paper/figures/stage2
git commit -m "feat: add stage2 training paper figure builder"
```

## Self-Review Checklist

- Spec coverage: this plan covers stage-2 package creation, output relocation, deterministic representative-patch selection, Python rendering, JSON metadata, failure cases, and synthetic tests.
- Placeholder scan: no `TBD`, `TODO`, or deferred implementation language remains.
- Type consistency: `RepresentativePatch`, `Stage2FigureAssets`, `pick_representative_patch`, `resolve_stage2_assets`, `render_stage2_training_figure`, and `build_stage2_training_figure` are named consistently across tasks and tests.

## Execution Handoff

The user already chose subagent-driven execution. Implement this plan with `superpowers:subagent-driven-development`, using one fresh implementation subagent per task and review gates between tasks.