# Cell Morphology Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute per-cell circularity from binary patch masks and generate violin plots (area, circularity, z-scored marker intensities) + a markdown summary table for CRC33 cell type QC.

**Architecture:** Two standalone scripts: `compute_shape_features.py` extracts and caches per-cell morphology from mask PNGs; `cell_vis.py` reads those features + cell assignments and writes violin plots and a markdown summary. Decoupled so slow mask processing runs once.

**Tech Stack:** Python 3.13, skimage (measure.label, regionprops), pandas, matplotlib, Pillow, pytest

---

## File Map

| File | Role |
|---|---|
| `scripts/compute_shape_features.py` | Load binary masks → label → regionprops → match to cells → CSV |
| `scripts/cell_vis.py` | Load CSV + shapes → violin plots + cell_summary.md |
| `tests/test_compute_shape_features.py` | Unit tests for mask loading, circularity, cell matching |
| `tests/test_cell_vis.py` | Unit tests for z-scoring, summary markdown generation |

---

## Task 1: Shape feature extraction core functions

**Files:**
- Create: `scripts/compute_shape_features.py`
- Create: `tests/test_compute_shape_features.py`

- [ ] **Step 1: Write failing tests for `load_mask_regions` and `compute_circularity`**

Create `tests/test_compute_shape_features.py`:

```python
"""Tests for compute_shape_features.py core functions."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.draw import disk
from skimage.measure import label, regionprops

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compute_shape_features import compute_circularity, load_mask_regions


def _make_mask_png(shape=(64, 64), circles=None) -> Path:
    """Write a binary mask PNG with filled circles. Returns path."""
    img = np.zeros(shape, dtype=np.uint8)
    for (r, c, radius) in (circles or []):
        rr, cc = disk((r, c), radius, shape=shape)
        img[rr, cc] = 255
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(img).save(tmp.name)
    return Path(tmp.name)


def test_load_mask_regions_single_circle():
    path = _make_mask_png(circles=[(32, 32, 10)])
    regions = load_mask_regions(path)
    assert len(regions) == 1
    assert 290 < regions[0].area < 340  # π*10² ≈ 314


def test_load_mask_regions_two_circles():
    path = _make_mask_png(shape=(128, 128), circles=[(30, 30, 8), (90, 90, 8)])
    regions = load_mask_regions(path)
    assert len(regions) == 2


def test_load_mask_regions_empty_mask():
    path = _make_mask_png()  # all zeros
    regions = load_mask_regions(path)
    assert regions == []


def test_compute_circularity_circle():
    img = np.zeros((128, 128), dtype=np.uint8)
    rr, cc = disk((64, 64), 30, shape=(128, 128))
    img[rr, cc] = 255
    labeled = label(img > 128)
    region = regionprops(labeled)[0]
    circ = compute_circularity(region)
    assert 0.85 < circ <= 1.0


def test_compute_circularity_zero_perimeter():
    # Single-pixel region — perimeter is 0 edge case
    from unittest.mock import MagicMock
    region = MagicMock()
    region.area = 1
    region.perimeter = 0.0
    assert compute_circularity(region) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/pohaoc2/UW/bagherilab/he-feature-visualizer
conda run -n he-feature-vis pytest tests/test_compute_shape_features.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'compute_shape_features'`

- [ ] **Step 3: Implement `load_mask_regions` and `compute_circularity`**

Create `scripts/compute_shape_features.py`:

```python
"""Compute per-cell shape features (area, perimeter, circularity) from binary patch masks."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops


def load_mask_regions(mask_path: Path) -> list:
    """Load binary mask PNG, label connected components, return regionprops list."""
    img = np.array(Image.open(mask_path))
    binary = (img > 128).astype(np.uint8)
    labeled = label(binary)
    return regionprops(labeled)


def compute_circularity(region) -> float:
    """Circularity = 4π·area/perimeter². Returns 0.0 if perimeter is zero."""
    if region.perimeter == 0:
        return 0.0
    return 4 * math.pi * region.area / (region.perimeter ** 2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n he-feature-vis pytest tests/test_compute_shape_features.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_shape_features.py tests/test_compute_shape_features.py
git commit -m "feat: add shape feature core functions (load_mask_regions, compute_circularity)"
```

---

## Task 2: Cell-to-region matching + main loop

**Files:**
- Modify: `scripts/compute_shape_features.py`
- Modify: `tests/test_compute_shape_features.py`

- [ ] **Step 1: Write failing tests for `match_cells_to_regions`**

Append to `tests/test_compute_shape_features.py`:

```python
def test_match_cells_to_regions_hit():
    """Cell centroid inside a circle → matched with valid circularity."""
    import pandas as pd
    from skimage.measure import label, regionprops

    img = np.zeros((64, 64), dtype=np.uint8)
    rr, cc = disk((32, 32), 10, shape=(64, 64))
    img[rr, cc] = 255
    labeled = label(img > 128)
    regions = regionprops(labeled)
    regions_by_label = {r.label: r for r in regions}

    cells = pd.DataFrame([{
        "CellID": 1, "PatchID": "0_0",
        "centroid_x_local": 32.0, "centroid_y_local": 32.0,
    }])

    from compute_shape_features import match_cells_to_regions
    result = match_cells_to_regions(cells, labeled, regions_by_label)
    assert result.loc[0, "circularity"] is not None
    assert not pd.isna(result.loc[0, "circularity"])
    assert result.loc[0, "area_px"] > 0


def test_match_cells_to_regions_miss():
    """Cell centroid outside all regions → NaN."""
    import pandas as pd
    from skimage.measure import label, regionprops

    img = np.zeros((64, 64), dtype=np.uint8)
    rr, cc = disk((10, 10), 5, shape=(64, 64))
    img[rr, cc] = 255
    labeled = label(img > 128)
    regions_by_label = {r.label: r for r in regionprops(labeled)}

    cells = pd.DataFrame([{
        "CellID": 2, "PatchID": "0_0",
        "centroid_x_local": 55.0, "centroid_y_local": 55.0,
    }])

    from compute_shape_features import match_cells_to_regions
    result = match_cells_to_regions(cells, labeled, regions_by_label)
    assert pd.isna(result.loc[0, "circularity"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n he-feature-vis pytest tests/test_compute_shape_features.py::test_match_cells_to_regions_hit tests/test_compute_shape_features.py::test_match_cells_to_regions_miss -v
```

Expected: `ImportError` on `match_cells_to_regions`

- [ ] **Step 3: Implement `match_cells_to_regions` and `main()`**

Append to `scripts/compute_shape_features.py`:

```python
def match_cells_to_regions(
    cells_df: pd.DataFrame,
    labeled_mask: np.ndarray,
    regions_by_label: dict,
) -> pd.DataFrame:
    """Match each cell to a mask region via local centroid pixel lookup.

    centroid_x_local → column index, centroid_y_local → row index.
    Returns DataFrame with columns: CellID, PatchID, area_px, perimeter_px, circularity.
    Unmatched cells get NaN for shape columns.
    """
    rows = []
    h, w = labeled_mask.shape
    for _, cell in cells_df.iterrows():
        row_idx = round(float(cell["centroid_y_local"]))
        col_idx = round(float(cell["centroid_x_local"]))
        base = {"CellID": cell["CellID"], "PatchID": cell["PatchID"]}
        if not (0 <= row_idx < h and 0 <= col_idx < w):
            rows.append({**base, "area_px": None, "perimeter_px": None, "circularity": None})
            continue
        lbl = labeled_mask[row_idx, col_idx]
        if lbl == 0 or lbl not in regions_by_label:
            rows.append({**base, "area_px": None, "perimeter_px": None, "circularity": None})
            continue
        region = regions_by_label[lbl]
        rows.append({
            **base,
            "area_px": region.area,
            "perimeter_px": float(region.perimeter),
            "circularity": compute_circularity(region),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-cell shape features from binary patch masks."
    )
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="processed_crc33/ directory containing cell_masks/ and cell_assignments.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    assignments_path = data_dir / "cell_assignments.csv"
    mask_dir = data_dir / "cell_masks"

    assignments = pd.read_csv(assignments_path)
    all_features: list[pd.DataFrame] = []

    mask_paths = sorted(mask_dir.glob("*.png"))
    print(f"Processing {len(mask_paths)} patch masks...")

    for mask_path in mask_paths:
        patch_id = mask_path.stem  # e.g. "9728_768"
        patch_cells = assignments[assignments["PatchID"] == patch_id]
        if patch_cells.empty:
            continue

        img = np.array(Image.open(mask_path))
        binary = (img > 128).astype(np.uint8)
        labeled_mask = label(binary)
        regions = regionprops(labeled_mask)
        regions_by_label = {r.label: r for r in regions}

        patch_features = match_cells_to_regions(patch_cells, labeled_mask, regions_by_label)
        all_features.append(patch_features)

    features_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame(
        columns=["CellID", "PatchID", "area_px", "perimeter_px", "circularity"]
    )

    unmatched = features_df["circularity"].isna().sum()
    total = len(features_df)
    rate = unmatched / total if total > 0 else 0.0
    if rate > 0.05:
        print(f"WARNING: {unmatched}/{total} cells ({rate:.1%}) unmatched to mask regions.")
    else:
        print(f"Matched {total - unmatched}/{total} cells ({1 - rate:.1%}).")

    out_path = data_dir / "cell_shape_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved {len(features_df)} records → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all shape feature tests**

```bash
conda run -n he-feature-vis pytest tests/test_compute_shape_features.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_shape_features.py tests/test_compute_shape_features.py
git commit -m "feat: add cell-to-region matching and main loop in compute_shape_features"
```

---

## Task 3: cell_vis.py — scaffold, data loading, z-score

**Files:**
- Create: `scripts/cell_vis.py`
- Create: `tests/test_cell_vis.py`

- [ ] **Step 1: Write failing tests for `load_data` and `zscore_markers`**

Create `tests/test_cell_vis.py`:

```python
"""Tests for cell_vis.py data loading and z-score normalization."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cell_vis import load_data, zscore_markers


def _make_csv_pair(tmp_dir: Path):
    """Write minimal cell_assignments.csv and cell_shape_features.csv to tmp_dir."""
    assignments = pd.DataFrame({
        "CellID": [1, 2, 3],
        "PatchID": ["0_0", "0_0", "0_0"],
        "cell_type": ["cancer", "immune", "healthy"],
        "cell_state": ["proliferative", "nonproliferative", "nonproliferative"],
        "Area_cellvit_px": [500, 200, 300],
        "Pan-CK": [10.0, 1.0, 2.0],
        "CD45": [1.0, 9.0, 2.0],
        "Ki67": [5.0, 2.0, 1.0],
    })
    shapes = pd.DataFrame({
        "CellID": [1, 2, 3],
        "PatchID": ["0_0", "0_0", "0_0"],
        "area_px": [480, 190, 290],
        "perimeter_px": [80.0, 50.0, 65.0],
        "circularity": [0.94, 0.96, 0.92],
    })
    assignments.to_csv(tmp_dir / "cell_assignments.csv", index=False)
    shapes.to_csv(tmp_dir / "cell_shape_features.csv", index=False)


def test_load_data_merges_shape_features():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
    assert "circularity" in df.columns
    assert len(df) == 3


def test_load_data_missing_shapes_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pd.DataFrame({"CellID": [1]}).to_csv(tmp_dir / "cell_assignments.csv", index=False)
        with pytest.raises(FileNotFoundError, match="compute_shape_features"):
            load_data(tmp_dir)


def test_zscore_markers_normalizes():
    df = pd.DataFrame({"Pan-CK": [1.0, 2.0, 3.0], "CD45": [10.0, 10.0, 10.0]})
    result = zscore_markers(df.copy(), ["Pan-CK", "CD45"])
    # Pan-CK should be z-scored
    assert abs(result["Pan-CK"].mean()) < 1e-10
    assert abs(result["Pan-CK"].std() - 1.0) < 1e-6
    # CD45 std=0 → all zeros
    assert (result["CD45"] == 0.0).all()


def test_zscore_markers_missing_column_warns(capsys):
    df = pd.DataFrame({"Pan-CK": [1.0, 2.0]})
    zscore_markers(df.copy(), ["Pan-CK", "MISSING_MARKER"])
    captured = capsys.readouterr()
    assert "MISSING_MARKER" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'cell_vis'`

- [ ] **Step 3: Implement scaffold, `load_data`, `zscore_markers`**

Create `scripts/cell_vis.py`:

```python
"""Generate violin plots for CRC33 cell type characterization (QC + SI figure)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELL_TYPES = ["cancer", "immune", "healthy"]
COLORS = {"cancer": "#e74c3c", "immune": "#3498db", "healthy": "#2ecc71"}

MARKERS = [
    "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550",
    "CD4", "FOXP3", "CD8a", "CD45RO", "CD20", "PD-L1",
    "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA",
]


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and merge cell_assignments.csv + cell_shape_features.csv."""
    assignments_path = data_dir / "cell_assignments.csv"
    shape_path = data_dir / "cell_shape_features.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing: {assignments_path}")
    if not shape_path.exists():
        raise FileNotFoundError(
            f"Missing: {shape_path}. Run compute_shape_features.py first."
        )
    assignments = pd.read_csv(assignments_path)
    shapes = pd.read_csv(shape_path)[["CellID", "area_px", "perimeter_px", "circularity"]]
    return assignments.merge(shapes, on="CellID", how="left")


def zscore_markers(df: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """Z-score each marker column independently in-place (returns copy)."""
    df = df.copy()
    for m in markers:
        if m not in df.columns:
            print(f"WARNING: marker '{m}' not found in data, skipping.")
            continue
        std = df[m].std()
        df[m] = 0.0 if std == 0 else (df[m] - df[m].mean()) / std
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell type violin plots and summary markdown.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--save-dir", type=Path, default=None)
    return parser.parse_args()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/cell_vis.py tests/test_cell_vis.py
git commit -m "feat: add cell_vis scaffold with load_data and zscore_markers"
```

---

## Task 4: Violin plots — area and circularity

**Files:**
- Modify: `scripts/cell_vis.py`
- Modify: `tests/test_cell_vis.py`

- [ ] **Step 1: Write failing tests for `plot_violin_area` and `plot_violin_circularity`**

Append to `tests/test_cell_vis.py`:

```python
def test_plot_violin_area_saves_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        from cell_vis import plot_violin_area
        out = tmp_dir / "violin_area.png"
        plot_violin_area(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_violin_circularity_saves_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        from cell_vis import plot_violin_circularity
        out = tmp_dir / "violin_circularity.png"
        plot_violin_circularity(df, out)
        assert out.exists()
        assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py::test_plot_violin_area_saves_file tests/test_cell_vis.py::test_plot_violin_circularity_saves_file -v
```

Expected: `ImportError` on `plot_violin_area`

- [ ] **Step 3: Implement `plot_violin_area` and `plot_violin_circularity`**

Append to `scripts/cell_vis.py`:

```python
def _style_violin_parts(parts: dict, cell_types: list[str]) -> None:
    """Apply per-type colors to violin bodies and set median line color."""
    for pc, ct in zip(parts["bodies"], cell_types):
        pc.set_facecolor(COLORS[ct])
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)


def plot_violin_area(df: pd.DataFrame, save_path: Path) -> None:
    """Violin plot of cell area (Area_cellvit_px) per cell type."""
    data = [df[df["cell_type"] == ct]["Area_cellvit_px"].dropna().values for ct in CELL_TYPES]
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data, positions=range(len(CELL_TYPES)), showmedians=True)
    _style_violin_parts(parts, CELL_TYPES)
    ax.set_xticks(range(len(CELL_TYPES)))
    ax.set_xticklabels(CELL_TYPES)
    ax.set_ylabel("Cell area (px²)")
    ax.set_title("Cell Area by Type")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_violin_circularity(df: pd.DataFrame, save_path: Path) -> None:
    """Violin plot of circularity per cell type."""
    data = [df[df["cell_type"] == ct]["circularity"].dropna().values for ct in CELL_TYPES]
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data, positions=range(len(CELL_TYPES)), showmedians=True)
    _style_violin_parts(parts, CELL_TYPES)
    ax.set_xticks(range(len(CELL_TYPES)))
    ax.set_xticklabels(CELL_TYPES)
    ax.set_ylabel("Circularity (0–1)")
    ax.set_title("Cell Circularity by Type")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/cell_vis.py tests/test_cell_vis.py
git commit -m "feat: add violin_area and violin_circularity plots"
```

---

## Task 5: Violin plot — z-scored marker intensities

**Files:**
- Modify: `scripts/cell_vis.py`
- Modify: `tests/test_cell_vis.py`

- [ ] **Step 1: Write failing test for `plot_violin_markers`**

Append to `tests/test_cell_vis.py`:

```python
def test_plot_violin_markers_saves_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        from cell_vis import plot_violin_markers
        out = tmp_dir / "violin_markers.png"
        plot_violin_markers(df, out)
        assert out.exists()
        assert out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py::test_plot_violin_markers_saves_file -v
```

Expected: `ImportError` on `plot_violin_markers`

- [ ] **Step 3: Implement `plot_violin_markers`**

Append to `scripts/cell_vis.py`:

```python
def plot_violin_markers(df: pd.DataFrame, save_path: Path) -> None:
    """Grouped violin plot of z-scored marker intensities per cell type.

    All available markers z-scored independently. Split into 2 row subplots
    (~9-10 markers each). Three overlaid violins per marker (cancer/immune/healthy).
    """
    from matplotlib.patches import Patch

    available = [m for m in MARKERS if m in df.columns]
    df_z = zscore_markers(df, available)

    mid = len(available) // 2
    row_groups = [available[:mid], available[mid:]]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharey=False)

    for ax, marker_subset in zip(axes, row_groups):
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        gap = len(CELL_TYPES) + 1  # spacing between marker groups

        for i, marker in enumerate(marker_subset):
            base = i * gap
            for j, ct in enumerate(CELL_TYPES):
                data = df_z[df_z["cell_type"] == ct][marker].dropna().values
                if len(data) < 2:
                    continue
                parts = ax.violinplot([data], positions=[base + j], showmedians=True, widths=0.8)
                for pc in parts["bodies"]:
                    pc.set_facecolor(COLORS[ct])
                    pc.set_alpha(0.6)
                for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                    if key in parts:
                        parts[key].set_color("black")
                        parts[key].set_linewidth(0.6)
            tick_positions.append(base + 1.0)  # center under middle violin
            tick_labels.append(marker)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Z-score intensity")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    legend_handles = [Patch(facecolor=COLORS[ct], label=ct) for ct in CELL_TYPES]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle("Multiplex Marker Intensities by Cell Type (Z-score normalized)", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
```

- [ ] **Step 4: Run all tests**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/cell_vis.py tests/test_cell_vis.py
git commit -m "feat: add violin_markers z-score grouped violin plot"
```

---

## Task 6: Markdown summary + main() wiring

**Files:**
- Modify: `scripts/cell_vis.py`
- Modify: `tests/test_cell_vis.py`

- [ ] **Step 1: Write failing test for `write_summary_md`**

Append to `tests/test_cell_vis.py`:

```python
def test_write_summary_md_contains_tables():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        from cell_vis import write_summary_md
        out = tmp_dir / "cell_summary.md"
        write_summary_md(df, out)
        assert out.exists()
        text = out.read_text()
        assert "cancer" in text
        assert "immune" in text
        assert "healthy" in text
        assert "proliferative" in text
        assert "nonproliferative" in text
        # Cross-tab header present
        assert "cell_type" in text
        assert "%" in text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py::test_write_summary_md_contains_tables -v
```

Expected: `ImportError` on `write_summary_md`

- [ ] **Step 3: Implement `write_summary_md` and `main()`**

Append to `scripts/cell_vis.py`:

```python
def write_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """Write cell count tables and cross-tab to a markdown file."""
    n_cells = len(df)
    n_patches = df["PatchID"].nunique()
    states = ["nonproliferative", "proliferative", "dead"]

    lines: list[str] = [
        f"# CRC33 Cell Summary (n = {n_cells:,} cells, {n_patches:,} patches)\n",
        "## Cell Type Distribution\n",
        "| Type | N | % |",
        "|---|---|---|",
    ]
    for ct in CELL_TYPES:
        n = int((df["cell_type"] == ct).sum())
        lines.append(f"| {ct} | {n:,} | {n / n_cells * 100:.1f}% |")

    lines += [
        "",
        "## Cell State Distribution\n",
        "| State | N | % |",
        "|---|---|---|",
    ]
    for state in states:
        n = int((df["cell_state"] == state).sum())
        lines.append(f"| {state} | {n:,} | {n / n_cells * 100:.1f}% |")

    lines += [
        "",
        "## Cell Type × State — Counts\n",
        "| cell_type | nonproliferative | proliferative | dead | Total |",
        "|---|---|---|---|---|",
    ]
    for ct in CELL_TYPES:
        sub = df[df["cell_type"] == ct]
        counts = [str(int((sub["cell_state"] == s).sum())) for s in states]
        lines.append(f"| {ct} | {' | '.join(counts)} | {len(sub):,} |")
    totals = [str(int((df["cell_state"] == s).sum())) for s in states]
    lines.append(f"| **Total** | {' | '.join(totals)} | {n_cells:,} |")

    lines += [
        "",
        "## Cell Type × State — Row %\n",
        "| cell_type | nonproliferative % | proliferative % | dead % |",
        "|---|---|---|---|",
    ]
    for ct in CELL_TYPES:
        sub = df[df["cell_type"] == ct]
        pcts = [f"{(sub['cell_state'] == s).sum() / len(sub) * 100:.1f}%" for s in states]
        lines.append(f"| {ct} | {' | '.join(pcts)} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_path}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) if args.save_dir else data_dir / "cell_vis"
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)

    plot_violin_area(df, save_dir / "violin_area.png")
    plot_violin_circularity(df, save_dir / "violin_circularity.png")
    plot_violin_markers(df, save_dir / "violin_markers.png")
    write_summary_md(df, data_dir / "cell_summary.md")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
conda run -n he-feature-vis pytest tests/test_cell_vis.py tests/test_compute_shape_features.py -v
```

Expected: all 15 tests PASS

- [ ] **Step 5: Smoke test on real data (compute_shape_features)**

```bash
conda run -n he-feature-vis python scripts/compute_shape_features.py \
    --data-dir processed_crc33/
```

Expected output (approximate):
```
Processing 10379 patch masks...
Matched 258xxx/263446 cells (98.x%).
Saved 263446 records → processed_crc33/cell_shape_features.csv
```

If unmatched rate > 5%: investigate centroid coordinate convention (x=col, y=row assumed).

- [ ] **Step 6: Smoke test on real data (cell_vis)**

```bash
conda run -n he-feature-vis python scripts/cell_vis.py \
    --data-dir processed_crc33/ \
    --save-dir processed_crc33/cell_vis/
```

Expected output:
```
Saved processed_crc33/cell_vis/violin_area.png
Saved processed_crc33/cell_vis/violin_circularity.png
Saved processed_crc33/cell_vis/violin_markers.png
Saved processed_crc33/cell_summary.md
Done.
```

Verify PNGs exist and are non-empty:
```bash
ls -lh processed_crc33/cell_vis/
```

- [ ] **Step 7: Commit**

```bash
git add scripts/cell_vis.py tests/test_cell_vis.py
git commit -m "feat: add write_summary_md and main() — cell_vis complete"
```
