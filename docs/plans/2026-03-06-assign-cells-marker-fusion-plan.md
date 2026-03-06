# Assign Cells Marker-CellViT Fusion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the cell type marker panel, add CellViT-based fallback for no-match cells, and track agreement confidence between marker and morphology signals.

**Architecture:** `assign_type()` uses a grouped any-positive rule over an expanded marker set; `match_cells()` falls back to `CELLVIT_TYPE_MAP` when no CSV row is within `max_dist`; a `cell_type_confidence` field is written per cell and aggregated into `cell_summary.json`. Thresholds are tunable via `--type-percentile`, `--state-percentile`, and `--thresholds-config`.

**Tech Stack:** Python 3.13, pandas, scipy.spatial.KDTree, cv2, pytest

---

### Task 1: Refactor `TYPE_MARKERS` and `compute_thresholds()`

**Files:**
- Modify: `stages/assign_cells.py`
- Test: `tests/test_assign_cells.py`

**Step 1: Write failing tests for new `compute_thresholds()` signature**

Add to `tests/test_assign_cells.py`:

```python
def test_compute_thresholds_default_percentile():
    """compute_thresholds uses default_type_percentile for all type markers."""
    from stages.assign_cells import compute_thresholds

    df = pd.DataFrame({
        "Keratin": [0.0, 50.0, 100.0],
        "CD45":    [0.0, 50.0, 100.0],
    })
    thresholds = compute_thresholds(df, default_type_percentile=50)
    # 50th percentile of [0, 50, 100] == 50.0
    assert thresholds["Keratin"] == pytest.approx(50.0)
    assert thresholds["CD45"] == pytest.approx(50.0)


def test_compute_thresholds_config_override():
    """Per-marker config_overrides replace the default percentile for that marker."""
    from stages.assign_cells import compute_thresholds

    df = pd.DataFrame({
        "Keratin": [0.0, 50.0, 100.0],
        "CD45":    [0.0, 50.0, 100.0],
    })
    thresholds = compute_thresholds(
        df,
        default_type_percentile=95,
        config_overrides={"Keratin": 50},
    )
    assert thresholds["Keratin"] == pytest.approx(50.0)  # overridden to p50
    assert thresholds["CD45"] == pytest.approx(100.0)    # p95 of [0,50,100]


def test_compute_thresholds_new_markers_covered():
    """compute_thresholds produces thresholds for all expanded type markers."""
    from stages.assign_cells import compute_thresholds

    new_markers = ["NaKATPase", "CDX2", "CD3", "CD4", "CD8a", "CD20",
                   "CD45RO", "CD68", "CD163", "FOXP3", "PD1", "Desmin", "Collagen"]
    data = {m: [0.0, 50.0, 100.0] for m in new_markers}
    df = pd.DataFrame(data)
    thresholds = compute_thresholds(df)
    for m in new_markers:
        assert m in thresholds, f"Threshold missing for marker '{m}'"
        assert thresholds[m] < float("inf"), f"Threshold for '{m}' should not be inf when column exists"


def test_compute_thresholds_missing_marker_is_inf():
    """Missing markers get threshold=inf (skipped without error)."""
    from stages.assign_cells import compute_thresholds

    df = pd.DataFrame({"Keratin": [0.0, 50.0, 100.0]})
    thresholds = compute_thresholds(df)
    assert thresholds["NaKATPase"] == float("inf")
    assert thresholds["CD3"] == float("inf")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assign_cells.py::test_compute_thresholds_default_percentile \
       tests/test_assign_cells.py::test_compute_thresholds_config_override \
       tests/test_assign_cells.py::test_compute_thresholds_new_markers_covered \
       tests/test_assign_cells.py::test_compute_thresholds_missing_marker_is_inf -v
```

Expected: FAIL (TypeError or wrong values)

**Step 3: Update `TYPE_MARKERS` and `compute_thresholds()` in `stages/assign_cells.py`**

Replace the existing `TYPE_MARKERS` constant and `compute_thresholds()` function:

```python
TYPE_MARKERS: dict[str, list[str]] = {
    "tumor":   ["Keratin", "NaKATPase", "CDX2"],
    "immune":  ["CD45", "CD3", "CD4", "CD8a", "CD20",
                "CD45RO", "CD68", "CD163", "FOXP3", "PD1"],
    "stromal": ["aSMA", "CD31", "Desmin", "Collagen"],
}


def compute_thresholds(
    df: pd.DataFrame,
    default_type_percentile: float = 95,
    default_state_percentile: float = 75,
    config_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute per-marker thresholds from the full CSV.

    Type markers: each marker in TYPE_MARKERS groups uses default_type_percentile
    unless overridden in config_overrides.

    State markers:
      Ki67, PCNA, Vimentin: default_state_percentile (default 75)
      Ecadherin low (EMT):  25th percentile
      Ecadherin high:       50th percentile

    config_overrides: dict mapping marker name → percentile to use instead of default.
    Missing columns → threshold set to inf (marker silently skipped).
    """
    if config_overrides is None:
        config_overrides = {}

    thresholds: dict[str, float] = {}

    # All type markers (flattened from groups)
    all_type_markers = [m for markers in TYPE_MARKERS.values() for m in markers]
    for marker in all_type_markers:
        percentile = config_overrides.get(marker, default_type_percentile)
        if marker in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thresholds[marker] = float(
                    np.nanpercentile(df[marker].to_numpy(dtype=float), percentile)
                )
        else:
            logging.warning("Marker '%s' not found in CSV; threshold set to inf.", marker)
            thresholds[marker] = float("inf")

    # State markers
    for marker in ["Ki67", "PCNA", "Vimentin"]:
        percentile = config_overrides.get(marker, default_state_percentile)
        if marker in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                thresholds[marker] = float(
                    np.nanpercentile(df[marker].to_numpy(dtype=float), percentile)
                )
        else:
            logging.warning("Marker '%s' not found in CSV; threshold set to inf.", marker)
            thresholds[marker] = float("inf")

    # Ecadherin: 25th pct = low (EMT); 50th pct = high (intact junctions)
    ecad_low_pct = config_overrides.get("Ecadherin", 25)
    ecad_high_pct = config_overrides.get("Ecadherin_high", 50)
    if "Ecadherin" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = df["Ecadherin"].to_numpy(dtype=float)
            thresholds["Ecadherin"] = float(np.nanpercentile(vals, ecad_low_pct))
            thresholds["Ecadherin_high"] = float(np.nanpercentile(vals, ecad_high_pct))
    else:
        logging.warning("Marker 'Ecadherin' not found in CSV; thresholds set to extremes.")
        thresholds["Ecadherin"] = float("-inf")
        thresholds["Ecadherin_high"] = float("inf")

    return thresholds
```

Also remove the old `STATE_MARKERS` constant (it was `["Ki67", "PCNA", "Vimentin", "Ecadherin"]`) — state markers are now handled inline in `compute_thresholds()`.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_assign_cells.py::test_compute_thresholds_default_percentile \
       tests/test_assign_cells.py::test_compute_thresholds_config_override \
       tests/test_assign_cells.py::test_compute_thresholds_new_markers_covered \
       tests/test_assign_cells.py::test_compute_thresholds_missing_marker_is_inf -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add stages/assign_cells.py tests/test_assign_cells.py
git commit -m "feat: expand TYPE_MARKERS panel and add tunable compute_thresholds()"
```

---

### Task 2: Update `assign_type()` for grouped any-positive rule

**Files:**
- Modify: `stages/assign_cells.py`
- Test: `tests/test_assign_cells.py`

**Step 1: Write failing tests**

Add to `tests/test_assign_cells.py`:

```python
def test_assign_type_any_positive_immune():
    """Any single immune marker above threshold → 'immune'."""
    from stages.assign_cells import assign_type

    immune_markers = ["CD45", "CD3", "CD4", "CD8a", "CD20",
                      "CD45RO", "CD68", "CD163", "FOXP3", "PD1"]
    thresholds = {m: 500.0 for m in immune_markers}
    thresholds.update({"Keratin": 500.0, "NaKATPase": 500.0, "CDX2": 500.0,
                       "aSMA": 500.0, "CD31": 500.0, "Desmin": 500.0, "Collagen": 500.0})

    for marker in immune_markers:
        row = pd.Series({m: (1000.0 if m == marker else 0.0)
                         for m in thresholds})
        result = assign_type(row, thresholds)
        assert result == "immune", f"Only {marker} high should yield 'immune', got '{result}'"


def test_assign_type_any_positive_tumor():
    """NaKATPase or CDX2 above threshold (without Keratin) → 'tumor'."""
    from stages.assign_cells import assign_type

    all_markers = ["Keratin", "NaKATPase", "CDX2", "CD45", "CD3", "CD4", "CD8a",
                   "CD20", "CD45RO", "CD68", "CD163", "FOXP3", "PD1",
                   "aSMA", "CD31", "Desmin", "Collagen"]
    thresholds = {m: 500.0 for m in all_markers}

    for marker in ["NaKATPase", "CDX2"]:
        row = pd.Series({m: (1000.0 if m == marker else 0.0) for m in all_markers})
        result = assign_type(row, thresholds)
        assert result == "tumor", f"Only {marker} high should yield 'tumor', got '{result}'"


def test_assign_type_any_positive_stromal():
    """Desmin or Collagen above threshold (without tumor/immune) → 'stromal'."""
    from stages.assign_cells import assign_type

    all_markers = ["Keratin", "NaKATPase", "CDX2", "CD45", "CD3", "CD4", "CD8a",
                   "CD20", "CD45RO", "CD68", "CD163", "FOXP3", "PD1",
                   "aSMA", "CD31", "Desmin", "Collagen"]
    thresholds = {m: 500.0 for m in all_markers}

    for marker in ["Desmin", "Collagen"]:
        row = pd.Series({m: (1000.0 if m == marker else 0.0) for m in all_markers})
        result = assign_type(row, thresholds)
        assert result == "stromal", f"Only {marker} high should yield 'stromal', got '{result}'"


def test_assign_type_tumor_beats_immune():
    """tumor > immune priority: Keratin + CD45 both high → 'tumor'."""
    from stages.assign_cells import assign_type

    thresholds = {"Keratin": 500.0, "NaKATPase": 500.0, "CDX2": 500.0,
                  "CD45": 500.0, "CD3": 500.0, "CD4": 500.0, "CD8a": 500.0,
                  "CD20": 500.0, "CD45RO": 500.0, "CD68": 500.0, "CD163": 500.0,
                  "FOXP3": 500.0, "PD1": 500.0,
                  "aSMA": 500.0, "CD31": 500.0, "Desmin": 500.0, "Collagen": 500.0}
    row = pd.Series({**{m: 0.0 for m in thresholds},
                     "Keratin": 1000.0, "CD45": 1000.0})
    assert assign_type(row, thresholds) == "tumor"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assign_cells.py::test_assign_type_any_positive_immune \
       tests/test_assign_cells.py::test_assign_type_any_positive_tumor \
       tests/test_assign_cells.py::test_assign_type_any_positive_stromal \
       tests/test_assign_cells.py::test_assign_type_tumor_beats_immune -v
```

Expected: FAIL (old `assign_type` doesn't know new markers)

**Step 3: Rewrite `assign_type()` in `stages/assign_cells.py`**

Replace the existing function:

```python
def assign_type(row: pd.Series, thresholds: dict[str, float]) -> str:
    """Assign cell type using grouped any-positive rule. Priority: tumor > immune > stromal > other.

    For each group in TYPE_MARKERS, returns that type if ANY marker in the group
    exceeds its individual threshold. Missing markers treated as 0. Never raises."""

    def _get(key: str) -> float:
        try:
            val = row.get(key, 0) if hasattr(row, "get") else getattr(row, key, 0)
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0

    try:
        for type_name, markers in TYPE_MARKERS.items():
            for marker in markers:
                if _get(marker) >= thresholds.get(marker, float("inf")):
                    return type_name
    except Exception:
        pass

    return "other"
```

**Step 4: Update existing `test_assign_type_priority_order` and `test_assign_type_missing_markers`**

The existing tests use the old 4-marker flat thresholds dict and check for `"vasculature"` type which was removed. Update them to use the new expanded thresholds. Replace both tests:

```python
def test_assign_type_priority_order():
    """assign_type follows tumor > immune > stromal > other priority."""
    from stages.assign_cells import assign_type

    thresholds = {
        "Keratin": 500.0, "NaKATPase": 500.0, "CDX2": 500.0,
        "CD45": 500.0, "CD3": 500.0, "CD4": 500.0, "CD8a": 500.0,
        "CD20": 500.0, "CD45RO": 500.0, "CD68": 500.0, "CD163": 500.0,
        "FOXP3": 500.0, "PD1": 500.0,
        "aSMA": 500.0, "CD31": 500.0, "Desmin": 500.0, "Collagen": 500.0,
    }

    # All markers high → tumor wins
    all_high = pd.Series({m: 1000.0 for m in thresholds})
    assert assign_type(all_high, thresholds) == "tumor"

    # Only CD45 and aSMA high → immune wins over stromal
    row = pd.Series({**{m: 0.0 for m in thresholds}, "CD45": 1000.0, "aSMA": 1000.0})
    assert assign_type(row, thresholds) == "immune"

    # Only aSMA → stromal
    row = pd.Series({**{m: 0.0 for m in thresholds}, "aSMA": 1000.0})
    assert assign_type(row, thresholds) == "stromal"

    # Nothing above threshold → other
    row = pd.Series({m: 0.0 for m in thresholds})
    assert assign_type(row, thresholds) == "other"


def test_assign_type_missing_markers():
    """assign_type treats any marker absent from the row as 0."""
    from stages.assign_cells import assign_type

    empty_row = pd.Series(dtype=float)
    thresholds = {"Keratin": 500.0, "CD45": 500.0, "aSMA": 500.0, "CD31": 500.0}
    assert assign_type(empty_row, thresholds) == "other"
```

**Step 5: Run all assign_type tests**

```bash
pytest tests/test_assign_cells.py -k "assign_type" -v
```

Expected: all PASSED

**Step 6: Commit**

```bash
git add stages/assign_cells.py tests/test_assign_cells.py
git commit -m "feat: rewrite assign_type() with grouped any-positive rule for expanded marker panel"
```

---

### Task 3: Add `CELLVIT_TYPE_MAP`, update `match_cells()` for fallback + confidence

**Files:**
- Modify: `stages/assign_cells.py`
- Test: `tests/test_assign_cells.py`

**Step 1: Write failing tests**

Add to `tests/test_assign_cells.py`:

```python
def _expanded_thresholds() -> dict:
    """Full expanded thresholds dict for tests — all markers set to 500.0."""
    from stages.assign_cells import TYPE_MARKERS
    thresholds = {}
    for markers in TYPE_MARKERS.values():
        for m in markers:
            thresholds[m] = 500.0
    thresholds.update({
        "Ki67": 500.0, "PCNA": 500.0, "Vimentin": 500.0,
        "Ecadherin": 200.0, "Ecadherin_high": 400.0,
    })
    return thresholds


def test_match_cells_no_match_uses_cellvit_fallback():
    """No-match cells get cell_type from CELLVIT_TYPE_MAP, not 'other'."""
    from stages.assign_cells import build_csv_index, match_cells

    thresholds = _expanded_thresholds()
    df = pd.DataFrame({"Xt": [500.0], "Yt": [500.0],
                       **{m: [0.0] for m in thresholds if m not in ("Ecadherin_high",)}})
    tree = build_csv_index(df, "Xt", "Yt")

    # CellViT type 2 = Inflammatory → "immune"
    cell = _make_cell([10, 10], _small_rect_contour(10, 10), type_cellvit=2)
    result = match_cells([cell], tree, df, thresholds, x0=0, y0=0, max_dist=15.0)
    assert result[0]["cell_type"] == "immune", (
        f"No-match cell with cellvit type 2 should be 'immune', got '{result[0]['cell_type']}'"
    )
    assert result[0]["cell_type_confidence"] == "low"


def test_match_cells_no_match_cellvit_type5_is_tumor():
    """CellViT type 5 (Epithelial) → 'tumor' for no-match cells in CRC context."""
    from stages.assign_cells import build_csv_index, match_cells

    thresholds = _expanded_thresholds()
    df = pd.DataFrame({"Xt": [500.0], "Yt": [500.0],
                       **{m: [0.0] for m in thresholds if m not in ("Ecadherin_high",)}})
    tree = build_csv_index(df, "Xt", "Yt")

    cell = _make_cell([10, 10], _small_rect_contour(10, 10), type_cellvit=5)
    result = match_cells([cell], tree, df, thresholds, x0=0, y0=0, max_dist=15.0)
    assert result[0]["cell_type"] == "tumor"
    assert result[0]["cell_type_confidence"] == "low"


def test_match_cells_matched_agreement_high_confidence():
    """When marker type and CellViT type agree, confidence is 'high'."""
    from stages.assign_cells import build_csv_index, match_cells

    thresholds = _expanded_thresholds()
    # CSV cell at (128, 128) with high Keratin → marker type = "tumor"
    df = pd.DataFrame({"Xt": [128.0], "Yt": [128.0],
                       "Keratin": [1000.0],
                       **{m: [0.0] for m in thresholds
                          if m not in ("Keratin", "Ecadherin_high")}})
    tree = build_csv_index(df, "Xt", "Yt")

    # CellViT type 1 = Neoplastic → maps to "tumor" — should agree
    cell = _make_cell([128, 128], _small_rect_contour(128, 128), type_cellvit=1)
    result = match_cells([cell], tree, df, thresholds, x0=0, y0=0, max_dist=15.0)
    assert result[0]["cell_type"] == "tumor"
    assert result[0]["cell_type_confidence"] == "high"


def test_match_cells_matched_disagreement_markers_win():
    """When marker type and CellViT type disagree, markers win with 'low' confidence."""
    from stages.assign_cells import build_csv_index, match_cells

    thresholds = _expanded_thresholds()
    # CSV cell with high Keratin → marker type = "tumor"
    df = pd.DataFrame({"Xt": [128.0], "Yt": [128.0],
                       "Keratin": [1000.0],
                       **{m: [0.0] for m in thresholds
                          if m not in ("Keratin", "Ecadherin_high")}})
    tree = build_csv_index(df, "Xt", "Yt")

    # CellViT type 2 = Inflammatory → maps to "immune" — disagrees with markers
    cell = _make_cell([128, 128], _small_rect_contour(128, 128), type_cellvit=2)
    result = match_cells([cell], tree, df, thresholds, x0=0, y0=0, max_dist=15.0)
    assert result[0]["cell_type"] == "tumor", "Markers win on conflict"
    assert result[0]["cell_type_confidence"] == "low"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assign_cells.py::test_match_cells_no_match_uses_cellvit_fallback \
       tests/test_assign_cells.py::test_match_cells_no_match_cellvit_type5_is_tumor \
       tests/test_assign_cells.py::test_match_cells_matched_agreement_high_confidence \
       tests/test_assign_cells.py::test_match_cells_matched_disagreement_markers_win -v
```

Expected: FAIL (no `CELLVIT_TYPE_MAP`, no `cell_type_confidence`)

**Step 3: Add `CELLVIT_TYPE_MAP` constant and update `match_cells()` in `stages/assign_cells.py`**

Add constant after `CELL_STATE_COLORS`:

```python
CELLVIT_TYPE_MAP: dict[int, str] = {
    0: "other",    # Unknown
    1: "tumor",    # Neoplastic
    2: "immune",   # Inflammatory
    3: "stromal",  # Connective
    4: "other",    # Dead — state assignment handles apoptotic
    5: "tumor",    # Epithelial → tumor in CRC context
}
```

Replace `match_cells()`:

```python
def match_cells(
    cells: list[dict],
    kdtree,
    df: pd.DataFrame,
    thresholds: dict[str, float],
    x0: int,
    y0: int,
    max_dist: float = 15.0,
    coord_scale: float = 1.0,
) -> list[dict]:
    """Match each cell to the nearest CSV row within max_dist pixels.

    Matched cells (dist <= max_dist):
      - cell_type from expanded marker panel (authoritative)
      - cell_type_confidence = 'high' if marker_type == cellvit_type, else 'low'

    No-match cells (dist > max_dist):
      - cell_type from CELLVIT_TYPE_MAP[type_cellvit]
      - cell_type_confidence = 'low' (no marker validation)
      - cell_state = 'other' (state requires marker data)
    """
    for cell in cells:
        try:
            centroid = cell.get("centroid", [0, 0])
            lx = float(centroid[0])
            ly = float(centroid[1])
            gx = (x0 + lx) * coord_scale
            gy = (y0 + ly) * coord_scale

            dist, idx = kdtree.query([gx, gy])
            type_cellvit = int(cell.get("type_cellvit", 0))
            cellvit_type = CELLVIT_TYPE_MAP.get(type_cellvit, "other")

            if dist <= max_dist:
                matched_row = df.iloc[idx]
                marker_type = assign_type(matched_row, thresholds)
                cell["cell_type"] = marker_type
                cell["cell_type_confidence"] = "high" if marker_type == cellvit_type else "low"
                cell["cell_state"] = assign_state(matched_row, thresholds, type_cellvit)
            else:
                cell["cell_type"] = cellvit_type
                cell["cell_type_confidence"] = "low"
                cell["cell_state"] = "other"
        except Exception:
            cell["cell_type"] = "other"
            cell["cell_type_confidence"] = "low"
            cell["cell_state"] = "other"

    return cells
```

**Step 4: Update existing `test_match_cells_unmatched_when_far`**

The existing test checks that unmatched cells get `cell_type="other"`. With the fallback, CellViT type 1 now maps to `"tumor"`. Update the test to use `type_cellvit=0` (Unknown → "other") to preserve its intent:

```python
def test_match_cells_unmatched_when_far():
    """No CSV match within max_dist: cell_type comes from CELLVIT_TYPE_MAP (type 0 → 'other')."""
    from stages.assign_cells import build_csv_index, match_cells

    thresholds = _expanded_thresholds()
    df = pd.DataFrame({"Xt": [500.0], "Yt": [500.0],
                       **{m: [0.0] for m in thresholds if m not in ("Ecadherin_high",)}})
    tree = build_csv_index(df, "Xt", "Yt")

    # type_cellvit=0 (Unknown) → "other"
    cell = _make_cell([10, 10], _small_rect_contour(10, 10), type_cellvit=0)
    result = match_cells([cell], tree, df, thresholds, x0=0, y0=0, max_dist=15.0)
    assert result[0]["cell_type"] == "other"
    assert result[0]["cell_state"] == "other"
    assert result[0]["cell_type_confidence"] == "low"
```

**Step 5: Run all match_cells tests**

```bash
pytest tests/test_assign_cells.py -k "match_cells" -v
```

Expected: all PASSED

**Step 6: Commit**

```bash
git add stages/assign_cells.py tests/test_assign_cells.py
git commit -m "feat: add CELLVIT_TYPE_MAP fallback and cell_type_confidence to match_cells()"
```

---

### Task 4: Add agreement stats to `cell_summary.json`

**Files:**
- Modify: `stages/assign_cells.py`
- Test: `tests/test_assign_cells.py`

**Step 1: Write failing CLI test**

Add to `tests/test_assign_cells.py`:

```python
def test_cli_summary_contains_agreement_stats(tmp_path):
    """cell_summary.json must contain 'agreement' and 'conflict_pairs' keys."""
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    out_dir = tmp_path / "out"

    # Two cells: one matched (Keratin high, CellViT type 1 → agree),
    # one matched (CD45 high, CellViT type 1 → disagree)
    cell_data = {
        "cells": [
            {"centroid": [64, 64], "contour": _small_rect_contour(64, 64),
             "bbox": [[54, 54], [74, 74]], "type_cellvit": 1, "type_prob": 0.9},
            {"centroid": [192, 192], "contour": _small_rect_contour(192, 192),
             "bbox": [[182, 182], [202, 202]], "type_cellvit": 1, "type_prob": 0.9},
        ]
    }
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    features_path = tmp_path / "CRC02.csv"
    features_path.write_text(
        "Xt,Yt,Keratin,CD45,aSMA,CD31,NaKATPase,CDX2,CD3,CD4,CD8a,CD20,"
        "CD45RO,CD68,CD163,FOXP3,PD1,Desmin,Collagen,Ki67,PCNA,Vimentin,Ecadherin\n"
        "64,64,5000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
        "192,192,0,5000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "stride": 256, "patch_size": 256, "tissue_min": 0.0,
        "img_w": 256, "img_h": 256, "channels": [],
    }
    (tmp_path / "index.json").write_text(json.dumps(index_data))

    cmd = [sys.executable, "-m", "stages.assign_cells",
           "--cellvit-dir", str(cellvit_dir),
           "--features-csv", str(features_path),
           "--index", str(tmp_path / "index.json"),
           "--out", str(out_dir), "--max-dist", "15.0"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    summary = json.loads((out_dir / "cell_summary.json").read_text())
    assert "agreement" in summary, "cell_summary.json missing 'agreement' key"
    assert "high" in summary["agreement"]
    assert "low" in summary["agreement"]
    assert "conflict_pairs" in summary, "cell_summary.json missing 'conflict_pairs' key"
    # Cell 1: Keratin high + cellvit type 1 (tumor) → agree → high confidence
    assert summary["agreement"]["high"] >= 1
    # Cell 2: CD45 high (immune) + cellvit type 1 (tumor) → disagree → low confidence
    assert summary["agreement"]["low"] >= 1
    assert "marker=immune,cellvit=tumor" in summary["conflict_pairs"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_assign_cells.py::test_cli_summary_contains_agreement_stats -v
```

Expected: FAIL (KeyError on `agreement`)

**Step 3: Update `main()` in `stages/assign_cells.py` to track and emit agreement stats**

In the per-patch loop, after `match_cells()`, accumulate agreement counts:

```python
# After: cells = match_cells(...)

# Accumulate agreement stats
for c in cells:
    conf = c.get("cell_type_confidence", "low")
    global_agreement_counts[conf] += 1
    if conf == "low":
        marker_t = c.get("cell_type", "other")
        cv_t = CELLVIT_TYPE_MAP.get(int(c.get("type_cellvit", 0)), "other")
        if marker_t != cv_t:  # only log true conflicts (not no-match fallbacks)
            key = f"marker={marker_t},cellvit={cv_t}"
            global_conflict_pairs[key] += 1
```

Initialize before the loop:
```python
global_agreement_counts: Counter = Counter()
global_conflict_pairs: Counter = Counter()
```

Add to the `summary` dict:
```python
summary = {
    ...
    "agreement": dict(global_agreement_counts),
    "conflict_pairs": dict(global_conflict_pairs),
}
```

**Step 4: Run test**

```bash
pytest tests/test_assign_cells.py::test_cli_summary_contains_agreement_stats -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add stages/assign_cells.py tests/test_assign_cells.py
git commit -m "feat: add agreement and conflict_pairs to cell_summary.json"
```

---

### Task 5: Add CLI flags `--type-percentile`, `--state-percentile`, `--thresholds-config`

**Files:**
- Modify: `stages/assign_cells.py`
- Test: `tests/test_assign_cells.py`

**Step 1: Write failing CLI test**

Add to `tests/test_assign_cells.py`:

```python
def test_cli_custom_type_percentile(tmp_path):
    """--type-percentile is passed through to compute_thresholds()."""
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()

    # Cell at origin with Keratin=1 (very low) — will only pass threshold at p0
    cell_data = {"cells": [
        {"centroid": [64, 64], "contour": _small_rect_contour(64, 64),
         "bbox": [[54,54],[74,74]], "type_cellvit": 1, "type_prob": 0.9}
    ]}
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    features_path = tmp_path / "CRC02.csv"
    # Keratin values: [1, 100, 200] → p0=1, p95=~199
    features_path.write_text(
        "Xt,Yt,Keratin,CD45,aSMA,CD31,NaKATPase,CDX2,CD3,CD4,CD8a,CD20,"
        "CD45RO,CD68,CD163,FOXP3,PD1,Desmin,Collagen,Ki67,PCNA,Vimentin,Ecadherin\n"
        "64,64,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
        "200,200,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
        "300,300,200,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
    )
    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "stride": 256, "patch_size": 256, "tissue_min": 0.0,
        "img_w": 256, "img_h": 256, "channels": [],
    }
    (tmp_path / "index.json").write_text(json.dumps(index_data))

    out_dir = tmp_path / "out"
    cmd = [sys.executable, "-m", "stages.assign_cells",
           "--cellvit-dir", str(cellvit_dir),
           "--features-csv", str(features_path),
           "--index", str(tmp_path / "index.json"),
           "--out", str(out_dir),
           "--max-dist", "15.0",
           "--type-percentile", "0"]   # p0 threshold → Keratin threshold = 1.0
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    summary = json.loads((out_dir / "cell_summary.json").read_text())
    # With p0 threshold, Keratin=1 >= 1.0 → cell should be tumor, not other
    assert summary["cell_types"].get("tumor", 0) >= 1, (
        "With --type-percentile 0, even Keratin=1 should qualify as tumor"
    )


def test_cli_thresholds_config(tmp_path):
    """--thresholds-config JSON overrides per-marker percentile."""
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    cell_data = {"cells": [
        {"centroid": [64, 64], "contour": _small_rect_contour(64, 64),
         "bbox": [[54,54],[74,74]], "type_cellvit": 1, "type_prob": 0.9}
    ]}
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    features_path = tmp_path / "CRC02.csv"
    features_path.write_text(
        "Xt,Yt,Keratin,CD45,aSMA,CD31,NaKATPase,CDX2,CD3,CD4,CD8a,CD20,"
        "CD45RO,CD68,CD163,FOXP3,PD1,Desmin,Collagen,Ki67,PCNA,Vimentin,Ecadherin\n"
        "64,64,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
        "200,200,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
        "300,300,200,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,500\n"
    )
    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "stride": 256, "patch_size": 256, "tissue_min": 0.0,
        "img_w": 256, "img_h": 256, "channels": [],
    }
    (tmp_path / "index.json").write_text(json.dumps(index_data))

    # Override Keratin to p0 via config file
    config_path = tmp_path / "thresholds.json"
    config_path.write_text(json.dumps({"Keratin": 0}))

    out_dir = tmp_path / "out"
    cmd = [sys.executable, "-m", "stages.assign_cells",
           "--cellvit-dir", str(cellvit_dir),
           "--features-csv", str(features_path),
           "--index", str(tmp_path / "index.json"),
           "--out", str(out_dir),
           "--max-dist", "15.0",
           "--thresholds-config", str(config_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    summary = json.loads((out_dir / "cell_summary.json").read_text())
    assert summary["cell_types"].get("tumor", 0) >= 1
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assign_cells.py::test_cli_custom_type_percentile \
       tests/test_assign_cells.py::test_cli_thresholds_config -v
```

Expected: FAIL (unrecognized arguments)

**Step 3: Add CLI flags to `main()` in `stages/assign_cells.py`**

Add arguments to the `ArgumentParser`:

```python
parser.add_argument(
    "--type-percentile",
    type=float,
    default=95.0,
    help="Percentile for type marker thresholds (default: 95). "
         "Per-marker overrides via --thresholds-config take precedence.",
)
parser.add_argument(
    "--state-percentile",
    type=float,
    default=75.0,
    help="Percentile for state marker thresholds Ki67/PCNA/Vimentin (default: 75).",
)
parser.add_argument(
    "--thresholds-config",
    default=None,
    help="Path to JSON file with per-marker percentile overrides, "
         'e.g. {"CD3": 85, "CDX2": 90}.',
)
```

Load config and pass to `compute_thresholds()`:

```python
config_overrides: dict[str, float] = {}
if args.thresholds_config:
    config_path = pathlib.Path(args.thresholds_config)
    with config_path.open(encoding="utf-8") as fh:
        config_overrides = json.load(fh)
    log.info("Loaded threshold config: %s", config_overrides)

log.info("Computing per-marker thresholds (type_pct=%.0f, state_pct=%.0f) …",
         args.type_percentile, args.state_percentile)
thresholds = compute_thresholds(
    df,
    default_type_percentile=args.type_percentile,
    default_state_percentile=args.state_percentile,
    config_overrides=config_overrides,
)
```

Also store `type_percentile`, `state_percentile`, `thresholds_config` in the `summary` dict for reproducibility:

```python
summary = {
    ...
    "type_percentile": args.type_percentile,
    "state_percentile": args.state_percentile,
    "thresholds_config": args.thresholds_config,
}
```

**Step 4: Run tests**

```bash
pytest tests/test_assign_cells.py::test_cli_custom_type_percentile \
       tests/test_assign_cells.py::test_cli_thresholds_config -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add stages/assign_cells.py tests/test_assign_cells.py
git commit -m "feat: add --type-percentile, --state-percentile, --thresholds-config CLI flags"
```

---

### Task 6: Full test suite pass + update existing CLI tests

**Files:**
- Test: `tests/test_assign_cells.py`

**Step 1: Run all assign_cells tests**

```bash
pytest tests/test_assign_cells.py -v
```

Some existing CLI tests (`test_cli_creates_output_dirs`, `test_cli_skips_patch_with_no_cellvit_json`) use a CSV with only the original 4 type markers. They will still pass because missing markers get `inf` thresholds (any-positive rule skips them). Verify they pass as-is.

If any test fails due to the CSV missing new marker columns, add all new marker columns with value `0` to that test's CSV string.

**Step 2: Run the full test suite to check for regressions**

```bash
pytest -v
```

Expected: all tests PASSED

**Step 3: Commit if any test fixes were needed**

```bash
git add tests/test_assign_cells.py
git commit -m "test: fix existing CLI tests for expanded marker panel"
```

---

### Task 7: Update Stage 1 and Stage 4 default `--channels` to match Stage 3 marker panel

**Files:**
- Modify: `stages/patchify.py`
- Modify: `stages/multiplex_layers.py`

**Context:**
Stage 1 (`patchify.py`) extracts multiplex channels from the OME-TIFF into per-patch `.npy` arrays.
Stage 4 (`multiplex_layers.py`) reads those `.npy` arrays — its `--channels` must match Stage 1's exactly.
Stage 3 reads from `CRC02.csv` (all markers always available), but the `.npy` files should cover the same panel so spatial/pixel-level analysis of any Stage 3 marker is possible.

**Canonical channel list** (derived from Stage 3's expanded TYPE_MARKERS + STATE_MARKERS):
```
Keratin NaKATPase CDX2
CD45 CD3 CD4 CD8a CD20 CD45RO CD68 CD163 FOXP3 PD1
aSMA CD31 Desmin Collagen
Ki67 PCNA Vimentin Ecadherin
```
(21 channels)

**Step 1: Update Stage 1 default in `stages/patchify.py`**

Find the `--channels` argument (line ~492):
```python
"--channels", nargs="+", default=["CD31", "Ki67", "CD45", "PCNA"]
```

Replace with:
```python
"--channels",
nargs="+",
default=[
    "Keratin", "NaKATPase", "CDX2",
    "CD45", "CD3", "CD4", "CD8a", "CD20", "CD45RO", "CD68", "CD163", "FOXP3", "PD1",
    "aSMA", "CD31", "Desmin", "Collagen",
    "Ki67", "PCNA", "Vimentin", "Ecadherin",
],
metavar="NAME",
help="Multiplex channel names to extract (default: full Stage 3 marker panel, 21 channels).",
```

**Step 2: Update Stage 4 default in `stages/multiplex_layers.py`**

Find the `--channels` argument (line ~211):
```python
"--channels", nargs="+", default=["CD31", "Ki67", "PCNA"]
```

Replace with the same 21-channel list:
```python
"--channels",
nargs="+",
default=[
    "Keratin", "NaKATPase", "CDX2",
    "CD45", "CD3", "CD4", "CD8a", "CD20", "CD45RO", "CD68", "CD163", "FOXP3", "PD1",
    "aSMA", "CD31", "Desmin", "Collagen",
    "Ki67", "PCNA", "Vimentin", "Ecadherin",
],
metavar="NAME",
help="Multiplex channel names present in the .npy files, in order "
     "(must match --channels passed to patchify.py). "
     "Default: full Stage 3 marker panel, 21 channels.",
```

**Step 3: Verify Stage 4 channel index lookups still work**

Stage 4 looks up specific channel positions inside the `.npy` with `get_channel_index(args.channels, "CD31")` etc. Since CD31, Ki67, PCNA are all present in the new default list, these lookups will succeed. Run Stage 4's tests:

```bash
pytest tests/test_multiplex_layers.py -v
```

Expected: all PASSED (no index-out-of-range errors)

**Step 4: Run full test suite**

```bash
pytest -v
```

Expected: all PASSED

**Step 5: Commit**

```bash
git add stages/patchify.py stages/multiplex_layers.py
git commit -m "feat: update Stage 1 and Stage 4 default --channels to full 21-channel marker panel"
```
