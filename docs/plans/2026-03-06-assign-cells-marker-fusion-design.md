# Design: Marker-CellViT Fusion for Cell Type Assignment

**Date:** 2026-03-06
**Status:** DRAFT
**File:** `stages/assign_cells.py`

## Motivation

Currently `assign_type()` uses only four markers (Keratin, CD45, aSMA, CD31). Two problems:

1. **No-match cells** (CellViT detected but outside `max_dist` of any CSV row) always get `"other"` — wasted signal from CellViT morphology
2. **Narrow marker panel** — many immune subtypes (CD8a, CD68, FOXP3, etc.) and additional tumor/stromal markers in the CSV are ignored

## Goals

- Expand the type marker panel using an any-positive rule per group
- Fall back to CellViT morphological type for no-match cells
- Track agreement between marker-based and CellViT-based type per matched cell
- Make thresholds tunable via CLI flag and per-marker config file

## Design

### 1. Expanded Marker Panel

Replace flat `TYPE_MARKERS` list with a grouped dict:

```python
TYPE_MARKERS = {
    "tumor":   ["Keratin", "NaKATPase", "CDX2"],
    "immune":  ["CD45", "CD3", "CD4", "CD8a", "CD20",
                "CD45RO", "CD68", "CD163", "FOXP3", "PD1"],
    "stromal": ["aSMA", "CD31", "Desmin", "Collagen"],
}
```

**Any-positive rule:** if any marker in a group exceeds its threshold, assign that type.
**Priority:** tumor > immune > stromal > other (unchanged).
Each marker gets an individual 95th-percentile threshold (not group-level).
Missing columns → threshold set to `inf` (marker skipped silently).

### 2. Tunable Thresholds

`compute_thresholds()` accepts:
- `default_type_percentile: float` (default 95) — applied to all type markers
- `default_state_percentile: float` (default 75 for Ki67/PCNA/Vimentin) — applied to state markers
- `config_overrides: dict[str, float]` — per-marker percentile overrides

Config file format (`--thresholds-config thresholds.json`):
```json
{
  "CD3": 85,
  "CDX2": 90,
  "Collagen": 80
}
```

CLI flags added:
- `--type-percentile FLOAT` (default: 95)
- `--state-percentile FLOAT` (default: 75)
- `--thresholds-config PATH` (optional JSON file)

### 3. CellViT Fallback for No-Match Cells

New constant:
```python
CELLVIT_TYPE_MAP = {
    0: "other",    # Unknown
    1: "tumor",    # Neoplastic
    2: "immune",   # Inflammatory
    3: "stromal",  # Connective
    4: "other",    # Dead — state handles apoptotic
    5: "tumor",    # Epithelial → tumor in CRC context
}
```

In `match_cells()`:
- `dist <= max_dist` → assign type from expanded marker panel (authoritative)
- `dist > max_dist` → assign type from `CELLVIT_TYPE_MAP[type_cellvit]`

`cell_state` for no-match cells stays `"other"` — state requires marker data.

### 4. Agreement Tracking (Confidence)

For every **matched** cell, compute both:
- `marker_type` — from expanded marker panel
- `cellvit_type` — from `CELLVIT_TYPE_MAP[type_cellvit]`

Set `cell_type_confidence`:
- `"high"` — marker_type == cellvit_type
- `"low"` — marker_type != cellvit_type (markers win)

For **no-match** cells: `cell_type_confidence = "low"` always.

**New field in each cell dict:**
```json
{ "cell_type": "immune", "cell_type_confidence": "high", "cell_state": "..." }
```

**New fields in `cell_summary.json`:**
```json
{
  "agreement": { "high": 18400, "low": 2300 },
  "conflict_pairs": {
    "marker=tumor,cellvit=immune": 120,
    "marker=immune,cellvit=stromal": 45
  }
}
```

## Out of Scope

- CSV-only mode (no CellViT) — CellViT remains required
- Immune subtype classification (e.g., T_cell, B_cell) — all immune stays `"immune"`
- Changes to rasterization, state assignment, output PNG format, or `cell_states/`

## Unchanged Behavior

- State assignment logic (`assign_state()`) — untouched
- Output directory structure (`cell_types/`, `cell_states/`, `cell_summary.json`)
- Rasterized PNG format and color maps
- `match_cells()` coordinate transform logic (`coord_scale`, `max_dist`)
