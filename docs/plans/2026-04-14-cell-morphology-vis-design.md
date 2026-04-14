# Cell Morphology Visualization — Design Spec

**Date:** 2026-04-14  
**Dataset:** CRC33 (`processed_crc33/`)  
**Goal:** Pipeline QC (primary) + SI figure (secondary). Characterize cancer/immune/healthy cell populations by morphology and marker intensity.

---

## Summary

Two-script pipeline:
1. `scripts/compute_shape_features.py` — compute per-cell circularity from binary patch masks
2. `scripts/cell_vis.py` — generate violin plots + markdown summary

---

## Data Context

- 263,446 cells across 10,379 patches
- Cell types: cancer (41.9%), healthy (44.0%), immune (14.1%)
- Cell states: nonproliferative (74.8%), proliferative (24.9%), dead (0.3%)
- Notable: cancer cells are 48% proliferative vs ~8-11% for immune/healthy

### Cross-tabulation (type × state)

| cell_type | nonproliferative | proliferative | dead | Total |
|---|---|---|---|---|
| cancer | 57,025 | 53,027 | 439 | 110,491 |
| immune | 33,021 | 3,943 | 159 | 37,123 |
| healthy | 106,944 | 8,767 | 121 | 115,832 |
| **Total** | 196,990 | 65,737 | 719 | 263,446 |

| cell_type | nonproliferative % | proliferative % | dead % |
|---|---|---|---|
| cancer | 51.6% | 48.0% | 0.4% |
| immune | 89.0% | 10.6% | 0.4% |
| healthy | 92.3% | 7.6% | 0.1% |

---

## Script 1: compute_shape_features.py

**Purpose:** Extract per-cell morphology features from binary patch masks.

**Inputs:**
- `processed_crc33/cell_masks/*.png` — binary masks (uint8, 0/255), one PNG per patch, filename = `{x}_{y}.png`
- `processed_crc33/cell_assignments.csv` — cell table with `centroid_x_local`, `centroid_y_local`, `CellID`, `PatchID`

**Algorithm:**
1. For each patch PNG: threshold to binary, `skimage.measure.label()` → labeled instance mask
2. `skimage.measure.regionprops()` → per-region: `area`, `perimeter`, `centroid`
3. Circularity = `4π · area / perimeter²` (1.0 = perfect circle, <1 = irregular)
4. Match each cell in `cell_assignments.csv` to a region by checking if `(centroid_x_local, centroid_y_local)` falls within the region's bounding box and label mask
5. Unmatched cells: log count; warn + continue if unmatched rate > 5%

**Output:**
- `processed_crc33/cell_shape_features.csv` — columns: `CellID, PatchID, area_px, perimeter_px, circularity`

**CLI:**
```
python scripts/compute_shape_features.py --data-dir processed_crc33/
```

---

## Script 2: cell_vis.py

**Purpose:** Generate violin plots and markdown summary for cell type characterization.

**Inputs:**
- `processed_crc33/cell_assignments.csv`
- `processed_crc33/cell_shape_features.csv` (output of Script 1)

**Outputs:** saved to `--save-dir` (default: `processed_crc33/cell_vis/`)

### Figures

1. **`violin_area.png`**
   - X-axis: cell type (cancer / immune / healthy)
   - Y-axis: `Area_cellvit_px` (raw, log-scaled if skewed)
   - One violin per cell type, colored by type

2. **`violin_circularity.png`**
   - X-axis: cell type
   - Y-axis: circularity (0–1)
   - One violin per cell type

3. **`violin_markers.png`**
   - All 18 multiplex markers: Hoechst, AF1, CD31, CD45, CD68, Argo550, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e, CD163, E-cadherin, PD-1, Ki67, Pan-CK, SMA
   - Z-score normalize each marker independently across all cells
   - Grouped violin: X-axis = marker, 3 overlaid violins (cancer/immune/healthy), colored by type
   - Split into 2 rows (~9–10 markers each) for readability
   - Y-axis label: "Z-score intensity"

### Markdown Summary: cell_summary.md

Saved to `processed_crc33/cell_summary.md`. Contains:
- Total cell count + patch count
- Count table (cell_type breakdown)
- State table (cell_state breakdown)
- Cross-tab table (type × state, counts + row %)
- Key observations (cancer proliferation rate, marker separation)

**CLI:**
```
python scripts/cell_vis.py --data-dir processed_crc33/ --save-dir processed_crc33/cell_vis/
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `cell_shape_features.csv` missing | Fail with clear message: "Run compute_shape_features.py first" |
| Cell unmatched to mask region (>5%) | Print warning with count; continue |
| Missing expected CSV column | Raise `KeyError` with column name |
| Marker column absent from CSV | Skip that marker, warn |

---

## File Layout

```
scripts/
  compute_shape_features.py
  cell_vis.py
processed_crc33/
  cell_shape_features.csv        # output of Script 1
  cell_vis/
    violin_area.png
    violin_circularity.png
    violin_markers.png
  cell_summary.md
```

---

## Out of Scope

- Per-state breakdown within each cell type (nonproliferative vs proliferative violin — not in this spec)
- H&E pixel intensity extraction (not in this spec)
- Statistical significance testing between groups
