# Assign Cells Astir-First — Implementation Plan

**Date:** 2026-03-14  
**Depends on design:** `docs/plans/2026-03-14-assign-cells-astir-first-design.md`

## Objective

Implement Astir-first cell type assignment while keeping current Stage 3 I/O and dual CSV modes.

Type output:
- `cancer`, `immune`, `healthy`

State output:
- `dead`, `proliferative`, `quiescent`

## Fixed marker configuration (user-provided panel)

- Type markers:
  - cancer: `Pan-CK`, `E-cadherin`
  - immune: `CD45`, `CD3e`, `CD4`, `CD8a`, `CD20`, `CD68`, `CD163`, `FOXP3`, `CD45RO`, `PD-1`
  - healthy: `SMA`, `CD31`
- State marker:
  - proliferative: `Ki67` (p75 threshold)
- Non-typing markers tracked but not used in v1:
  - `Hoechst`, `AF1`, `Argo550`, `PD-L1`

## Task 1: Add new ontology and mappings

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `tests/test_assign_cells.py`

**Changes**
- Add new type/state label constants.
- Add CellViT prior mapping including epithelial split prior.
- Add metadata version flag for summary.

**Validation**
- Unit tests for mapping correctness and label set integrity.

## Task 2: Add Astir integration layer

**Files**
- Modify: `stages/assign_cells.py`
- Optional add: `utils/astir_adapter.py`
- Modify: `requirements.txt`

**Changes**
- Add adapter function:
  - takes canonicalized marker matrix,
  - returns `P_astir(cancer/immune/healthy)` per cell.
- Add CLI flags:
  - `--classifier {astir,rule}` default `astir`
  - `--allow-astir-fallback` bool

**Validation**
- Unit tests with mocked Astir outputs.
- Dependency import/fallback behavior tests.

## Task 3: Canonical marker mapping for Astir input

**Files**
- Modify: `utils/marker_aliases.py`
- Modify: `stages/assign_cells.py`

**Changes**
- Ensure canonical mapping includes:
  - Pan-CK aliases
  - E-cadherin aliases
  - PD-1 aliases
  - CD3e -> CD3 compatibility
  - SMA aliases
- Build Astir feature matrix from available columns and log missing markers.
- Ensure v1 requires only markers available in this panel (no Desmin/Collagen/PCNA dependency).

**Validation**
- Unit tests for alias resolution and missing-marker handling.

## Task 4: Implement Astir-first fusion with CellViT prior

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `tests/test_assign_cells.py`

**Changes**
- Compute:
  - `P_astir`
  - `P_cellvit_prior`
  - `P_final = 0.85*P_astir + 0.15*P_cellvit_prior`
- Assign `cell_type = argmax(P_final)`.
- Track mismatch (`argmax(P_astir) != argmax(P_cellvit_prior)`).

**Validation**
- Unit tests for fused posterior calculations and class selection.

## Task 5: Keep locked state policy

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `tests/test_assign_cells.py`

**Changes**
- Dead override: `type_cellvit == 4 -> dead`.
- For non-dead:
  - `Ki67_norm >= p75 -> proliferative`
  - else `quiescent`
- Add warning + fallback when Ki67 missing.

**Validation**
- Boundary tests for Ki67 p75.
- Dead override precedence tests.

## Task 6: Confidence and mismatch outputs

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `tests/test_assign_cells.py`

**Changes**
- Margin-based confidence on `P_final`.
- Mismatch downgrade by one level.
- Add per-cell fields:
  - `type_astir`, `type_cellvit_prior`, `is_mismatch`
  - `p_final_*`

**Validation**
- Unit tests for confidence bins and downgrade behavior.

## Task 7: Summary schema updates

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `README.md`

**Changes**
- Add summary fields:
  - `astir_first: true`
  - confidence counts
  - mismatch rate
  - channel coverage
- Keep existing output folders and file names.

**Validation**
- CLI tests for summary schema in both input modes.

## Task 8: End-to-end test pass

**Files**
- Modify: `tests/test_assign_cells.py`

**Add tests**
- Astir-first type assignment in provided-CSV mode
- Astir-first type assignment in auto-extract mode
- Fallback path when Astir unavailable (`--allow-astir-fallback`)

**Run**
```bash
pytest tests/test_assign_cells.py -v
```

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Astir dependency friction | Runtime failures | explicit import checks + optional fallback flag |
| Missing marker channels | unstable probabilities | channel-coverage logging + confidence downgrade |
| Overriding prior too much | morphology disagreement | keep Astir-first blend fixed at 0.85/0.15 |
| Cross-dataset drift | degraded labels | marker alias map + per-run summary diagnostics |

## Acceptance criteria

- Astir-first path works in both Stage 3 modes.
- New type/state ontology is used end-to-end.
- Mismatch/confidence fields are present and tested.
- Existing PNG outputs remain compatible.
