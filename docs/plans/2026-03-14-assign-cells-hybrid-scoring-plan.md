# Assign Cells Hybrid Scoring — Implementation Plan

**Date:** 2026-03-14  
**Depends on design:** `docs/plans/2026-03-14-assign-cells-hybrid-scoring-design.md`

## Objective

Implement the approved MX-first hybrid scoring classifier in Stage 3:

- types: `cancer`, `immune`, `healthy`
- states: `dead`, `proliferative`, `quiescent`
- fused score: `0.75 * MX + 0.25 * CellViT prior`
- split epithelial prior

## Task 1: Add scoring config and marker groups

**Files**
- Modify: `stages/assign_cells.py`

**Changes**
- Add canonical class names/constants for 3 types and 3 states.
- Add panel definitions for:
  - immune T-cell panel
  - immune myeloid/B-cell panel
  - stromal panel
- Add CellViT prior map for new class set, including epithelial split prior.

**Validation**
- New unit tests: constants and prior mapping for CellViT IDs.

## Task 2: Add percentile normalization utilities

**Files**
- Modify: `stages/assign_cells.py`
- Optionally modify: `utils/marker_aliases.py` (if aliases are missing)

**Changes**
- Implement per-slide percentile rank normalization helper.
- Resolve aliases consistently (Pan-CK/E-cadherin/PD-1/SMA/CD3e).
- Implement available-term weighted mean with automatic weight renormalization.

**Validation**
- Unit tests for:
  - percentile normalization range `[0,1]`
  - monotonicity of normalized values
  - missing-column handling and weight renormalization

## Task 3: Implement MX score and fused score

**Files**
- Modify: `stages/assign_cells.py`

**Changes**
- Implement:
  - `compute_mx_scores(row_norm)`
  - `compute_cellvit_prior(type_cellvit)`
  - `compute_fused_scores(mx_scores, cv_prior, w_mx=0.75, w_cv=0.25)`
- Replace current type assignment call path with fused argmax.

**Validation**
- Unit tests for deterministic scoring:
  - strong CD45 -> immune
  - strong PanCK+Ecad and low CD45 -> cancer
  - stromal signature and low immune/cancer -> healthy
  - epithelial prior split effect

## Task 4: Implement state redesign

**Files**
- Modify: `stages/assign_cells.py`

**Changes**
- Replace state logic with:
  1. dead if CellViT type 4
  2. proliferative if `Ki67_norm >= 0.75`
  3. quiescent otherwise

**Validation**
- Unit tests:
  - dead override always wins
  - Ki67 p75 boundary behavior
  - non-dead, low Ki67 -> quiescent

## Task 5: Mismatch and confidence outputs

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `tests/test_assign_cells.py`

**Changes**
- Add mismatch fields:
  - `mx_top_type`
  - `cellvit_prior_type`
  - `is_mismatch`
- Add confidence rule:
  - high/medium/low by fused margin
  - downgrade by one level on mismatch

**Validation**
- Unit tests for confidence bins and downgrade behavior.
- CLI test that summary includes mismatch/confidence counts.

## Task 6: Update summary schema and backward compatibility

**Files**
- Modify: `stages/assign_cells.py`
- Modify: `README.md`

**Changes**
- Ensure `cell_summary.json` includes:
  - type/state counts with new labels
  - mismatch rate and conflict counts
  - confidence distribution
  - missing/available channel stats
- Keep output PNG directories unchanged.

**Validation**
- CLI integration tests pass for:
  - precomputed CSV mode
  - auto-extract mode

## Task 7: Expand/refresh tests

**Files**
- Modify: `tests/test_assign_cells.py`

**Add tests**
- Hybrid scoring formula tests
- Split epithelial prior test
- Alias coverage tests for marker names
- Missing marker resilience tests
- End-to-end CLI tests for both modes

**Run**
```bash
pytest tests/test_assign_cells.py -v
```

## Task 8: Documentation updates

**Files**
- Modify: `README.md`
- Optional addendum: `docs/plans/2026-03-06-assign-cells-marker-fusion-plan.md`

**Changes**
- Replace old type/state wording with new taxonomy.
- Add concise description of hybrid scoring and confidence semantics.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Missing channels on some datasets | Unstable class scores | Available-term renormalization + summary channel coverage |
| Over-trusting CellViT prior | Type drift from marker biology | Keep fixed MX-first weights (0.75/0.25) |
| Ki67-only state may overcall proliferation | State noise | Keep dead override; add Ki67 distribution QC in summary |
| Schema transition confusion | Downstream breakage | Preserve existing output folders and add summary version notes |

## Acceptance criteria

- All `assign_cells` tests pass.
- Both CLI modes succeed.
- New labels used end-to-end:
  - types: cancer/immune/healthy
  - states: dead/proliferative/quiescent
- Summary contains mismatch and confidence metrics.
