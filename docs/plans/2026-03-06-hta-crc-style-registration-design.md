# Design: HTA-Style H&E↔CyCIF Registration for Same-Section Pairs

**Date:** 2026-03-06
**Status:** APPROVED (user-reviewed)
**Primary files:** `stages/patchify.py`, `tools/debug_match_he_mul.py`, `test.py`

## Context

Current registration in this repo is based on global ECC affine alignment between tissue masks.
For some pairs, this is insufficient for patch-level overlap quality, even when global overlap looks acceptable.

For this project, `WD-76845-096` (H&E) and `WD-76845-097` (CyCIF) should be treated as the same physical section in different modalities.
CyCIF channels are expected to be internally registered upstream, but this should still be verified once per CyCIF image.

## Goals

1. Reproduce HTA-like registration behavior pragmatically in this repo.
2. Use automatic alignment first.
3. Trigger deformable refinement only when QC fails.
4. Avoid repeated manual work by saving and reusing transforms per slide pair.
5. Keep current debug behavior (`coords[:10]`) for fast iteration.

## Non-Goals

1. Re-implement full MCMICRO/ASHLAR raw-tile pipeline in this repo.
2. Build a general-purpose GUI registration tool in this phase.
3. Guarantee perfect alignment for all modality pairs without QC/manual fallback.

## Method Summary (HTA-Inspired)

1. Upstream CyCIF channel/cycle alignment (ASHLAR-like) to create one internally consistent CyCIF image.
2. Global rigid/affine registration for coarse H&E↔CyCIF alignment.
3. Landmark-based geometric correction when global affine is insufficient.
4. Optional deformable refinement (Demons/TPS/B-spline style) for local residual mismatch.
5. Strict QC and transform curation.

This repo will implement an equivalent workflow adapted to existing code and dependencies.

## Chosen Approach

Use an auto-first pipeline with QC gates and conditional fallback:

1. Verify CyCIF internal drift once per CyCIF file.
2. Run automatic affine registration.
3. Evaluate global and patch-level QC.
4. If QC passes, save and reuse transform.
5. If QC fails:
  - run semi-automatic landmarks when global alignment is poor;
  - run deformable refinement when global alignment is acceptable but local mismatch remains.

## Architecture

### 1. Registration Artifacts

Create and persist in `processed_*/registration/`:

- `affine.json`:
  - `warp_matrix` (2x3 H&E->MX)
  - affine diagnostics
- `qc_metrics.json`:
  - channel drift metrics
  - global QC metrics
  - patch QC metrics
  - decision path
- `landmarks.json` (optional):
  - paired control points and method metadata
- `deform_field.npz` (optional):
  - dense displacement map(s)
- `final_transform.json`:
  - canonical transform metadata used by patch extraction

### 2. QC Gate Logic

#### CyCIF Internal Drift Check (once per CyCIF file)

- Reference channel: `DNA` (or configured channel index).
- Method: phase-correlation shift per channel vs reference at overview scale.
- Pass:
  - median drift <= 1.5 overview pixels
  - max drift <= 4.0 overview pixels

#### Global Registration QC

- Tissue IoU in overview space after affine warp.
- Tissue centroid offset (as % slide diagonal).
- Scale sanity against expected image-size ratio.
- Pass:
  - IoU >= 0.75
  - centroid offset <= 3%
  - relative scale error <= 10%

#### Patch-Level QC

- Sample `N=50` tissue patches (`[:10]` remains for debug mode).
- Metrics:
  - registration gain (gradient NCC or MI delta vs baseline)
  - inside-FOV fraction after warp
- Pass:
  - > =80% of sampled patches improve over baseline
  - median gain > 0
  - > =95% of patches have inside-FOV >= 0.85

### 3. Decision Tree

1. Run affine.
2. Run QC.
3. If pass: accept affine.
4. If fail:
  - if global QC fails: use landmarks, rerun QC.
  - if global QC passes but patch QC fails: run deformable refinement, rerun QC.
5. Persist final transform and branch metadata.

### 4. Patch Extraction Integration

- Keep current affine-aware sampling path.
- Extend read path to support optional dense field warping for multiplex patch extraction.
- Ensure outside-FOV behavior remains deterministic (zero-padding + inside mask).

## Error Handling

1. If CyCIF drift check fails hard (missing channels/empty data), log and continue to affine with warning.
2. If affine fails, fall back to scale-only and mark QC as failed-required.
3. If landmarks/deformable fails, keep best previous transform and return explicit failure reason.
4. Never silently claim success when QC thresholds are not met.

## Testing Strategy (TDD)

1. Unit tests:
  - drift metric computation
  - global QC metrics and thresholds
  - patch-level QC scoring and thresholds
  - decision tree branch selection
2. Integration tests:
  - affine-pass path does not invoke deformable
  - forced global-fail invokes landmarks
  - forced local-fail invokes deformable
3. Regression tests:
  - existing affine patch extraction behavior remains valid
  - debug cap `coords[:10]` remains active

## Acceptance Criteria

1. For target pair (`096/097`), pipeline outputs QC report and deterministic final transform artifacts.
2. When QC passes after affine, no deformable stage is executed.
3. When QC fails in controlled tests, fallback branch triggers correctly.
4. Patch-level overlay quality improves relative to current ECC-only baseline on sampled regions.

## Rollout

1. Implement QC metrics and gating first.
2. Implement fallback branch scaffolding with clear artifact contracts.
3. Add deformable stage and integrate with patch extraction.
4. Update README with decision tree and usage examples.

