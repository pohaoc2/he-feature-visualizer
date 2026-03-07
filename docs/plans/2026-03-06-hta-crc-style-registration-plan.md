# Plan: Implement HTA-Style Auto-First Registration with QC-Gated Fallback

**Date:** 2026-03-06
**Status:** READY
**Design reference:** `docs/plans/2026-03-06-hta-crc-style-registration-design.md`

## Phase 1: QC and Gating (No new deformable math yet)

1. Add registration artifact schema and output directory contracts.
2. Implement CyCIF channel-drift checker (overview phase correlation).
3. Implement global QC metric functions:
   - tissue IoU
   - centroid offset %
   - scale sanity error %
4. Implement patch-level QC metric functions:
   - similarity gain vs unregistered baseline
   - inside-FOV coverage
5. Implement decision gate function:
   - `PASS_AFFINE`
   - `FAIL_GLOBAL_NEEDS_LANDMARKS`
   - `FAIL_LOCAL_NEEDS_DEFORMABLE`

## Phase 2: TDD for Phase 1

1. Add unit tests for drift metrics and threshold evaluation.
2. Add unit tests for global QC metric calculations.
3. Add unit tests for patch-level QC scoring and threshold evaluation.
4. Add unit tests for decision gate branch selection.
5. Ensure current tests keep passing.

## Phase 3: Pipeline Wiring

1. Wire QC collection into `stages.patchify` after affine estimation.
2. Save `registration/affine.json` and `registration/qc_metrics.json`.
3. Emit explicit decision path logs in CLI output.
4. Keep `coords[:10]` behavior unchanged.

## Phase 4: Fallback Branch Scaffolding

1. Add landmark-fallback interfaces and artifact IO (`landmarks.json`) without GUI dependency.
2. Add deformable-fallback interfaces and artifact IO (`deform_field.npz`), initially as no-op stub with explicit `not_implemented` state.
3. Ensure gate results cleanly route to these branches.

## Phase 5: Deformable and Landmark Implementation

1. Implement semi-automatic landmark path (load existing landmark files, fit transform, re-QC).
2. Implement deformable refinement path (trigger only when local QC fails and global QC passes).
3. Integrate optional dense transform into multiplex patch sampling.
4. Save `final_transform.json` with provenance and selected branch.

## Phase 6: Integration Tests

1. Affine-pass integration test: deformable branch not called.
2. Global-fail integration test: landmarks branch selected.
3. Local-fail integration test: deformable branch selected.
4. End-to-end run on small TIFF pair with artifact checks.

## Phase 7: Documentation

1. Update README registration section with QC policy and branch behavior.
2. Add example commands for:
   - auto-only pass
   - QC-failure fallback path
   - reuse of saved transforms.

## Verification Checklist

1. `python3 -m pytest -q` passes.
2. `python3 -m stages.patchify ... --register` writes registration artifacts.
3. `python3 -m tools.debug_match_he_mul ...` shows improved patch-level overlap for accepted transform.
4. Deformable branch is only executed when QC gate says local fail.
