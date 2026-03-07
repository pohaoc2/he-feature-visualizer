# Plan: PDE-Based Oxygen/Glucose Proxies and Binary Vasculature Mask

**Date:** 2026-03-06
**Status:** READY
**Scope:** `stages/multiplex_layers.py`, `tests/test_multiplex_layers.py`, `README.md`

## Goal

Add an explicit binary vasculature mask product and optional PDE-based oxygen/glucose density proxies while keeping current behavior available for backward compatibility.

## Why This Plan

1. Current oxygen map is distance-based from CD31 and ignores consumption heterogeneity.
2. Current glucose proxy (`max(Ki67, PCNA)`) is useful but not transport-aware.
3. A PDE model gives a more biologically grounded diffusion-consumption approximation, while still producing relative proxy maps.

## Non-Goals

1. No claim of absolute oxygen partial pressure or glucose concentration without external calibration.
2. No full fluid dynamics / perfusion model.
3. No breaking change to existing default outputs unless explicitly enabled.

## Phase 1: Interface and Backward-Compatible Wiring

1. Add model-selection flags in `stages.multiplex_layers`:
  - `--oxygen-model {distance,pde}` (default: `distance`)
  - `--glucose-model {max,pde}` (default: `max`)
2. Add PDE parameter flags with safe defaults:
  - `--pde-max-iters`, `--pde-tol`, `--pde-diffusion`
  - `--oxygen-consumption-base`, `--oxygen-consumption-demand-weight`
  - `--glucose-consumption-base`, `--glucose-consumption-demand-weight`
3. Keep existing output folders and file names (`oxygen/*.png`, `glucose/*.png`) so downstream viewer code remains compatible.

## Phase 2: Vasculature Binary Mask Product

1. Produce and save explicit binary mask per patch:
  - `processed/vasculature_mask/{x0}_{y0}.npy` (bool)
  - optional visualization PNG in existing `vasculature/` overlay folder.
2. Keep CD31 as primary vessel marker.
3. Add optional SMA-assisted refinement mode:
  - use `aSMA` only when spatially adjacent to CD31 support (avoid `CD31 OR aSMA` global union).
  - morphology cleanup (open/close, min area) to reduce speckle.
4. Handle edge cases explicitly:
  - empty/noisy vessel mask emits warning and deterministic fallback.

## Phase 3: PDE Core Solver

1. Implement a steady-state 2D diffusion-consumption solver for normalized fields:
  - equation: `D * Laplacian(u) - k(x) * u + s(x) = 0`
2. Numerical method:
  - finite-difference stencil + iterative relaxation (Jacobi/Gauss-Seidel), bounded to `[0, 1]`.
  - deterministic convergence criteria: max update < `tol` or `max_iters` reached.
3. Boundary conditions:
  - default zero-flux (Neumann) at patch boundary.
  - keep as internal implementation detail for now (no CLI until needed).
4. Add small helper functions for clarity:
  - `build_vessel_source_map(...)`
  - `build_consumption_map(...)`
  - `solve_steady_state_diffusion(...)`

## Phase 4: Oxygen and Glucose PDE Proxies

1. Oxygen PDE (`--oxygen-model pde`):
  - `s_o2(x)` from vessel mask (and optional vessel confidence weighting).
  - `k_o2(x)` from baseline + proliferative demand (`Ki67/PCNA` normalized map).
2. Glucose PDE (`--glucose-model pde`):
  - `s_glu(x)` from vessel mask.
  - `k_glu(x)` from baseline + demand map; retain current max-fusion demand signal as the first consumption driver.
3. Preserve current non-PDE modes:
  - oxygen distance-transform mode unchanged.
  - glucose max-fusion mode unchanged.
4. Label all PDE outputs as relative density proxies in logs/docs.

## Phase 5: Testing Strategy

1. Unit tests for solver behavior:
  - convergence on synthetic source/consumption fields.
  - bounded output in `[0, 1]`.
  - expected radial decay from a central source in homogeneous media.
2. Unit tests for vessel mask generation:
  - CD31-only baseline mask behavior.
  - SMA-adjacent refinement does not flood-fill stromal regions.
3. Integration test updates for CLI:
  - run with default flags (legacy behavior).
  - run with `--oxygen-model pde --glucose-model pde` and verify output artifacts exist.
4. Regression tests for edge cases:
  - no vessel pixels.
  - uniform channels.
  - missing optional SMA channel when SMA refinement is disabled/enabled.

## Phase 6: Documentation and Operational Guidance

1. Update Stage 4 section in `README.md` with new flags and examples.
2. Document biological interpretation caveats:
  - relative proxy, not absolute concentration.
  - sensitivity to channel quality and segmentation quality.
3. Add recommended default presets for exploratory analysis vs strict reproducibility.

## Success Criteria

1. Stage 4 produces explicit binary vessel masks for every processed patch.
2. PDE oxygen/glucose modes run end-to-end on synthetic and real patches.
3. Existing commands (without new flags) preserve current outputs.
4. Test suite passes for both legacy and PDE modes.

## Verification Checklist

1. `pytest tests/test_multiplex_layers.py -v`
2. `pytest tests/ -v` (full regression)
3. Smoke test (legacy):
  - `python3 -m stages.multiplex_layers ... --oxygen-model distance --glucose-model max`
4. Smoke test (PDE):
  - `python3 -m stages.multiplex_layers ... --oxygen-model pde --glucose-model pde`
5. Confirm outputs exist:
  - `processed/vasculature_mask/*.npy`
  - `processed/vasculature/*.png`
  - `processed/oxygen/*.png`
  - `processed/glucose/*.png`

## Risks and Mitigations

1. **Risk:** PDE runtime overhead per patch.
  - **Mitigation:** coarse-grid solve + upsample option; cap iterations; vectorized updates.
2. **Risk:** Overcalling vessels when SMA is used naively.
  - **Mitigation:** require CD31 adjacency gate for SMA-assisted pixels.
3. **Risk:** Users interpret maps as absolute physiology.
  - **Mitigation:** explicit “relative proxy” labeling in CLI logs and README.

