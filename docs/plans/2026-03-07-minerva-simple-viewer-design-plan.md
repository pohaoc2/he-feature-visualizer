# Design Plan: Simple Minerva-Style Viewer (Current State)

**Date:** 2026-03-07  
**Status:** ACTIVE  
**Primary files:** `server_minerva.py`, `viewer_minerva.html`, `utils/minerva_groups.py`, `tests/test_server_minerva.py`, `tests/test_minerva_groups.py`

## Context

We need a minimal interactive viewer for paired H&E + multiplex-derived cell overlays that behaves like a simplified Minerva:

1. Smooth zoom and pan on large WSIs.
2. Fast group switching (`H&E`, `Immune`, `Tissue`, `Cancer`, `Proliferative`, `Vasculature`).
3. Correct alignment between multiplex cell coordinates and H&E image space.

The current blocking issues were:

1. Slow/unstable loading when large TIFFs were used naively.
2. Group switches occasionally felt delayed.
3. H&E vs multiplex spatial offset needed explicit transform usage.

## Goals

1. Keep the viewer minimal and reliable (no annotation arrows, no advanced UI).
2. Apply `processed_wd/index.json` registration (`warp_matrix`) before rendering overlays.
3. Avoid loading TIFF full-resolution level 0 by default.
4. Maintain responsive group toggling while zooming/panning.

## Non-Goals

1. Full Minerva feature parity.
2. Server-side dynamic heatmaps or clustering overlays.
3. Advanced persistence/session state.

## Current Architecture

### Backend (`server_minerva.py`)

1. Load features and compute marker-threshold-based group flags once at startup.
2. Read TIFF pyramid and preload a single render level RGB array into memory.
3. Default level policy:
   - skip level 0 when multi-level pyramid exists (`--min-render-level 1`);
   - auto-select first level with max dimension <= `--max-render-dim`;
   - fallback to coarsest level if needed.
4. Load and invert `warp_matrix` (HE->MX) to MX->HE affine and apply to coordinates.
5. Scale transformed coordinates into selected render-level space.
6. Precompute per-group point arrays for fast `/cells` viewport queries.
7. Serve DZI tiles + lightweight `/meta` + `/cells` API.

### Frontend (`viewer_minerva.html`)

1. OpenSeadragon drives pan/zoom tile viewing.
2. Group buttons switch active overlay type.
3. Overlay points drawn on a canvas (screen-space circles).
4. Viewport-aware `/cells` calls with:
   - request debouncing;
   - quantized in-memory cache;
   - prefetch of non-active groups for current viewport.
5. Dynamic `max_cells` based on zoom to control draw and response cost.

## Data/Transform Contract

1. Features source: `data/WD-76845-097.csv` (`Xt/Yt` preferred; fallback `X/Y`).
2. Registration source: `processed_wd/index.json`.
3. Required key: `warp_matrix` shape `(2, 3)` representing HE->MX.
4. Runtime uses inverse affine (MX->HE) before viewport filtering and drawing.

## Performance Strategy

1. Never use full-res level 0 by default on pyramidal TIFFs.
2. Preload one practical level once; avoid repeated random disk reads.
3. Serve coarse DZI levels from thumbnail fallback when possible.
4. Precompute `group_points` to make `/cells` selection vectorized and predictable.
5. Deterministic downsampling when `total_in_view > max_cells`.

## Verification Plan

1. Automated tests:
   - `tests/test_server_minerva.py`
   - `tests/test_minerva_groups.py`
2. Key assertions:
   - transform inversion correctness;
   - group filtering behavior;
   - DZI low-res fallback path;
   - render-level chooser skips level 0 by default.
3. Manual smoke checks:
   - pan/zoom remains responsive;
   - clicking each group updates overlay quickly;
   - apparent offset is corrected versus H&E landmarks.

## Operational Runbook

Example:

```bash
python server_minerva.py \
  --image data/WD-76845-096.ome.tif \
  --features data/WD-76845-097.csv \
  --index-json processed_wd/index.json \
  --min-render-level 1 \
  --max-render-dim 7000 \
  --port 8010
```

## Next Decisions

1. Finalize if groups remain marker-threshold heuristics or move to curated cell-type labels.
2. Decide default `--max-render-dim` for EC2 instance sizes.
3. Decide whether to add optional startup pre-warm for all groups in initial viewport.
