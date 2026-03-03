# Patch-based H&E + feature viewer — design

**Date:** 2025-03-03  
**Status:** Validated

## Purpose

Enable viewing H&E and cell/vasculature/immune overlays on hardware that cannot render the full whole-slide image by:

1. Cropping the H&E image into 256×256 patches (with configurable stride; drop mostly-empty patches).
2. Building spatial feature rasters from the feature CSV for the same patch regions.
3. Saving all patches under a `processed/` tree.
4. Visualizing by loading one patch at a time (H&E + optional overlay and/or isolated feature PNGs).

## Constraints

- Reuse existing cell-type and marker logic (thresholds, colors) from `preprocess.py` where possible.
- Do not load the full OME-TIFF into RAM; read 256×256 windows (e.g. via tifffile/zarr).
- User chooses which feature layers to generate (cell_mask, vasculature, immune).

## Data and coordinates

- **Image:** Full-resolution H&E (`CRC02-HE.ome.tif`). Dimensions from OME-TIFF or `cache/meta.json` (`img_w`, `img_h`).
- **Cells:** Feature CSV (e.g. `data/CRC02.csv`) with columns `Xt`, `Yt` for image-pixel coordinates (same space as the OME-TIFF). Use existing thresholds from `cache/meta.json` if present.
- **Patch grid:** 0-based patch index `(i, j)`. Top-left pixel of patch `(i,j)`:
  - `x0 = j * stride`, `y0 = i * stride`.
  - Box: `[x0, x0+256) x [y0, y0+256)` (full-res pixels). Stride is configurable (e.g. 256 = no overlap, 128 = 50% overlap).

## Empty-patch filter

- For each 256×256 H&E crop: convert to grayscale, compute fraction of pixels below a background threshold (tissue).
- If tissue fraction &lt; `tissue_min` (e.g. 0.1): skip the patch (write no files for that `(i,j)`).

## Output layout under `processed/`

| Path | When written | Content |
|------|----------------|--------|
| `he/{i}_{j}.png` | Always (if not empty) | 256×256 RGB H&E crop |
| `overlay_cells/{i}_{j}.png` | If any feature requested | Transparent overlay: all cell-type dots (for blending on H&E) |
| `cell_mask/{i}_{j}.png` | If `cell_mask` in `--features` | Cells colored by type (tumor/immune/stromal/other), isolated |
| `vasculature/{i}_{j}.png` | If `vasculature` in `--features` | CD31⁺ cells only, isolated |
| `immune/{i}_{j}.png` | If `immune` in `--features` | CD45⁺ cells colored by subtype (CD8a, CD68, FOXP3, CD4, CD20, other), single raster |

- **Index:** `processed/index.json` (or `.parquet`) listing kept patches: `(i, j, x0, y0, x1, y1)` and optionally stride, tissue_min, for the viewer.

## Preprocessing script (CLI)

- **Positional / required:** `--image`, `--features-csv`, `--out` (e.g. `processed/`).
- **Patch:** `--stride` (default 256).
- **Features:** `--features` space-separated list: any subset of `cell_mask`, `vasculature`, `immune` (e.g. `--features cell_mask vasculature immune`).
- **Empty detection:** `--tissue-min` (default 0.1), `--background-threshold` (optional; else derive from histogram/Otsu).
- **Optional:** `--cache-meta` path to reuse `cache/meta.json` for image size and thresholds.

**Algorithm (per patch):**

1. Read 256×256 H&E window at `(x0, y0)`.
2. If tissue fraction &lt; `tissue_min`, skip.
3. Write `he/{i}_{j}.png`.
4. Filter CSV rows with `Xt`, `Yt` in `[x0, x0+256) x [y0, y0+256)`; map to local 0–255 in the 256×256 patch.
5. For each requested feature, draw dots (same colors as current pipeline) into a 256×256 RGBA PNG; save under `{feature}/{i}_{j}.png`.
6. If any feature requested, write `overlay_cells/{i}_{j}.png` (e.g. composite of cell types or reuse one layer).
7. Append `(i, j, x0, y0, x0+256, y0+256)` to the index for kept patches.

## Viewer

- Minimal UI that loads one patch at a time: list or grid of patch IDs from the index; on selection, load `he/{i}_{j}.png` and optionally `overlay_cells/{i}_{j}.png` and/or `cell_mask/`, `vasculature/`, `immune/` for that `(i,j)`.
- No need to hold the full slide in memory.

## Out of scope (YAGNI)

- Per–immune-subtype isolated folders (single `immune/` raster only).
- DZI / multi-resolution for patches (flat 256×256 only).
- Changes to existing whole-slide viewer or cache layout.
