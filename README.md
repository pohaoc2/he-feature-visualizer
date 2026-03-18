# H&E + Multiplex Feature Generator

[Tests](https://github.com/pohaoc2/he-feature-visualizer/actions/workflows/test.yml)
[codecov](https://codecov.io/gh/pohaoc2/he-feature-visualizer)
[Python 3.13](https://www.python.org/downloads/release/python-3130/)
[Code style: black](https://github.com/psf/black)

A pipeline for generating spatially-resolved, multi-channel feature maps from co-registered H&E and multiplex immunofluorescence (mIF) whole-slide images.

## Goal

Given a pair of H&E and multiplex OME-TIFF images, this pipeline:

1. Extracts tissue patches and corresponding multiplex channel arrays
2. Runs GPU-accelerated cell segmentation (CellViT) to detect and classify cells
3. Refines cell type assignments using multiplex marker intensities
4. Produces per-patch feature layers (cell type maps, vasculature, signaling channels) suitable for downstream spatial analysis or model training

## Project Structure

```
stages/
  patchify.py            # Stage 1: extract 256×256 H&E patches + multiplex arrays (MPP-ratio coordinate mapping)
  extract_cell_features.py # Stage 2.5: extract CellViT-aligned MX marker features to CSV
  assign_cells.py        # Stage 3: assign cell types/states (CSV or auto CellViT+MX)
  multiplex_layers.py    # Stage 4: derive vasculature/signaling layers
utils/
  normalize.py           # percentile_norm, percentile_to_uint8
  channels.py            # load_channel_metadata, resolve_channel_indices
  ome.py                 # get_ome_mpp, open_zarr_store, read_overview_chw, get_image_dims
  group_visualizer.py    # Shared H&E + multiplex group rendering helpers
tools/
  build_pyramid.py       # Convert mask TIF to pyramidal OME-TIFF
  viz_mask.py            # Visualize mask + OME side-by-side (overview / crop)
  check_shape.py         # Inspect OME-TIFF dimensions (local or S3)
  visualize_pipeline.py  # 6-panel pipeline summary figure
  view_groups_web.py     # Local FastAPI web viewer (H&E + multiplex groups)
cellvit_backend.py       # CellViT model integration stub
notebooks/               # Stage 2: GPU cell segmentation on Colab
tests/                   # pytest test suite
```

## Pipeline Overview

```
Local machine                          Google Colab (GPU)
─────────────────────────────          ──────────────────────────────────
Stage 1: python -m stages.patchify
  → processed_wd/he/*.png
  → processed_wd/multiplex/*.npy
  → processed_wd/masks/*.npy   (optional, if --mask-image given)
  → processed_wd/index.json
         │
         │ upload patches (aws s3 sync)
         ▼
      [AWS S3]                    ──►   cellvit_colab_stage2.ipynb
                                          ↓
                                   processed_wd/cellvit/*.json
         ◄─────────────────────────────────
         │ download results
         ▼
Stage 3: python -m stages.assign_cells
Stage 4: python -m stages.multiplex_layers
```

---

## Setup

```bash
conda create -n he-multiplex python=3.13
conda activate he-multiplex
pip install -r requirements.txt
```

### Directory layout

```
lin-2021-crc-atlas/
└── data/
    ├── WD-76845-096.ome.tif
    ├── WD-76845-097.ome.tif
    ├── WD-76845-097-metadata.csv
    └── WD-76845-097.csv
```

---

## Stage 1 — Patch Extraction

Run `stages.patchify` to extract 256×256 H&E patches and corresponding multiplex arrays. Multiplex coordinates are mapped from H&E space using the MPP ratio (`he_mpp / mx_mpp`) — the two images are assumed to be pre-aligned.

```bash
python3 -m stages.patchify \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd \
  --channels DNA \
  --workers 8
  # add --mask-image path/to/cell-mask.ome.tif to extract mask patches (uint32 label IDs) to processed_wd/masks/
  # --workers N  parallelise patch extraction over N processes (default: 4)
```

Patch extraction is parallelised with `ProcessPoolExecutor`. Each worker opens its own zarr file handles, so the default of 4 workers is safe. On a machine with fast NVMe storage, `--workers 8` or higher is recommended.

### Outputs


| Output                         | Contents                                                                              | Used in         |
| ------------------------------ | ------------------------------------------------------------------------------------- | --------------- |
| `processed_wd/he/`             | RGB PNG patches                                                                       | Stage 2 (Colab) |
| `processed_wd/multiplex/`      | Per-channel `.npy` arrays (uint16)                                                    | Stage 4         |
| `processed_wd/masks/`          | Cell segmentation mask patches (uint32 label IDs); only if `--mask-image` given       | Downstream      |
| `processed_wd/index.json`      | Patch coordinate index + `mpp_scale` + flags per patch                                | All stages      |
| `processed_wd/vis_patches.jpg` | H&E + multiplex overview with patch grid                                              | QC              |


Only `processed_wd/he/` needs to be uploaded to Colab for Stage 2.

---

## Stage 2 — Cell Segmentation on Google Colab

[Open In Colab](https://colab.research.google.com/github/pohaoc2/he-feature-visualizer/blob/main/notebooks/cellvit_colab_stage2.ipynb)

Colab provides free T4 GPU access — sufficient for up to ~2000 patches.

### Prerequisites

- **AWS account** (for S3)
- **Local:** Python environment with the pipeline installed (`pip install -r requirements.txt`)
- **Colab:** T4 GPU runtime — the free tier is sufficient for up to approximately 2000 patches

### Upload patches

```bash
aws s3 sync processed_wd/he/ s3://YOUR-BUCKET/he-feature-visualizer/processed_wd/he/ --storage-class STANDARD_IA
```

Cost note: approximately 1000 patches × 60 KB = ~60 MB upload. The cost is negligible on STANDARD_IA storage class.

### Run the notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `notebooks/cellvit_colab_stage2.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
4. Edit the **Configuration cell** (Cell 1) at the top:


| Variable          | Description                          | Example                                   |
| ----------------- | ------------------------------------ | ----------------------------------------- |
| `STORAGE_BACKEND` | Storage backend                      | `'s3'`                                    |
| `S3_BUCKET`       | S3 bucket name                       | `'my-bucket'`                             |
| `S3_HE_PREFIX`    | S3 key prefix for patches            | `'he-feature-visualizer/processed_wd/he'` |
| `MODEL_VARIANT`   | `'CellViT-256'` or `'CellViT-SAM-H'` | `'CellViT-256'`                           |
| `BATCH_SIZE`      | Patches per GPU batch                | `32` (reduce to 8 if OOM)                 |
| `MAGNIFICATION`   | Scan magnification                   | `40` (use `20` if 20x slide)              |


1. **Runtime → Run all**

Expected timeline on a T4 GPU:


| Step                                             | Time                   |
| ------------------------------------------------ | ---------------------- |
| Install dependencies and clone CellViT repo      | ~2 min                 |
| Download model checkpoint (CellViT-256, ~400 MB) | ~1 min                 |
| Copy patches from S3 to Colab local SSD          | ~1 min per 500 patches |
| Inference (CellViT-256 at batch size 32)         | ~1 s per 32 patches    |
| Total for 1000 patches                           | ~20 min                |


### Download results

```bash
aws s3 sync s3://YOUR-BUCKET/he-feature-visualizer/processed_wd/cellvit/ processed_wd/cellvit/
```

### Resuming interrupted runs

The notebook sets `SKIP_EXISTING = True` by default. If the Colab session disconnects or you stop inference early, re-running from Cell 5 onwards will skip any patch that already has a corresponding JSON output. It is safe to interrupt and resume without reprocessing completed patches.

### Output schema (`cellvit/*.json`)

Each JSON file corresponds to one 256×256 patch, named by patch coordinate (e.g., `3_7.json`):

```json
{
  "patch": "3_7",
  "cells": [
    {
      "centroid": [128.4, 95.1],
      "contour":  [[120,88], [125,83], "..."],
      "bbox":     [[83,118], [103,139]],
      "type_cellvit": 2,
      "type_name": "Inflammatory",
      "type_prob": 0.87
    }
  ]
}
```

Cell type IDs used by CellViT:


| ID  | Type         |
| --- | ------------ |
| 1   | Neoplastic   |
| 2   | Inflammatory |
| 3   | Connective   |
| 4   | Dead         |
| 5   | Epithelial   |


---

## Stage 3 — Cell Type Assignment

Assign cell types and states using **CODEX-style clustering** plus CellViT priors.

Stage 3 label schema:
- Cell type: `cancer`, `immune`, `healthy`
- Cell state: `dead`, `proliferative`, `quiescent`

CellViT `type_cellvit=4` is a dead-state override.

Recommended Stage 3 workflow:
- use `--classifier codex`
- classify fine MX subtypes first (`epithelial`, `cd4_t`, `cd8_t`, `treg`, `b_cell`, `macrophage`, `endothelial`, `sma_stromal`)
- collapse those fine subtypes into final `cancer / immune / healthy`
- fuse the model probabilities with the CellViT prior

Current collapse map:
- `cancer <- epithelial`
- `immune <- cd4_t, cd8_t, treg, b_cell, macrophage`
- `healthy <- endothelial, sma_stromal`

### Mode A: Existing workflow (precomputed features CSV)

```bash
python -m stages.assign_cells \
  --cellvit-dir processed_crc33_crop/cellvit/ \
  --features-csv processed_crc33_crop/cellvit_mx_features.csv \
  --out processed_crc33_crop/ \
  --index processed_crc33_crop/index.json \
  --coord-scale 1.0 \
  --classifier codex \
  --csv-mpp 1.0 \
  --model-weight 0.5 \
  --workers 8
```

### Mode B: Auto-extract features from CellViT + multiplex patches

If `--features-csv` is omitted, Stage 3 can build one automatically from
CellViT contours and `processed_wd/multiplex/{x0}_{y0}.npy`.

```bash
python -m stages.assign_cells \
  --cellvit-dir processed_crc33_crop/cellvit/ \
  --multiplex-dir processed_crc33_crop/multiplex/ \
  --index processed_crc33_crop/index.json \
  --out processed_crc33_crop/ \
  --classifier codex \
  --workers 8
```

In Mode B, `--workers` is shared between the feature-extraction step (Stage 2.5) and the cell-assignment step.

This writes `processed_crc33_crop/cellvit_mx_features.csv` and then uses it for
assignment.

`--workers N` parallelises both the cell-assignment loop and (in Mode B) the upstream feature-extraction step using `ProcessPoolExecutor`. Default is 4; recommended value is `os.cpu_count() // 2` or the number of physical cores.

Stage 3 outputs now include:

- `cell_types/{x0}_{y0}.png`
- `cell_states/{x0}_{y0}.png`
- `cell_summary.json`
- `cell_assignments.csv` with one row per matched cell for downstream reporting

### Optional standalone extraction

```bash
python -m stages.extract_cell_features \
  --cellvit-dir processed_crc33_crop/cellvit/ \
  --multiplex-dir processed_crc33_crop/multiplex/ \
  --index processed_crc33_crop/index.json \
  --out-csv processed_crc33_crop/cellvit_mx_features.csv \
  --workers 8
```

---

## Stage 4 — Multiplex Layers

Derive vasculature, oxygen, and glucose proxy layers from multiplex channels.
`mx_mpp` is read automatically from `index.json` and used to convert all physical
distance clamps (µm) to pixels.

### Proxy models

| Layer | Channel(s) | Model | Method | Reference |
|---|---|---|---|---|
| Vasculature | CD31 (+ SMA optional) | — | Otsu threshold → binary mask | — |
| Oxygen | CD31 | `distance` (default) | Euclidean distance transform clamped at **160 µm** | Grimes 2014, Zaidi 2019 |
| Oxygen | CD31, Ki67, CD68 | `wsi-pde` | Steady-state `D∇²u − k(x)u + s(x) = 0` solved **WSI-wide** at coarse resolution; per-patch results cropped from the global field | Kumar 2024 |
| Glucose | CD31 | `distance` | Same distance transform clamped at **450 µm** (wider supply zone) | Grimes 2014 |
| Glucose | Ki67 | `max` (default) | `percentile_norm(Ki67)` | — |
| Glucose | CD31, Ki67, CD68 | `wsi-pde` | Same WSI-scale PDE solver as oxygen | Kumar 2024 |

**Why WSI-scale?**  The Krogh oxygen diffusion radius (~160 µm) is larger than a single 256 px patch (~83 µm at 0.325 µm/px).
Per-patch computation cannot capture nutrients supplied by vessels in adjacent patches.
The WSI-scale solver reads CD31/Ki67 directly from the OME-TIFF pyramid at a coarse downsampling factor (`--wsi-pde-ds`, default 8), computes a Euclidean distance transform once on the full slide, and extracts per-patch slices in milliseconds.

**Algorithm** (O(N), runs in seconds even for 30 M-pixel WSIs):
1. Build vessel mask from CD31 at coarse resolution
2. Compute `dist = distance_transform_edt(~vessel_mask)` — exact for constant k
3. Compute spatially varying decay length `L(x) = krogh_um / (mpp_coarse × √(k(x)/k_base))` where k(x) = base + Ki67 + CD68 demand terms
4. `u(x) = exp(−dist(x) / L(x))` — exact PDE solution for constant k; WKB approximation for varying k

Only CD31, Ki67, and CD68 channels are loaded — memory ≤ 400 MB for a 30 M-pixel WSI at ds=8.

### Outputs

| Path | Contents |
|---|---|
| `{out}/vasculature/{x0}_{y0}.png` | RGBA red vessel overlay |
| `{out}/vasculature_mask/{x0}_{y0}.npy` | bool vessel mask for downstream |
| `{out}/oxygen/{x0}_{y0}.png` | RdYlBu oxygenation map (blue=near vessel) |
| `{out}/glucose/{x0}_{y0}.png` | Hot glucose-availability map |
| `{out}/validation/ki67_vs_distance.csv` | Ki67 mean vs distance-from-vessel bins (optional, `--validate-ki67-distance`) |

`--workers N` parallelises the per-patch loop (numpy/cv2 operations, file I/O) using `ProcessPoolExecutor`. The WSI-scale PDE pre-computation (when `--oxygen-model wsi-pde` or `--glucose-model wsi-pde`) runs once serially before the parallel loop and is unaffected by `--workers`.

### Distance model run (default, Zaidi 2019 / Grimes 2014)

Uses physically clamped distance transforms with separate Krogh radii for O₂ and glucose.
MPP is read from `index.json` automatically; `--oxygen-krogh-um` and `--glucose-krogh-um`
control both the distance clamp and (when using `wsi-pde`) the D calibration.

```bash
python -m stages.multiplex_layers \
  --multiplex-dir processed_crc33/multiplex/ \
  --index processed_crc33/index.json \
  --metadata-csv data/markers.csv \
  --out processed_crc33/ \
  --oxygen-model distance \
  --oxygen-krogh-um 160 \
  --glucose-model distance \
  --glucose-krogh-um 450 \
  --workers 8
```

### WSI-scale PDE run (Kumar 2024, physically correct)

Solves `D∇²u − k(x)u + s(x) = 0` once on the full WSI at coarse resolution
(`--wsi-pde-ds 4` → 1.3 µm/px coarse grid), then crops per-patch results.
Requires the raw multiplex OME-TIFF via `--multiplex-tiff`.

```bash
# Full WSI (recommended: ds=8 keeps memory ~400 MB and converges in <30k iters)
python -m stages.multiplex_layers \
  --multiplex-dir processed_crc33/multiplex/ \
  --index processed_crc33/index.json \
  --metadata-csv data/markers.csv \
  --out processed_crc33/ \
  --multiplex-tiff data/mx_crc33.ome.tiff \
  --oxygen-model wsi-pde \
  --oxygen-krogh-um 160 \
  --glucose-model wsi-pde \
  --glucose-krogh-um 450 \
  --wsi-pde-ds 8 \
  --wsi-pde-max-iters 40000 \
  --oxygen-consumption-base 0.1 \
  --oxygen-consumption-demand-weight 0.3 \
  --glucose-consumption-base 0.1 \
  --glucose-consumption-demand-weight 0.3 \
  --cd68-consumption-weight 0.1 \
  --workers 8
```

Key `wsi-pde` parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--multiplex-tiff` | required | Path to the multiplex OME-TIFF |
| `--wsi-pde-ds` | `4` | Downsampling factor. Use **8** for full WSIs to keep memory ≤ 500 MB and converge in ≤ 40k iters |
| `--wsi-pde-max-iters` | `20000` | Jacobi iterations needed ≈ 4 × L_coarse² where L_coarse = krogh_um / (mpp × ds) |
| `--wsi-pde-tol` | `1e-4` | Convergence tolerance |
| `--oxygen-krogh-um` | `160` | Krogh radius → auto-calibrates D = (krogh_um / mpp_coarse)² × k_base |
| `--glucose-krogh-um` | `450` | Same for glucose |

Memory footprint at ds=8 (only CD31 + Ki67 + CD68 are loaded — all other channels are skipped):

| WSI size | Coarse grid | Memory (3 ch) |
|---|---|---|
| 1 024 × 1 024 crop | 128 × 128 | < 1 MB |
| 52 740 × 36 354 full | 6 593 × 4 545 | ~360 MB |

The PDE equation structure and additive k(x) form follow Kumar et al. (2024).
Scalar weights (`base`, `w_ki67`, `w_cd68`) are heuristic defaults; Kumar 2024
fits them data-driven from CA9 hypoxia staining (absent from this panel).

### Ki67-vs-distance validation (Zaidi 2019)

Accumulates per-pixel Ki67 intensity binned by distance from CD31 across all patches.
Expect Ki67 to peak at ~50–100 µm from vessels and drop beyond ~150 µm (hypoxic
quiescence), validating that the distance proxy correlates with proliferative biology.

```bash
python -m stages.multiplex_layers \
  --multiplex-dir processed_crc33/multiplex/ \
  --index processed_crc33/index.json \
  --metadata-csv data/markers.csv \
  --out processed_crc33/ \
  --oxygen-model distance \
  --oxygen-krogh-um 160 \
  --glucose-model max \
  --validate-ki67-distance \
  --validate-bin-um 10 \
  --workers 8
```

Output: `{out}/validation/ki67_vs_distance.csv` with columns `distance_um`, `ki67_mean`, `pixel_count`.

### Optional vessel-mask refinement and cleanup

```bash
python -m stages.multiplex_layers \
  --multiplex-dir processed_crc33/multiplex/ \
  --index processed_crc33/index.json \
  --metadata-csv data/markers.csv \
  --out processed_crc33/ \
  --vasc-sma-refine \
  --sma-adjacency-px 2 \
  --vasc-open-kernel-size 3 \
  --vasc-close-kernel-size 3 \
  --vasc-min-area 16 \
  --vasc-noisy-max-fraction 0.98 \
  --workers 8
```

SMA refinement adds αSMA⁺ pixels adjacent to CD31 mask (pericyte-covered vessels
are more likely to be functionally perfused). The noisy-fallback threshold discards
masks with >98% coverage as artefactual.

### Suggested presets

| Use case | Oxygen model | Glucose model | Notes |
|---|---|---|---|
| Quick exploratory | `distance` | `max` | Fast; no OME-TIFF needed at runtime |
| Physically calibrated | `distance` | `distance` | Both Krogh-clamped; cross-patch comparable |
| Full mechanistic (WSI) | `wsi-pde` | `wsi-pde` | Requires `--multiplex-tiff`; captures cross-patch gradients |

### Interpretation caveats

- **Distance maps** use a physically grounded clamp (160 µm O₂, 450 µm glucose per Grimes 2014),
  making values cross-patch comparable. Pixels beyond the clamp are all set to fully depleted.
- **WSI-PDE maps** are normalized relative nutrient-density proxies (vessel pixel = 1, depleted = 0),
  not absolute physiological concentrations. Normalization is done per-WSI-solve.
- Glucose `wsi-pde` at small crops (< 400 µm) will yield nearly flat output — glucose reaches
  everywhere within the Krogh radius. Use the full WSI OME-TIFF for meaningful glucose gradients.
- All 2D models systematically underestimate oxygenation near out-of-plane vessels (Grimes 2016).
- Output quality is sensitive to CD31 channel quality.

---

## Visualization

Generate a 6-panel summary figure for a single patch or a grid of random patches:

```bash
# Single patch (patch key is x0_y0 in pixel coordinates)
python -m tools.visualize_pipeline \
  --processed processed_wd/ \
  --patch 58624_4096 \
  --he-image data/WD-76845-096.ome.tif

# Random grid of N patches
python -m tools.visualize_pipeline \
  --processed processed_wd/ \
  --random 6 \
  --seed 42 \
  --he-image data/WD-76845-096.ome.tif \
  --mx-channel 0
```

Panels shown per patch: original location · H&E · multiplex channel · cell segmentation · cell type · cell state.

---

### Scientific-Vis Figure: Random Patch Grid

Use `tools.scientific_vis_patch_grid` to create a publication-style N-row × 5-column
grid sampled randomly from available patches. Requires Stage 3 output.

```bash
python -m tools.scientific_vis_patch_grid \
  --processed processed_crc33_crop/ \
  --random 3 \
  --seed 42 \
  --assignments-csv processed_crc33_crop/cell_assignments.csv \
  --out-prefix processed_crc33_crop/patch_grid \
  --formats png
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--random N` | required | Number of patches to sample |
| `--seed` | `None` | Random seed for reproducibility |
| `--hoechst` | `DNA` | Hoechst/DNA marker name in `index.json` |
| `--vasc-cd31` | `CD31` | CD31 marker name in `index.json` |

Column layout (one row per patch):

| Col | Contents |
|-----|----------|
| C1 | H&E |
| C2 | Hoechst/DNA channel (Blues colormap) |
| C3 | CellViT segmentation mask — contours filled by 5-class type (neoplastic / inflammatory / connective / dead / epithelial) |
| C4 | Final fused type overlay — contours matched to Stage 3 assignments (cancer / immune / healthy) |
| C5 | Cell state overlay (proliferative / quiescent / dead) |
| C6 | Vasculature RGBA overlay (CD31 mask composited on H&E) |
| C7 | Oxygen proxy (RdYlBu: blue = near vessel / oxygenated, red = hypoxic) + colorbar |
| C8 | Glucose proxy (hot colormap: bright = high supply, dark = depleted) + colorbar |

Patches are discovered automatically from `he/`, `cellvit/`, and `cell_assignments.csv`.
If `--random N` exceeds the number of valid patches, it clamps silently.
Hoechst and CD31 degrade to a "not in panel" placeholder if absent from `index.json`.

---

### Scientific-Vis Figure: CellViT vs Model Patch Comparison

For a publication-style patch comparison figure (2×4 grid), run Stage 3 and render
with `tools.scientific_vis_cellvit_mx`.

```bash
# 1) Stage 3: auto-extract CellViT+MX features and assign type/state
python -m stages.assign_cells \
  --cellvit-dir processed_crc33_crop/cellvit/ \
  --multiplex-dir processed_crc33_crop/multiplex/ \
  --features-csv processed_crc33_crop/cellvit_mx_features.csv \
  --index processed_crc33_crop/index.json \
  --out processed_crc33_crop/ \
  --classifier codex \
  --coord-scale 1.0 \
  --csv-mpp 1.0 \
  --model-weight 0.5

# 2) Render one patch (auto-selects evidence marker; CD31/SMA vasculature column added automatically)
python -m tools.scientific_vis_cellvit_mx \
  --processed processed_crc33_crop/ \
  --patch 768_0 \
  --assignments-csv processed_crc33_crop/cell_assignments.csv \
  --out-prefix processed_crc33_crop/patch_768_0_cd45 \
  --formats png

# 3) Override the main marker and/or vasculature marker names
python -m tools.scientific_vis_cellvit_mx \
  --processed processed_crc33_crop_demo/ \
  --patch 256_256 \
  --assignments-csv processed_crc33_crop_demo/cell_assignments.csv \
  --mx-marker Ki67 \
  --vasc-cd31 CD31 \
  --vasc-sma aSMA \
  --out-prefix processed_crc33_crop_demo/scivis_patch_256_256_ki67 \
  --formats png,pdf
```

Output figure panels (2×4, figsize 11.2×5.6 in):

| Panel | Contents |
|-------|----------|
| A | H&E patch |
| B | Selected MX marker channel (auto-selected or `--mx-marker`) |
| C | Cell state overlay (`proliferative` / `quiescent` / `dead`) with color legend |
| D | CD31 channel in `Reds` colormap — vasculature proxy (`--vasc-cd31`, default `CD31`) |
| E | Model type overlay (`CODEX fine type` when `--classifier codex`) with color legend |
| F | Final fused type overlay with color legend |
| G | CellViT mapped 3-class overlay (`cancer` / `immune` / `healthy`) with color legend |
| H | Vasculature composite: **R=CD31, G=SMA**, yellow=co-localized larger vessels (`--vasc-sma`, default `SMA`) |

Panels D and H degrade gracefully: if CD31 or SMA is absent from the multiplex panel,
a "not in panel" placeholder renders and the figure still saves normally.

**Color palettes** are designed to be non-overlapping:
- Cell types: red (cancer), blue (immune), green (healthy)
- Cell states: magenta (proliferative), amber (quiescent), purple (dead)

### Scientific-Vis Figure: Sample-Level Model Evidence Report

Use `tools.scientific_vis_model_report` to generate a sample-level report with:

- mapped CellViT vs model vs final class counts
- CellViT/model mismatch heatmap
- model probability distributions over `cancer / immune / healthy`
- model subtype distributions
- fine-to-final collapse summary used by the model report
- representative example cells for each final class

```bash
python -m tools.scientific_vis_model_report \
  --processed processed_crc33_crop_demo/ \
  --assignments-csv processed_crc33_crop_demo/cell_assignments.csv \
  --summary-json processed_crc33_crop_demo/cell_summary.json \
  --out-prefix processed_crc33_crop_demo/model_report \
  --formats png,pdf
```

If `cell_summary.json` reports `classifier_used=rule_fallback`, the report is
labeled accordingly and its probability panels should be interpreted as
rule-based outputs.

---

### Scientific-Vis Figure: CellViT vs CODEX Comparison

`tools.scientific_vis_codex_comparison` — 16×16 publication figure comparing
CellViT (morphology-based) and CODEX (multiplex marker-based) cell type assignments.

**Panel layout:**

| Panel | Content |
|-------|---------|
| A | 3×3 confusion matrix (CellViT rows × CODEX cols), raw counts + row-normalised % |
| B | Cancer triptych: Agree / Medium / Disagree examples (H&E crop + marker bar) |
| C | Immune triptych |
| D | Healthy triptych |

**CLI:**

```bash
python -m tools.scientific_vis_codex_comparison \
  --processed processed_crc33_crop/ \
  --assignments-csv processed_crc33_crop/cell_assignments.csv \
  --out-prefix processed_crc33_crop/codex_comparison \
  --formats png \
  --dpi 300
```

| Flag | Default | Description |
|------|---------|-------------|
| `--processed` | required | Processed directory (for `he/` patches) |
| `--assignments-csv` | `<processed>/cell_assignments.csv` | Stage 3 output |
| `--out-prefix` | `<processed>/codex_comparison` | Output path prefix |
| `--formats` | `pdf,png` | Comma-separated output formats |
| `--dpi` | `300` | Raster DPI |
| `--cancer-marker` | `Pan-CK` | Canonical cancer marker column name |
| `--immune-marker` | `CD45` | Canonical immune marker column name |
| `--healthy-marker` | `SMA` | Canonical healthy marker column name |

**Marker bars** are percentile-normalized once across all cells in the CSV,
so bars are comparable across examples within the figure.
Cells with `cell_type == "other"` are excluded.
If a canonical marker column is absent, its bar renders as "n/a".

---

### Overlay modes

- **All Cells** — colored by inferred type (`cancer` / `immune` / `healthy`)
- **Vasculature** — CD31⁺ endothelial cells in red
- **Immune** — CD45⁺ cells colored by subtype (CD8a, CD68, FOXP3, CD4, CD20)

At low zoom a kernel-density heatmap is shown; zoom past level 3 for individual cell dots.

### Cell type assignment

Stage 3 is documented around the **CODEX-style clustering** workflow:

- Type labels: `cancer`, `immune`, `healthy`
- State labels: `dead`, `proliferative`, `quiescent`
- Fine model subtypes: `epithelial`, `cd4_t`, `cd8_t`, `treg`, `b_cell`, `macrophage`, `endothelial`, `sma_stromal`
- Final collapse: immune subtypes → `immune`, endothelial/stromal → `healthy`, epithelial → `cancer`
- Fusion: `P_final = w * P_model + (1-w) * P_cellvit_prior` where `w` is adaptive per CellViT type:
  - `type=1` (Neoplastic): `w=0.30` — H&E morphology strongly trusted; cancer cells stay cancer
  - `type=2` (Inflammatory): `w=0.70` — CODEX trusted for immune subtyping
  - `type=3` (Connective): `w=0.40` — H&E trusted; stromal cells stay healthy
  - `type=0/4/5` (unknown/dead/epithelial): `w=0.50` — equal weight
  - `--model-weight` overrides the default fallback (0.85) for types not in the table
- Normalization: winsorize per marker to [1st, 99th pct] → per-marker Z-score (Bai et al. 2021, *Front. Immunol.*)
- Dead-state override: `type_cellvit == 4 -> dead`


---

## Group Viewer (Web)

Use these tools for interactive H&E + multiplex group inspection.

### Web viewer (`tools.view_groups_web`)

```bash
python -m tools.view_groups_web \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --index-json processed_wd/index.json \
  --min-level 1 \
  --auto-max-dim 1200 \
  --host 127.0.0.1 \
  --port 8010
```

Open **[http://127.0.0.1:8010](http://127.0.0.1:8010)**.

Remote (EC2) usage is the same, then tunnel:

```bash
ssh -i /path/to/key.pem -L 8010:127.0.0.1:8010 ec2-user@<EC2_PUBLIC_DNS_OR_IP>
```

Viewer controls:

- Group buttons: `H&E`, `Immune`, `Vasculature`, `Cancer Cells`, `Proliferative Cells`
- Transparency slider for multiplex overlay
- Pan: two-finger touchpad scroll or click-drag
- Zoom: pinch or `Ctrl/Alt + wheel`
- Left and right panes are synchronized during pan/zoom
- Color legend is drawn on the figure (right panel)

Rendering behavior:

- Defaults to `min-level >= 1` to avoid full-resolution level 0 loading.
- Multi-marker groups use per-channel min-max normalization (no percentile), then multicolor max projection.

Notes:

- If `--index-json` is omitted, the viewer tries `processed_wd/index.json` then `proceeded_wd/index.json`.
- Use `--no-preload-multiplex` if memory is limited.

---

## Tests

All tests use synthetic data — no real images or data files are required.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=stages --cov=utils --cov-report=term-missing

```

Coverage is reported automatically on every CI run via [Codecov](https://codecov.io/gh/pohaoc2/he-feature-visualizer).

## Code Style & Linting

The project uses [Black](https://black.readthedocs.io/) for formatting and [Pylint](https://pylint.readthedocs.io/) for linting. Both run in CI on every push and pull request.

```bash
# Format check (CI runs this)
black --check stages/ utils/ tools/ tests/

# Format code in place
black stages/ utils/ tools/ tests/

# Lint (CI runs errors-only)
pylint stages/ utils/ tools/ --errors-only

# Full lint report
pylint stages/ utils/ tools/
```

Black enforces a consistent style (88‑char line length, no manual formatting). Pylint catches logical errors, unused imports, and common bugs; CI runs with `--errors-only` so only E/F-level issues fail the build.
