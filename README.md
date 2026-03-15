# H&E + Multiplex Feature Generator

[Tests](https://github.com/pohaoc2/he-feature-visualizer/actions/workflows/test.yml)
[codecov](https://codecov.io/gh/pohaoc2/he-feature-visualizer)
[Python 3.13](https://www.python.org/downloads/release/python-3130/)
[Code style: black](https://github.com/psf/black)

A pipeline for generating spatially-resolved, multi-channel feature maps from co-registered H&E and multiplex immunofluorescence (mIF) whole-slide images.

## Goal

Given a pair of H&E and multiplex OME-TIFF images, this pipeline:

1. Extracts tissue patches and aligns multiplex channel arrays to each patch
2. Runs GPU-accelerated cell segmentation (CellViT) to detect and classify cells
3. Refines cell type assignments using multiplex marker intensities
4. Produces per-patch feature layers (cell type maps, vasculature, signaling channels) suitable for downstream spatial analysis or model training

## Project Structure

```
stages/
  patchify.py            # Stage 1: extract 256×256 H&E patches + multiplex arrays + ECC registration
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
  debug_match_he_mul.py  # Interactive/headless H&E↔multiplex alignment viewer (QC)
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

Run `stages.patchify` to extract 256×256 H&E patches and corresponding multiplex arrays. ECC affine registration between the two images is performed automatically (use `--no-register` to disable).

```bash
python3 -m stages.patchify \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd \
  --channels DNA \
  # --register
  # add --no-register to skip ECC and use mpp-ratio scaling only
  # add --mask-image path/to/cell-mask.ome.tif to extract mask patches (uint32 label IDs) to processed_wd/masks/
```

### Registration

By default, Stage 1 computes an affine warp between the H&E and multiplex images using `cv2.findTransformECC` on binary tissue masks (H&E: HSV saturation → Otsu; MX: DNA/DAPI ch0 with blur+morphology) at overview resolution. This corrects translational and rotational misalignment without requiring manual landmarks.

Stage 1 then runs QC gates (channel drift, global IoU/centroid/scale, patch-level gain). If QC returns `FAIL_LOCAL_NEEDS_DEFORMABLE`, a deformable refinement is attempted and only applied when it improves QC.

The resulting 2×3 `warp_matrix` (H&E full-res → MX full-res) is stored in `index.json`. During patch extraction, each multiplex patch is sampled with this transform (affine or deformable-refined) into the H&E patch frame (not only top-left scale mapping), improving patch-level correspondence when shear/rotation offsets exist.

### Outputs


| Output                         | Contents                                                                              | Used in         |
| ------------------------------ | ------------------------------------------------------------------------------------- | --------------- |
| `processed_wd/he/`             | RGB PNG patches                                                                       | Stage 2 (Colab) |
| `processed_wd/multiplex/`      | Per-channel `.npy` arrays (uint16)                                                    | Stage 4         |
| `processed_wd/masks/`          | Cell segmentation mask patches (uint32 label IDs); only if `--mask-image` given       | Downstream      |
| `processed_wd/index.json`      | Patch coordinate index + `warp_matrix` + `registration_mode` + flags per patch        | All stages      |
| `processed_wd/registration/`   | `affine.json`, `qc_metrics.json`, `final_transform.json`, optional `deform_field.npz` | QC / debug      |
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
  --csv-mpp 1.0
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
  --classifier codex
```

This writes `processed_crc33_crop/cellvit_mx_features.csv` and then uses it for
assignment.

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
  --out-csv processed_crc33_crop/cellvit_mx_features.csv
```

---

## Stage 4 — Multiplex Layers

Derive vasculature and oxygen/glucose layers from multiplex channels.

Outputs per patch:

- `processed_wd/vasculature/{x0}_{y0}.png` (RGBA vessel overlay)
- `processed_wd/vasculature_mask/{x0}_{y0}.npy` (bool vessel mask)
- `processed_wd/oxygen/{x0}_{y0}.png` (oxygen proxy)
- `processed_wd/glucose/{x0}_{y0}.png` (glucose proxy)f

### Legacy-compatible run (default models)

```bash
python -m stages.multiplex_layers \
  --multiplex-dir processed_wd/multiplex/ \
  --index processed_wd/index.json \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd/ \
  --oxygen-model distance \
  --glucose-model max
```

### PDE proxy run (oxygen + glucose)

```bash
python -m stages.multiplex_layers \
  --multiplex-dir processed_wd/multiplex/ \
  --index processed_wd/index.json \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd/ \
  --oxygen-model pde \
  --glucose-model pde \
  --pde-max-iters 500 \
  --pde-tol 1e-4 \
  --pde-diffusion 1.0 \
  --oxygen-consumption-base 0.1 \
  --oxygen-consumption-demand-weight 0.3 \
  --glucose-consumption-base 0.1 \
  --glucose-consumption-demand-weight 0.3
```

### Optional vessel-mask refinement and cleanup

```bash
python -m stages.multiplex_layers \
  --multiplex-dir processed_wd/multiplex/ \
  --index processed_wd/index.json \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd/ \
  --vasc-sma-refine \
  --sma-adjacency-px 2 \
  --vasc-open-kernel-size 3 \
  --vasc-close-kernel-size 3 \
  --vasc-min-area 16 \
  --vasc-noisy-max-fraction 0.98
```

### Suggested presets

- Exploratory analysis: use PDE models and SMA-assisted vessel cleanup.
- Strict reproducibility: set all model and cleanup flags explicitly in the command and keep them fixed across runs.

### Interpretation caveats

- Oxygen and glucose PDE maps are relative density proxies, not absolute physiological concentrations.
- Output quality is sensitive to channel quality (especially CD31/aSMA/Ki67/PCNA) and upstream registration/segmentation quality.

---

## Alignment QC

Use `tools.debug_match_he_mul` to visually verify H&E↔multiplex registration. Without `--index-json` it does a naive resize; with it, it applies the registration transform from `index.json` (ECC affine, plus deformable field when available and active).

```bash
# Interactive viewer (requires display)
python -m tools.debug_match_he_mul \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --downsample 64 \
  --index-json processed_wd/index.json

# Headless — save static 3-panel PNG (no display needed)
python -m tools.debug_match_he_mul \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --index-json processed_wd/index.json \
  --save-png processed_wd/vis_registered_check.png
```

The overlay panel (right) shows H&E blended with the selected multiplex channel. With good registration, tissue boundaries should coincide. The title bar indicates whether affine or deformable registration was applied.

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
| C5 | CD31 channel (Reds colormap) — vasculature proxy |

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
  --csv-mpp 1.0

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
  --formats png,pdf \
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
- Fusion: `P_final = 0.85 * P_model + 0.15 * P_cellvit_prior`
- Dead-state override: `type_cellvit == 4 -> dead`


---

## Group Viewer (Web)

Use these tools for interactive H&E + multiplex group inspection after registration.

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

- Uses registration matrix (`warp_matrix`) from `index.json` to transform multiplex into H&E space.
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

# Run only registration / mapping tests
pytest tests/test_patchify_registration.py -v
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
