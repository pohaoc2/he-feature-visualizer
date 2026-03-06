# H&E + Multiplex Feature Generator

A pipeline for generating spatially-resolved, multi-channel feature maps from co-registered H&E and multiplex immunofluorescence (mIF) whole-slide images.

## Goal

Given a pair of H&E and multiplex OME-TIFF images, this pipeline:

1. Extracts tissue patches and aligns multiplex channel arrays to each patch
2. Runs GPU-accelerated cell segmentation (CellViT) to detect and classify cells
3. Refines cell type assignments using multiplex marker intensities
4. Produces per-patch feature layers (cell type maps, vasculature, signaling channels) suitable for downstream spatial analysis or model training

## Pipeline Overview

```
Local machine                          Google Colab (GPU)
─────────────────────────────          ──────────────────────────────────
Stage 1: python patchify.py
  → processed/he/*.png
  → processed/multiplex/*.npy
  → processed/index.json
         │
         │ upload patches
         ▼
      [AWS S3]                    ──►   cellvit_colab_stage2.ipynb
                                          ↓
                                   processed/cellvit/*.json
         ◄─────────────────────────────────
         │ download results
         ▼
Stage 3: python assign_cells.py
Stage 4: python multiplex_layers.py
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
    ├── CRC02-HE.ome.tif
    ├── CRC02.ome.tif
    └── CRC202105 HTAN channel metadata.csv
```

---

## Stage 1 — Patch Extraction

Run `patchify.py` to extract 256×256 H&E patches and corresponding multiplex arrays. ECC affine registration between the two images is performed automatically (use `--no-register` to disable).

```bash
python patchify.py \
  --he-image data/CRC02-HE.ome.tif \
  --multiplex-image data/CRC02.ome.tif \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --out processed/ \
  --patch-size 256 \
  --stride 256 \
  --tissue-min 0.1 \
  --channels CD31 Ki67 CD45 PCNA \
  --overview-downsample 64 \
  --vis-channels 0 10 20
  # add --no-register to skip ECC and use mpp-ratio scaling only
```

### Registration

By default, Stage 1 computes an affine warp between the H&E and multiplex images using `cv2.findTransformECC` on binary tissue masks (H&E: HSV saturation → Otsu; MX: DNA/DAPI ch0 → Otsu) at ~1/64 overview resolution. This corrects translational and rotational misalignment without requiring manual landmarks.

The resulting 2×3 `warp_matrix` (H&E full-res → MX full-res) is stored in `index.json` and used to compute multiplex patch coordinates for every H&E patch.

### Outputs

| Output                      | Contents                                              | Used in         |
| --------------------------- | ----------------------------------------------------- | --------------- |
| `processed/he/`             | RGB PNG patches                                       | Stage 2 (Colab) |
| `processed/multiplex/`      | Per-channel `.npy` arrays (uint16)                    | Stage 4         |
| `processed/index.json`      | Patch coordinate index + `warp_matrix` + `registration` flag | All stages      |
| `processed/vis_patches.jpg` | H&E + multiplex overview with patch grid              | QC              |

Only `processed/he/` needs to be uploaded to Colab for Stage 2.

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
aws s3 sync processed/he/ s3://YOUR-BUCKET/he-feature-visualizer/processed/he/ --storage-class STANDARD_IA
```

Cost note: approximately 1000 patches × 60 KB = ~60 MB upload. The cost is negligible on STANDARD_IA storage class.

### Run the notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `notebooks/cellvit_colab_stage2.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
4. Edit the **Configuration cell** (Cell 1) at the top:


| Variable          | Description                          | Example                                |
| ----------------- | ------------------------------------ | -------------------------------------- |
| `STORAGE_BACKEND` | Storage backend                      | `'s3'`                                 |
| `S3_BUCKET`       | S3 bucket name                       | `'my-bucket'`                          |
| `S3_HE_PREFIX`    | S3 key prefix for patches            | `'he-feature-visualizer/processed/he'` |
| `MODEL_VARIANT`   | `'CellViT-256'` or `'CellViT-SAM-H'` | `'CellViT-256'`                        |
| `BATCH_SIZE`      | Patches per GPU batch                | `32` (reduce to 8 if OOM)              |
| `MAGNIFICATION`   | Scan magnification                   | `40` (use `20` if 20x slide)           |


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
aws s3 sync s3://YOUR-BUCKET/he-feature-visualizer/processed/cellvit/ processed/cellvit/
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

Assign cell types and states using CellViT detections combined with multiplex marker features:

```bash
python assign_cells.py \
  --cellvit-dir processed/cellvit/ \
  --features-csv data/CRC02.csv \
  --out processed/ \
  --index processed/index.json \
  --coord-scale 0.5
```

---

## Alignment QC

Use `debug_match_he_mul.py` to visually verify H&E↔multiplex registration. Without `--index-json` it does a naive resize; with it, it applies the ECC `warp_matrix` from `index.json`.

```bash
# Interactive viewer (requires display)
python debug_match_he_mul.py \
  --he-image data/CRC02-HE.ome.tif \
  --multiplex-image data/CRC02.ome.tif \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --downsample 64 \
  --index-json processed/index.json

# Headless — save static 3-panel PNG (no display needed)
python debug_match_he_mul.py \
  --he-image data/CRC02-HE.ome.tif \
  --multiplex-image data/CRC02.ome.tif \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --downsample 64 \
  --index-json processed/index.json \
  --save-png /tmp/vis_registered.png
```

The overlay panel (right) shows H&E blended with the selected multiplex channel. With good registration, tissue boundaries should coincide. The title bar indicates whether ECC registration was applied.

---

## Visualization

Generate a 6-panel summary figure for a single patch or a grid of random patches:

```bash
# Single patch (patch key is x0_y0 in pixel coordinates)
python visualize_pipeline.py \
  --processed processed/ \
  --patch 58624_4096 \
  --he-image data/CRC02-HE.ome.tif

# Random grid of N patches
python visualize_pipeline.py \
  --processed processed/ \
  --random 6 \
  --seed 42 \
  --he-image data/CRC02-HE.ome.tif \
  --mx-channel 0        # 0–3
```

Panels shown per patch: original location · H&E · multiplex channel · cell segmentation · cell type · cell state.

---

## Stage 4 — Multiplex Layers

Derive vasculature and oxygen/glucose layers from multiplex channels:

```bash
python multiplex_layers.py \
  --multiplex-dir processed/multiplex/ \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --out processed/
```

---

## Whole-Slide Viewer

For machines that can handle the full WSI, preprocess and serve the slide directly.

### Step 1 — Preprocess (run once, ~5–15 min)

```bash
python preprocess.py \
    --features data/crc02-features.csv \
    --image    data/CRC02-HE.ome.tif \
    --out      cache/
```

Outputs: `cache/features.parquet`, `cache/meta.json`, `cache/tiles/heatmap/…`

### Step 2 — Start the viewer

```bash
python server.py \
    --image data/CRC02-HE.ome.tif \
    --cache cache/ \
    --port  8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### Overlay modes

- **All Cells** — colored by inferred type (tumor / immune / stromal / other)
- **Vasculature** — CD31⁺ endothelial cells in red
- **Immune** — CD45⁺ cells colored by subtype (CD8a, CD68, FOXP3, CD4, CD20)

At low zoom a kernel-density heatmap is shown; zoom past level 3 for individual cell dots.

### Cell type assignment

Thresholds are the **95th percentile** of each marker across all cells.


| Mode        | Marker        | Color  |
| ----------- | ------------- | ------ |
| Tumor       | Keratin > p95 | Pink   |
| Immune      | CD45 > p95    | Green  |
| Stromal     | aSMA > p95    | Orange |
| Vasculature | CD31 > p95    | Red    |
| CD8a        | CD8a > p95    | Blue   |
| CD68        | CD68 > p95    | Green  |
| FOXP3       | FOXP3 > p95   | Purple |
| CD4         | CD4 > p95     | Yellow |
| CD20        | CD20 > p95    | Cyan   |


