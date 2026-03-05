# Running Stage 2 (CellViT) on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pohaoc2/he-feature-visualizer/blob/main/notebooks/cellvit_colab_stage2.ipynb)

This guide explains how to run the CellViT cell segmentation stage of the
he-feature-visualizer pipeline on Google Colab, which provides free GPU access
when a local GPU is not available.

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
  [Google Drive]  OR  [AWS S3]   ──►   cellvit_colab_stage2.ipynb
                                          ↓
                                   processed/cellvit/*.json
         ◄─────────────────────────────────
         │ download results
         ▼
Stage 3: python assign_cells.py
Stage 4+5: python multiplex_layers.py
```

---

## Prerequisites

- **Google account** (for Colab + Drive) OR **AWS account** (for S3)
- **Local:** Python environment with the pipeline installed (`pip install -r requirements.txt`)
- **Colab:** T4 GPU runtime — the free tier is sufficient for up to approximately 2000 patches

---

## Step 1 — Run Stage 1 Locally

Run `patchify.py` to extract 256×256 H&E patches and corresponding multiplex
arrays from the raw OME-TIFF files:

```bash
python patchify.py \
  --he-image data/CRC02-HE.ome.tif \
  --multiplex-image data/CRC02.ome.tif \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --out processed/ \
  --patch-size 256 \
  --stride 256 \
  --tissue-min 0.1 \
  --channels CD31 Ki67 CD45 PCNA
```

This produces three output directories under `processed/`:

| Directory | Contents | Used in |
|---|---|---|
| `processed/he/` | RGB PNG patches | Stage 2 (Colab) |
| `processed/multiplex/` | Per-channel `.npy` arrays | Stage 4+5 |
| `processed/index.json` | Patch coordinate index | All stages |

Only `processed/he/` needs to be uploaded to Colab for Stage 2. The multiplex
outputs stay local and are used later.

---

## Step 2 — Upload Patches

Choose one of the two storage backends.

### Option A: Google Drive

Upload via the web UI (drag and drop) or with `rclone` for large patch sets:

```bash
# Web UI: drag processed/he/ into MyDrive/he-feature-visualizer/processed/he/

# OR with rclone (faster for large patch sets):
rclone copy processed/he/ gdrive:he-feature-visualizer/processed/he/ --progress
```

### Option B: AWS S3

```bash
aws s3 sync processed/he/ s3://YOUR-BUCKET/he-feature-visualizer/processed/he/ --storage-class STANDARD_IA
```

Cost note: approximately 1000 patches x 60 KB = ~60 MB upload. The cost is
negligible on STANDARD_IA storage class.

---

## Step 3 — Open the Colab Notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** and select `notebooks/cellvit_colab_stage2.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 4 — Configure the Notebook

Edit the **Configuration cell** (Cell 1) at the top of the notebook. The
relevant variables are:

| Variable | Description | Example |
|---|---|---|
| `STORAGE_BACKEND` | `'drive'` or `'s3'` | `'drive'` |
| `DRIVE_HE_DIR` | Drive path to patches | `'he-feature-visualizer/processed/he'` |
| `S3_BUCKET` | S3 bucket name | `'my-bucket'` |
| `S3_HE_PREFIX` | S3 key prefix for patches | `'he-feature-visualizer/processed/he'` |
| `MODEL_VARIANT` | `'CellViT-256'` or `'CellViT-SAM-H'` | `'CellViT-256'` |
| `BATCH_SIZE` | Patches per GPU batch | `32` (reduce to 8 if OOM) |
| `MAGNIFICATION` | Scan magnification | `40` (use `20` if 20x slide) |

---

## Step 5 — Run All Cells

Click **Runtime → Run all**. Expected timeline on a T4 GPU:

| Step | Time |
|---|---|
| Install dependencies and clone CellViT repo | ~2 min |
| Download model checkpoint (CellViT-256, ~400 MB) | ~1 min |
| Copy patches from Drive/S3 to Colab local SSD | ~1 min per 500 patches |
| Inference (CellViT-256 at batch size 32) | ~1 s per 32 patches |
| Total for 1000 patches | ~20 min |

---

## Step 6 — Download Results

### Option A: Google Drive

Results are auto-saved to Drive during inference. Copy them back locally:

```bash
rclone copy gdrive:he-feature-visualizer/processed/cellvit/ processed/cellvit/ --progress
```

### Option B: AWS S3

```bash
aws s3 sync s3://YOUR-BUCKET/he-feature-visualizer/processed/cellvit/ processed/cellvit/
```

---

## Step 7 — Continue the Pipeline Locally

Once `processed/cellvit/` is populated, run the remaining stages:

```bash
# Stage 3: assign cell types and states using CellViT detections + multiplex features
python assign_cells.py \
  --cellvit-dir processed/cellvit/ \
  --features-csv data/CRC02.csv \
  --out processed/

# Stage 4+5: vasculature + oxygen/glucose layers from multiplex channels
python multiplex_layers.py \
  --multiplex-dir processed/multiplex/ \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --out processed/
```

---

## Output Schema — What `cellvit/*.json` Contains

Each JSON file corresponds to one 256x256 patch, named by patch coordinate
(e.g., `3_7.json`):

```json
{
  "patch": "3_7",
  "cells": [
    {
      "centroid": [128.4, 95.1],
      "contour":  [[120,88], [125,83], ...],
      "bbox":     [[83,118], [103,139]],
      "type_cellvit": 2,
      "type_name": "Inflammatory",
      "type_prob": 0.87
    }
  ]
}
```

Cell type IDs used by CellViT:

| ID | Type |
|---|---|
| 1 | Neoplastic |
| 2 | Inflammatory |
| 3 | Connective |
| 4 | Dead |
| 5 | Epithelial |

---

## Resuming Interrupted Runs

The notebook sets `SKIP_EXISTING = True` by default. If the Colab session
disconnects or you stop inference early, re-running from Cell 5 onwards will
skip any patch that already has a corresponding JSON output in the output
directory. It is safe to interrupt and resume without reprocessing completed
patches.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No GPU detected` | Runtime → Change runtime type → T4 GPU |
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 8 or 4 |
| `ModuleNotFoundError: pydantic` | Re-run the install cell; restart the runtime |
| `KeyError: model_state_dict` | Replace `checkpoint['model_state_dict']` with `checkpoint` in the model load cell |
| `AssertionError: H&E patch directory not found` | Check `DRIVE_HE_DIR` / `S3_HE_PREFIX` in the config cell |
| Colab session disconnects | Re-run from Cell 5 onwards; `SKIP_EXISTING=True` resumes from where it stopped |
| S3 `NoCredentialsError` | Re-enter AWS credentials in the S3 configuration cell |

---

## Notes on Magnification

CellViT-256 was trained on the PanNuke dataset at **40x magnification**
(0.25 µm/px). CRC02 is a cyclic immunofluorescence slide — verify the pixel
size in the OME-TIFF metadata before choosing a model variant:

```bash
python -c "import tifffile; t=tifffile.TiffFile('data/CRC02-HE.ome.tif'); print(t.ome_metadata)"
```

- If the pixel size is approximately **0.25 µm/px (40x)**: use
  `MODEL_VARIANT='CellViT-256'` with `MAGNIFICATION=40`
- If the pixel size is approximately **0.5 µm/px (20x)**: use
  `MODEL_VARIANT='CellViT-256'` with `MAGNIFICATION=20`, or switch to
  `CellViT-SAM-H` which handles lower magnifications more robustly
