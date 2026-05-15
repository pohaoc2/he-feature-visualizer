# CellViT Local Runner — Stage 2 Off-Colab

**Status:** IMPLEMENTED — 2026-05-12

Local equivalent of `notebooks/cellvit_colab_stage2.ipynb`. Run CellViT
inference on a local GPU instead of round-tripping H&E patches through S3 +
Colab.

## Artifacts

- `scripts/setup_cellvit_local.sh` — one-shot conda env + repo clone + checkpoint download.
- `stages/run_cellvit_local.py` — CLI: `--zip` patches → `--out` per-patch JSON + bundle zip.

## Quickstart

```bash
# 1. one-time setup (creates conda env "cellvit", clones repo, pulls checkpoint)
bash scripts/setup_cellvit_local.sh                       # default: CellViT-256
# or:
bash scripts/setup_cellvit_local.sh --model CellViT-SAM-H

# 2. inference
conda activate cellvit
python stages/run_cellvit_local.py \
    --zip processed/he.zip \
    --out processed/cellvit \
    --checkpoint ~/checkpoints/CellViT-256.pth \
    --cellvit-repo ~/CellViT \
    --batch-size 32

# 3. resume Stage 3 in the main project env
conda activate he-multiplex
python stages/assign_cells.py \
    --cellvit-dir processed/cellvit/ \
    --features-csv data/CRC02.csv \
    --out processed/index.json
```

## Output layout

```
processed/cellvit/
  {patch_id}.json   # {"patch": "...", "cells": [{centroid, contour, bbox, type_cellvit, type_name, type_prob}, ...]}
processed/cellvit.zip  # bundled equivalent — drop-in replacement for Colab S3 output
```

JSON schema matches the Colab notebook exactly, so Stage 3 (`assign_cells.py`)
consumes either output without changes.

## Caveats

### Environment isolation is mandatory
CellViT requires `pydantic==1.10.4`. The main project env (`he-multiplex`) and
most modern stacks expect pydantic 2.x. **Always use a dedicated env**
(`conda activate cellvit`) for inference, never the project env.

### Python version
CellViT repo + pydantic 1.10.4 require **Python 3.10**. Setup script pins this
explicitly. Python 3.13 (this project's default) will fail to install pydantic 1.x.

### CUDA wheels
Setup script installs `torch==2.1.2+cu121`. Verify with
`nvidia-smi` that your driver supports CUDA 12.1. For older drivers, edit the
`--index-url` in `scripts/setup_cellvit_local.sh` to `cu118` and pin a
compatible torch (e.g. `torch==2.1.2+cu118`).

### VRAM
- **CellViT-256, batch 32:** ~8 GB → fits comfortably on T4 (15 GB).
- **CellViT-SAM-H, batch 32:** OOM on T4. Drop to `--batch-size 4` or run on A10/A100.

### Checkpoint download via gdown
Google Drive throttles anonymous downloads. If gdown returns an HTML quota
page, either retry later or download manually in a browser and drop the file
at `~/checkpoints/CellViT-256.pth`.

### Patch zip layout
Script unzips into a temp dir and globs `**/*.png`. The Stage 1 output zip
(`processed/he.zip` produced by `patchify.py`) works as-is. Patch IDs are
derived from the PNG stem (`{i}_{j}.png` → `{i}_{j}.json`), matching the
notebook convention.

### Post-processing parity
Same cv2-contour post-processor as the notebook (HV-map watershed is *not*
used). Tunables at top of `run_cellvit_local.py`:
- `MIN_NUCLEUS_AREA=30`, `MAX_NUCLEUS_AREA=6000` (pixels², at 0.325 µm/px).
- `NUC_THRESHOLD=0.5` foreground probability cutoff.

### Resume support
`--skip-existing` (default on) skips patches whose JSON already exists in
`--out`. Use `--no-skip-existing` to force a full re-run.

### No S3 dependency
Unlike the notebook, no AWS credentials or `boto3` needed. If you still want
to ship the output zip to S3, use `aws s3 cp processed/cellvit.zip s3://...`
after the run.
