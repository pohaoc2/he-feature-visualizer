# CRC02 TME Viewer

Interactive H&E viewer with tumor microenvironment overlays for the Lin 2021 CRC Atlas.

## Setup

```bash
pip install -r requirements.txt
```

## Directory layout expected

```
lin-2021-crc-atlas/
└── data/
    ├── CRC02-HE.ome.tif
    └── crc02-features.csv        # or .xlsx (full ~600k rows)
```

Place all four files (`preprocess.py`, `server.py`, `viewer.html`, `requirements.txt`)
in the same directory, e.g. `lin-2021-crc-atlas/`.

## Step 1 — Preprocess (run once, takes ~5–15 min)

```bash
python preprocess.py \
    --features data/crc02-features.csv \
    --image    data/CRC02-HE.ome.tif \
    --out      cache/
```

If your features file is XLSX:
```bash
python preprocess.py \
    --features data/crc02-features.xlsx \
    --image    data/CRC02-HE.ome.tif \
    --out      cache/
```

This will generate:
- `cache/features.parquet`         — indexed cell data with assigned types
- `cache/meta.json`                — image dimensions + marker thresholds
- `cache/tiles/heatmap/{mode}/…`   — pre-rendered PNG heatmap tiles

## Step 2 — Start the viewer server

```bash
python server.py \
    --image data/CRC02-HE.ome.tif \
    --cache cache/ \
    --port  8000
```

Then open **http://127.0.0.1:8000** in your browser.

## Usage

- **Toggle buttons** at the top switch between three overlay modes:
  - **All Cells** — colored by inferred type (tumor / immune / stromal / other)
  - **Vasculature** — CD31⁺ endothelial cells in red
  - **Immune** — CD45⁺ cells colored by subtype (CD8a, CD68, FOXP3, CD4, CD20)

- At **low zoom**: a smooth kernel-density heatmap is shown.
- **Zoom in past level 3**: switches to individual cell dots fetched live from the server.

## Cell type assignment logic

Thresholds are the **95th percentile** of each marker across all cells.

| Mode | Marker | Color |
|---|---|---|
| Cells → Tumor | Keratin > p95 | Pink |
| Cells → Immune | CD45 > p95 | Green |
| Cells → Stromal | aSMA > p95 | Orange |
| Vasculature | CD31 > p95 | Red |
| Immune → CD8a | CD8a > p95 | Blue |
| Immune → CD68 | CD68 > p95 | Green |
| Immune → FOXP3 | FOXP3 > p95 | Purple |
| Immune → CD4 | CD4 > p95 | Yellow |
| Immune → CD20 | CD20 > p95 | Cyan |

---

## Patch-based viewer (lightweight)

If your machine cannot render the full whole-slide image, you can pre-crop into 256×256 patches and view one at a time.

### Step 1 — Generate patches (run once)

```bash
python patchify.py \
    --image data/CRC02-HE.ome.tif \
    --features-csv data/CRC02.csv \
    --out processed/ \
    --stride 256 \
    --features cell_mask vasculature immune
```

- **`--stride`**: step between patch top-lefts (256 = no overlap; 128 = 50% overlap).
- **`--features`**: which layers to write (`cell_mask`, `vasculature`, `immune`; any subset).
- **`--tissue-min`**: drop patches with tissue fraction below this (default 0.1).
- **`--cache-meta`**: optional path to `cache/meta.json` to reuse image size and thresholds.

Output under `processed/`:
- `he/{i}_{j}.png` — H&E crops
- `overlay_cells/{i}_{j}.png` — cell-type overlay (if any feature requested)
- `cell_mask/`, `vasculature/`, `immune/` — isolated feature rasters (if requested)
- `index.json` — list of kept patches

### Step 2 — Start the patch viewer

```bash
python server_patches.py --processed processed/ --port 8000
```

Open **http://127.0.0.1:8000**. Use the sidebar to pick a patch; toggle Overlay and feature layers (cell mask, vasculature, immune). Prev/Next or the index input jump between patches.
