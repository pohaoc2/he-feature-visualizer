# patchify.py Rewrite Design

Date: 2026-03-05

## Context

`patchify.py` extracts paired 256×256 patches from an H&E OME-TIFF and a
multiplex OME-TIFF for downstream cell segmentation and marker quantification.

### Problems with the current implementation

1. **Per-patch tissue detection** — runs HSV+Otsu independently on every patch
   candidate. Slow for large images (thousands of S3 reads) and lacks whole-slide
   context.
2. **No visualization** — no way to confirm patch coverage or H&E↔multiplex
   alignment visually.
3. **Tissue detection on wrong data** — per-patch normalization can produce
   inconsistent results across patches.

### Dataset facts (CRC02)

| | H&E | Multiplex |
|---|---|---|
| File | CRC02-HE.ome.tif (17 GiB) | CRC02.ome.tif (93 GiB) |
| Format | BigTIFF, planar CYX | BigTIFF, planar CYX |
| Dims | 82,230 × 54,363 px | 41,094 × 24,017 px |
| mpp | 0.325 µm/px | 0.65 µm/px |
| Channels | 3 (RGB) | 36 |
| Physical size | 26,725 × 17,668 µm | 26,711 × 15,611 µm |

Scale factor H&E → multiplex: `0.325 / 0.65 = 0.5` (exact 2× in both axes).
The Y physical extent differs by ~13% — multiplex FOV is smaller. Patches
outside the multiplex FOV are kept for H&E but marked `has_multiplex: false`.

---

## Design

### Approach chosen: overview-mask tissue detection

Download a ~1/64 resolution overview of the H&E via zarr stride sampling
(~3 MB), run HSV+Otsu on the overview to get a whole-slide tissue mask, then
filter patch candidates by mask fraction. This gives CLAM-quality tissue
detection without contour-tracking complexity.

### Overall flow

```
patchify.py
│
├── 1. open_ome_tiff(path)
│       tifffile + zarr → (store, axes, img_w, img_h, mpp_x, mpp_y)
│
├── 2. build_tissue_mask(he_store, axes, img_w, img_h, downsample=64)
│       zarr stride read → (H/64, W/64, 3) uint8 overview
│       HSV + Otsu + morphological close → bool mask
│
├── 3. get_tissue_patches(mask, img_w, img_h, patch_size, stride, tissue_min, downsample)
│       grid over level-0 coords
│       per patch: fraction = mask[py0:py1, px0:px1].mean() >= tissue_min
│       returns list of (x0, y0)
│
├── 4. extract_patches(he_store, mx_store, coords, patch_size, scale, channels)
│       H&E: read (x0, y0, patch_size, patch_size) → uint8 RGB PNG
│       Multiplex: x0_mx = round(x0*scale), size_mx = round(patch_size*scale)
│                  read (x0_mx, y0_mx, size_mx, size_mx) → resize to patch_size
│                  save as (C, patch_size, patch_size) uint16 .npy
│       skip multiplex if patch falls outside mx FOV → has_multiplex=false
│
└── 5. visualize(he_store, mx_store, coords, scale, vis_channels)
        H&E thumbnail (overview, reused) + multiplex composite thumbnail
        draw patch grid on both panels (green=has_mx, yellow=no_mx)
        side-by-side → vis_patches.jpg
```

### Multiplex alignment

```python
scale   = he_mpp / mx_mpp          # 0.5
x0_mx   = round(x0 * scale)
y0_mx   = round(y0 * scale)
size_mx = round(patch_size * scale) # 128 at 0.5×
# resize each channel: (C, 128, 128) → (C, 256, 256)
```

### Visualization layout

```
┌─────────────────┬─────────────────┐
│   H&E overview  │  Multiplex RGB  │
│  (82K×54K /64)  │  (41K×24K /64)  │
│  green = has mx │  same grid,     │
│  yellow = no mx │  scaled by 0.5  │
└─────────────────┴─────────────────┘
         vis_patches.jpg
```

Multiplex composite: 3 user-selected channels (default: 0, 10, 20) each
scaled to uint8 via p1/p99 percentile normalization, composited as RGB.

### index.json schema (additive change)

```json
{
  "patches": [
    {"x0": 512, "y0": 1024, "has_multiplex": true},
    {"x0": 512, "y0": 49000, "has_multiplex": false}
  ],
  "patch_size": 256,
  "stride": 256,
  "tissue_min": 0.1,
  "img_w": 82230,
  "img_h": 54363,
  "he_mpp": 0.325,
  "mx_mpp": 0.65,
  "scale_he_to_mx": 0.5,
  "channels": ["CD31", "Ki67", "CD45", "PCNA"]
}
```

### CLI (unchanged from current)

```bash
python patchify.py \
    --he-image data/CRC02-HE.ome.tif \
    --multiplex-image data/CRC02.ome.tif \
    --metadata-csv data/channel_metadata.csv \
    --out processed/ \
    --patch-size 256 \
    --stride 256 \
    --tissue-min 0.1 \
    --channels CD31 Ki67 CD45 PCNA \
    --overview-downsample 64 \
    --vis-channels 0 10 20
```

### Functions to keep unchanged

- `tissue_mask_hsv` — already correct
- `read_multiplex_patch` — already correct
- `load_channel_indices` — already correct
- `get_ome_mpp` — already correct

### Functions to rewrite

- `read_he_patch` — minor: ensure CYX→YXC axis handling is robust
- `main` — replace per-patch Otsu loop with overview-mask approach
- **New**: `build_tissue_mask(store, axes, img_w, img_h, downsample)`
- **New**: `get_tissue_patches(mask, img_w, img_h, patch_size, stride, tissue_min, downsample)`
- **New**: `visualize(he_store, he_axes, mx_store, mx_axes, coords, scale, vis_channels, out_dir)`
