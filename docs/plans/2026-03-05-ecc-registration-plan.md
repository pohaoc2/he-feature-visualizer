# Plan: Affine Registration of H&E ↔ CyCIF via ECC on Tissue Masks

**Status: IMPLEMENTED** (2026-03-05)

## Context

The current pipeline maps H&E patch coordinates to CyCIF using a pure isotropic scale
(`scale = he_mpp / mx_mpp = 0.5`) with a shared origin assumption. This works well for
well-aligned pairs (WD-76845-096/097: 0.37%/3.57% extent error) but fails for poorly
co-registered ones (CRC02: 11.6% height mismatch), producing multiplex patches that
don't correspond to the correct tissue region.

The tissue boundary polygon (the blurry outline of tissue vs. glass) is visible in both
images — in H&E via HSV saturation, in CyCIF via the DNA/DAPI ch0 — and can be used to
compute a corrective affine transform before patching.

---

## Approach: ECC on Binary Tissue Masks

Both images at overview resolution (~1/64) produce binary tissue masks:
- **H&E mask**: already computed by `build_tissue_mask()` (HSV saturation → Otsu)
- **MX mask**: DNA channel (ch0 = DAPI, bright where nuclei exist = tissue) → Otsu threshold

Resize MX mask to H&E overview dimensions, then run `cv2.findTransformECC` with
`MOTION_AFFINE` to find the 2×3 warp matrix `M_ov` that aligns MX tissue to H&E tissue.
Compose `M_ov` with the overview↔full-res scaling to get `M_full` (H&E full-res → MX
full-res), which replaces the current `x0_mx = round(x0 * scale)` logic.

---

## Implementation

### Functions added to `patchify.py`

#### `build_mx_tissue_mask(store, axes, mx_h, mx_w, ds) -> np.ndarray`
Reads DNA/DAPI ch0 at overview resolution via `_read_overview_chw`, percentile-normalizes
(p1/p99) to uint8, Otsu-thresholds to binary. Returns bool ndarray `(mx_h // ds, mx_w // ds)`.

#### `register_he_mx_affine(he_mask, mx_mask, ds, he_h, he_w, mx_h, mx_w) -> np.ndarray`
Resizes MX mask to H&E overview dimensions, Gaussian-blurs both masks, runs
`cv2.findTransformECC(MOTION_AFFINE)` with 500-iteration / 1e-6 tolerance criteria.
Converts overview-space warp `M_ov` to full-resolution `M_full` via:

```
rx = he_ov_w / mx_ov_w
ry = he_ov_h / mx_ov_h

M_full = [[M_ov[0,0]/rx,  M_ov[0,1]/rx,  M_ov[0,2]*ds/rx],
           [M_ov[1,0]/ry,  M_ov[1,1]/ry,  M_ov[1,2]*ds/ry]]
```

Falls back to pixel-ratio scale (`[[1/scale,0,0],[0,1/scale,0]]`) if ECC raises `cv2.error`.

#### `transform_he_to_mx_point(M_full, x0, y0) -> tuple[int, int]`
Applies the 2×3 `M_full` to homogeneous point `[x0, y0, 1]` and returns rounded integers.

### Changes to `main()` in `patchify.py`

- Added `--register` / `--no-register` flag (`BooleanOptionalAction`, default: `True`)
- After `build_tissue_mask`: conditionally builds `mx_mask` and calls `register_he_mx_affine`
  to get `M_full`; when `--no-register`, falls back to `[[scale,0,0],[0,scale,0]]`
- Patch loop: replaced `round(x0 * scale)` / `round(y0 * scale)` with
  `transform_he_to_mx_point(M_full, x0, y0)`
- `index.json` gains `"warp_matrix"` (2×3 list) and `"registration"` (bool) fields

---

## Coordinate Transform Math

`M_ov` (2×3) from ECC operates in H&E overview space.
`M_full` converts it to H&E full-res → MX full-res space:

```
x_mx = M_full[0,0]*x0 + M_full[0,1]*y0 + M_full[0,2]
y_mx = M_full[1,0]*x0 + M_full[1,1]*y0 + M_full[1,2]
```

When no misalignment exists, M_ov ≈ identity → M_full ≈ `[[0.5,0,0],[0,0.5,0]]`
(equivalent to previous behaviour).

---

## Files Modified

| File | Change |
|------|--------|
| `patchify.py` | Added `build_mx_tissue_mask`, `register_he_mx_affine`, `transform_he_to_mx_point`; modified `main()` |

---

## Verification

```bash
# Run patchify with registration on CRC02 (known 11.6% height mismatch)
python patchify.py \
  --he-image data/CRC02-HE.ome.tif \
  --multiplex-image data/CRC02.ome.tif \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --out processed/ --patch-size 256 --stride 256 \
  --tissue-min 0.1 --channels CD31 "Antigen Ki67" CD45 PCNA \
  --overview-downsample 64 --vis-channels 0 10 20

# Check warp_matrix: diagonal should be close to [0.5, 0.5]
# with non-zero translation correcting the ~11.6% height offset
cat processed/index.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['warp_matrix'])"

# Visually verify with debug overlay tool
python debug_match_he_mul.py \
  --he-image data/CRC02-HE.ome.tif \
  --multiplex-image data/CRC02.ome.tif \
  --metadata-csv "data/CRC202105 HTAN channel metadata.csv" \
  --downsample 64

# Well-aligned pair: warp_matrix should be very close to [[0.5,0,0],[0,0.5,0]]
python patchify.py \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv "data/WD-76845-097-metadata.csv" \
  --out processed/ --channels CD31 "Antigen Ki67" CD45 PCNA
```
