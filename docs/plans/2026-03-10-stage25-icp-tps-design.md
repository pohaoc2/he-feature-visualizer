# Stage 2.5 Redesign: ICP + TPS Registration in H&E Space

**Date:** 2026-03-10
**Status:** IMPLEMENTED

---

## Motivation

The original Stage 2.5 matched CellViT centroids (H&E px) to CSV cell centroids (MX px) using the
existing `m_full` affine as an initial guess, then fitted a thin-plate-spline (TPS) directly from
H&E px → MX px. Two problems motivated the redesign:

1. **Matching in MX space is opaque.** Errors in `m_full` corrupt the matching step in a way that
   is hard to inspect or quantify. Working entirely in H&E space makes each step visually
   verifiable with the same coordinate frame as the H&E image.

2. **No iterative refinement.** A single mutual-NN pass with the existing affine leaves systematic
   drift uncorrected. An ICP step before TPS fitting corrects global affine drift first, so TPS
   only needs to model local non-rigid residuals.

---

## Coordinate Spaces

| Space | Units | Notes |
|-------|-------|-------|
| H&E px | pixels | Crop image, origin = top-left of crop |
| MX px | pixels | Crop image, origin = top-left of crop |
| CSV µm | micrometres | Full-slide coordinates; divide by `csv_mpp` to get full-slide MX px |
| Full-slide MX px | pixels | Subtract `crop_origin` to get crop MX px |

The `m_full` (2×3 affine) maps **H&E crop px → MX crop px**.
Its inverse maps **MX crop px → H&E crop px**, used to bring CSV centroids into H&E space.

**When using cropped images:** `--mx-crop-origin OX OY` must be provided so that full-slide MX px
coords from the CSV are shifted into crop MX px space before `inv(m_full)` is applied.

---

## Pipeline

```
Stage 1  →  crop/he/*.png + crop/multiplex/*.npy + crop/index.json
Stage 2  →  crop/cellvit/<x0>_<y0>.json   (CellViT on Colab GPU)
Stage 2.5 →  crop/multiplex/*.npy (updated) + crop/index.json (updated)
```

---

## Algorithm

### Step 1 — Load CellViT centroids (unchanged)
Read all `crop/cellvit/<x0>_<y0>.json` files.  Local centroid coords are offset by `(x0, y0)` to
produce global H&E crop pixel coordinates.

### Step 2 — CSV → MX px → H&E px  *(new)*
```
CSV Xt, Yt  (µm, full-slide)
  ÷ csv_mpp                    → full-slide MX px
  − crop_origin                → crop MX px
  × inv(m_full)                → H&E crop px   (csv_in_he)
```
Function: `csv_to_he_coords(csv_path, m_full, csv_mpp, crop_origin)`

### Step 3 — Affine ICP in H&E space  *(new)*
Iteratively align the CellViT cloud toward the `csv_in_he` cloud:

1. Nearest-neighbour match `src_curr → csv_in_he` (KDTree, optional `distance_gate`).
2. Fit affine via `cv2.estimateAffine2D(RANSAC)` on matched pairs.
3. Apply step transform to `src_curr`; accumulate into `M_icp`.
4. Converge when `‖M_step[:,:2] − I‖_F + ‖M_step[:,2]‖ < tol`.

Function: `affine_icp(src_he, dst_he, max_iter=50, tol=1e-4, distance_gate)`
Returns `M_icp` (2×3), `n_matches`, `n_iters`.

ICP role: correct **global affine drift** in `m_full` (translation, rotation, scale).

### Step 4 — Post-ICP matching + RANSAC
```
icp_he = M_icp @ CellViT_he     (ICP-corrected cloud, in H&E space)
mutual NN:  icp_he  ↔  csv_in_he   (distance_gate)
RANSAC affine filter (cv2, thresh=5 H&E px)
```
Function: `match_centroids_he(icp_he, he_pts_he, csv_in_he, mx_pts, distance_gate)`
Returns: `src_he` = original CellViT H&E px, `dst_mx` = matched CSV crop MX px.

### Step 5 — Fit TPS (H&E px → MX px)
```
src_tps = original CellViT H&E coords   [RANSAC inliers]
dst_tps = matched CSV crop MX px        [RANSAC inliers]
TPS = scipy.RBFInterpolator(thin_plate_spline, degree=1)
```
Two interpolators: `tps_x` and `tps_y` (one per output dimension).
Spatially uniform subsampling to `max_tps_points` before fitting.

TPS role: correct **local non-rigid distortions** after global drift is removed by ICP.

### Step 6 — Re-extract multiplex patches (unchanged)
For each H&E patch, create a pixel grid, evaluate `tps_x / tps_y` to get MX coords, then sample
the MX TIFF with `cv2.remap` (bilinear). Overwrites `crop/multiplex/<x0>_<y0>.npy`.

### Step 7 — Update index.json
New fields added:
```json
{
  "registration_mode": "icp_tps",
  "icp_matrix":        [[...], [...]],
  "icp_n_iters":       int,
  "icp_n_matches":     int,
  "tps_n_matches":     int,
  "tps_inlier_fraction": float,
  "tps_control_he":    [[x,y], ...],
  "tps_control_mx":    [[x,y], ...]
}
```

---

## Diagnostic Visualization (new tool)

`tools/viz_registration_debug.py` — run **after Stage 2, before Stage 2.5**:

| Panel | Content | Coordinates |
|-------|---------|-------------|
| a | H&E overview (ds=4) + patch grid boxes | overview px (axes in H&E px via `extent`) |
| b | CellViT centroids overlaid on H&E | H&E px |
| c | CSV-in-HE (blue) vs ICP-aligned CellViT (orange), zoomed to patch ROI | H&E px axes |
| d | CellViT (green) vs CSV-in-HE (red) overlap, same ROI | H&E px axes |

Panels c/d zoom to the bounding box of Stage 1 patches (± half patch_size margin) and display H&E
px on both axes. Points outside the ROI are filtered before scatter to avoid crowding.

CLI:
```bash
python -m tools.viz_registration_debug \
  --processed crop/ \
  --he-image data/WD-76845-096-crop.ome.tif \
  --csv data/WD-76845-097.csv \
  --csv-mpp 0.65 \
  --mx-crop-origin OX OY \
  --out-png crop/registration_debug.png
```

> **Note:** `--mx-crop-origin` is required for crop images. Without it, CSV centroids project to
> full-slide MX coordinates and fall far outside the crop's H&E space.

---

## Files Changed

| File | Change |
|------|--------|
| `stages/refine_registration.py` | New functions `csv_to_he_coords`, `affine_icp`, `match_centroids_he`; updated `main()`, `_parse_args()`, `update_index()` |
| `tools/viz_registration_debug.py` | New tool |
| `tests/test_refine_registration.py` | 12 new tests; updated import list and `test_update_index` |

---

## CLI Reference

```bash
# Stage 1
python stages/patchify.py \
  --he-image data/WD-76845-096-crop.ome.tif \
  --multiplex-image data/WD-76845-097-crop.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out crop/ --patch-size 1024 --stride 512 --channels DNA

# (user runs Stage 2 on Colab → crop/cellvit/*.json)

# Diagnostic viz
python -m tools.viz_registration_debug \
  --processed crop/ \
  --he-image data/WD-76845-096-crop.ome.tif \
  --csv data/WD-76845-097.csv \
  --csv-mpp 0.65 --mx-crop-origin OX OY \
  --out-png crop/registration_debug.png

# Stage 2.5
python -m stages.refine_registration \
  --processed crop/ \
  --he-image data/WD-76845-096-crop.ome.tif \
  --multiplex-image data/WD-76845-097-crop.ome.tif \
  --csv data/WD-76845-097.csv \
  --csv-mpp 0.65 --mx-crop-origin OX OY \
  --distance-gate 20 --icp-max-iter 50 --max-tps-points 2000
```

---

## Key Design Decisions

**Why ICP before TPS?**
ICP corrects systematic global error in `m_full` (common when ECC registration falls back to
landmarks or centroid-based estimates). After ICP, the residual misalignment is local and
non-systematic — exactly what TPS is designed for. The two-stage decomposition makes each
correction independently interpretable and debuggable.

**Why work in H&E space?**
Both CellViT output (H&E patch centroids) and the visualization are naturally in H&E px space.
Converting CSV to H&E space first means that overlay panels c/d can be directly inspected on the
H&E image using the same coordinate system — no mental conversion needed.

**Why keep TPS source = original CellViT (not ICP-transformed)?**
The TPS maps H&E px → MX px for patch re-extraction. Using original CellViT H&E coords as source
(with ICP only used for improved matching, not changing the source space) means the TPS warp acts
as a direct replacement for `m_full` — no composed transforms needed at extraction time.
