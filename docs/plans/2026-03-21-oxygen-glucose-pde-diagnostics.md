# Oxygen & Glucose PDE Estimation — Diagnostic Analysis

**Date:** 2026-03-21
**Stage:** Stage 4 (`stages/multiplex_layers.py` + `stages/multiplex_layers_lib/wsi_pde.py`)
**Dataset:** CRC33 (`data/mx_crc33.ome.tif`)

---

## Background

Stage 4 estimates tissue oxygen and glucose levels using a WSI-scale PDE proxy. The solver (`solve_wsi_pde_map`) runs once on the entire WSI before the per-patch loop, then slices the result per patch via `extract_patch_from_coarse`.

The reported issue: **nearly all estimated oxygen/glucose values were high (everything appeared oxygenated/well-supplied).**

---

## Implementation: What the solver actually does

Despite the name "WSI-PDE", `solve_wsi_pde_map` uses a **closed-form WKB approximation**, not an iterative PDE solver:

```python
dist = distance_transform_edt(~vessel_mask)          # Euclidean distance from vessels (coarse px)
L_local = L_base / sqrt(k_map / k_base)              # Spatially varying decay length
u = exp(-dist / L_local)                             # O2/glucose proxy in [0, 1]
```

where `L_base = krogh_um / (mpp * ds)` (decay length in coarse pixels).

For homogeneous `k`, `exp(-dist/L)` is the **exact** solution to the Krogh cylinder diffusion model `D∇²u − ku = 0`. For spatially varying `k(x)` (from Ki67/CD68 demand), the local decay length is adjusted — this is the WKB approximation.

### Note on `--wsi-pde-max-iters`

The CLI argument `--wsi-pde-max-iters` (default 20000) is parsed, logged, and passed to `solve_wsi_pde_map`, but **never used**. The function signature accepts `max_iters` and `tol` but the body contains only the closed-form computation above. These are relics from an earlier iterative Jacobi solver design that was replaced. They have zero effect on the output.

---

## Why the solver is WSI-scale (not patch-scale)

The Krogh diffusion radius (160 µm for O₂, 450 µm for glucose) is **larger than a single patch** (~83 µm at 0.325 µm/px for a 256px patch). Solving per-patch would confine the domain — no patch would be large enough to show the full gradient from vessel to avascular zone. WSI-scale is the physically correct approach.

---

## Diagnostic 1: Small crop (`data/mx_crc33_crop.ome.tif`)

| Metric | Value |
|--------|-------|
| Coarse grid (ds=4) | 256 × 256 px |
| Vessel fraction | 3.0% |
| Otsu threshold | 0.48 (conservative) |
| `dist.max()` | **60 coarse px = ~78 µm** |
| `L_base` (Krogh=160 µm) | 123 coarse px |
| `dist.max() / L_base` | **0.49** |

**Conclusion:** Every pixel in the crop is within **half a decay length** of a vessel. Minimum O₂ = `exp(-0.49) ≈ 0.61` — no hypoxic regions possible.

| Krogh (O₂ / Glucose) | O₂ range | Glucose range |
|----------------------|----------|---------------|
| 160 µm / 450 µm | [0.54, 1.00] | [0.80, 1.00] |
| 60 µm / 100 µm | [0.19, 1.00] | [0.37, 1.00] |

The crop is too small (~333 µm wide) to contain a genuinely avascular region at this vessel density. **This was the root cause of the "everything oxygenated" observation.**

---

## Diagnostic 2: Full WSI (`data/mx_crc33.ome.tif`)

### Vessel mask statistics (ds=4)

| Metric | Value |
|--------|-------|
| Coarse grid | 13185 × 9089 px |
| Vessel fraction | **15.5%** |
| `mpp_coarse` | 1.3 µm/px |

### Distance-to-vessel statistics

| Stat | Coarse px | Physical (µm) |
|------|-----------|---------------|
| max | 1996 | **2595** |
| mean | 174 | 226 |
| p95 | 749 | 974 |
| p99 | 1164 | 1513 |
| `L_base` (Krogh=160 µm) | 123 | 160 |
| `dist.max / L_base` | **16.2×** | — |

### Distance histogram

| Distance from vessel | Pixel fraction |
|----------------------|---------------|
| < 13 µm | 36.9% |
| 13–26 µm | 9.1% |
| 26–39 µm | 4.9% |
| 39–65 µm | 4.7% |
| 65–97 µm | 2.9% |
| 97–130 µm | 2.4% |
| 130–195 µm | 4.6% |
| 195–260 µm | 4.4% |
| 260–390 µm | 8.1% |
| 390–650 µm | 10.7% |
| > 650 µm | **11.3%** |

**34.5% of tissue is beyond 195 µm (exceeds the 160 µm Krogh radius).**

### O₂ proxy distribution (WKB, Krogh=160 µm)

| Metric | Value |
|--------|-------|
| u.min | 0.000 |
| u.max | 1.000 |
| u.mean | 0.591 |
| u < 0.5 (hypoxic) | **40.6%** |
| u < 0.1 (severely hypoxic) | **23.3%** |
| u < 0.01 (anoxic) | 9.1% |

### Pipeline output (from `multiplex_layers.py` logs)

```
O2 WSI-PDE:      Krogh=160 µm, L_coarse=123.1 px → range [0.000, 1.000]
Glucose WSI-PDE: Krogh=450 µm, L_coarse=346.2 px → range [0.000, 1.000]

O2 WSI-PDE:      Krogh=60 µm,  L_coarse=46.2 px  → range [0.000, 1.000]
Glucose WSI-PDE: Krogh=100 µm, L_coarse=76.9 px  → range [0.000, 1.000]
```

Both Krogh settings span the full [0, 1] range. PDE solve time: ~25 s each. Per-patch extraction: ~50 patches/sec, 10,379 patches total, 0 skipped.

---

## Final Conclusions

### 1. The "everything oxygenated" issue was crop-specific

The small crop (~333 µm) had no region far enough from a vessel to go hypoxic. On the full WSI, 40.6% of tissue falls below O₂=0.5 and the full [0, 1] range is utilized. **No bug.**

### 2. `max_iters` / `tol` are dead parameters

These should be cleaned up from the function signature and CLI to avoid misleading users. The solver is a closed-form approximation.

### 3. Parameter guidance

| Use case | `--oxygen-krogh-um` | `--glucose-krogh-um` |
|----------|--------------------|--------------------|
| Full WSI (recommended) | 160 (default) | 450 (default) |
| Small crop / dense vasculature | 60–80 | 100–150 |

### 4. The WSI-scale architecture is correct

Solving on the full WSI is required because the Krogh diffusion radius exceeds patch size. The crop showed artificially flat results precisely because per-region (sub-WSI) estimation cannot capture tissue-scale gradients.
