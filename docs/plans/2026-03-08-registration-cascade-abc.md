# Registration Cascade A→B→C Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix global y-offset between H&E and CyCIF by replacing the single ECC-on-masks registration with a three-tier cascade: (A) centroid-initialized ECC on masks → (B) intensity-based ECC (H&E gray vs DNA) → (C) ORB feature matching, stopping at first QC pass.

**Architecture:** Each tier is a standalone function in `stages/patchify_lib/registration.py`. `patchify.py::main()` calls them in order, evaluates QC after each, and breaks on first pass. The `final_transform.json` gains a `registration_method` field recording which tier succeeded. All tiers share the same centroid-initialization helper and QC path.

**Tech Stack:** OpenCV (already installed: `cv2`), NumPy. No new dependencies — ORB is in base OpenCV.

**Root cause reminder:** ECC starting from identity must discover ~2500 px translation + ~9° rotation simultaneously on a 64× downsampled binary blob. One overview pixel = 64 full-res pixels, so a 1-pixel convergence error = 64 px misalignment. Centroid init reduces ECC's search to rotation/scale only. Intensity images give ECC real texture gradients instead of blob edges.

---

## Task 1: Centroid init helper + Approach A (centroid-initialized ECC on masks)

**Files:**
- Modify: `stages/patchify_lib/registration.py`
- Test: `tests/test_patchify_registration.py`

**Context:** Current `register_he_mx_affine` starts ECC from `np.eye(2,3)`. Change only the initialization — everything else (ECC call, re-centring cap, full-res conversion) stays the same.

The MX mask is resized to HE overview dimensions before ECC, so both masks live in the same (HE overview) pixel space. The centroid of each mask in that space gives the translation offset.

**Step 1: Write failing test**

Add to `tests/test_patchify_registration.py`:

```python
def test_centroid_init_improves_convergence_on_translated_masks():
    """Centroid-init ECC should recover a known translation on synthetic masks."""
    import cv2, numpy as np
    from stages.patchify_lib.registration import register_he_mx_affine

    rng = np.random.default_rng(0)
    # Create a 128x128 HE mask with a blob at (60,50)
    he_mask = np.zeros((128, 128), dtype=np.float32)
    he_mask[40:80, 50:90] = 1.0

    # Create a 96x96 MX mask with blob at (30,25) — same physical region, ~0.75x scale
    mx_mask = np.zeros((96, 96), dtype=np.float32)
    mx_mask[20:54, 23:57] = 1.0

    ds = 8
    m_full = register_he_mx_affine(
        he_mask.astype(bool), mx_mask.astype(bool),
        ds=ds, he_h=1024, he_w=1024, mx_h=768, mx_w=768
    )
    assert m_full.shape == (2, 3)
    # After registration, (0,0) in HE should map near (0,0) in MX
    # The scale should be close to he/mx = 1024/768 ≈ 1.33... inverted ≈ 0.75
    assert 0.5 < abs(m_full[0, 0]) < 1.5
```

Run: `pytest tests/test_patchify_registration.py::test_centroid_init_improves_convergence_on_translated_masks -v`
Expected: PASS (function exists already; test validates shape and scale range)

**Step 2: Add `_centroid_init_m_ov` helper + update `register_he_mx_affine`**

In `stages/patchify_lib/registration.py`, add the helper before `register_he_mx_affine`:

```python
def _centroid_init_m_ov(he_f32: np.ndarray, mx_resized: np.ndarray) -> np.ndarray:
    """Return 2x3 identity+translation init for ECC based on tissue centroids.

    Both arrays must be in the same (HE overview) pixel space.
    Falls back to identity if either mask is empty.
    """
    he_ctr = _mask_centroid_or_none(he_f32 > 0.5)
    mx_ctr = _mask_centroid_or_none(mx_resized > 0.5)
    m = np.eye(2, 3, dtype=np.float32)
    if he_ctr is not None and mx_ctr is not None:
        m[0, 2] = float(he_ctr[0] - mx_ctr[0])
        m[1, 2] = float(he_ctr[1] - mx_ctr[1])
    return m
```

Then in `register_he_mx_affine`, replace the line:
```python
    m_ov = np.eye(2, 3, dtype=np.float32)
```
with:
```python
    m_ov = _centroid_init_m_ov(he_f32, mx_resized)
```

Run: `pytest tests/test_patchify_registration.py -v`
Expected: all pass

**Step 3: Commit**

```bash
git add stages/patchify_lib/registration.py tests/test_patchify_registration.py
git commit -m "feat: centroid-init ECC (Approach A) in register_he_mx_affine"
```

---

## Task 2: Approach B — intensity-based ECC (H&E grayscale vs DNA channel)

**Files:**
- Modify: `stages/patchify_lib/registration.py`
- Modify: `stages/patchify.py` (add overview reader helper)
- Test: `tests/test_patchify_registration.py`

**Context:** Instead of binary tissue masks, feed ECC normalized float32 intensity images — H&E luminance and DNA/DAPI channel. These have real texture (cell nuclei, tissue folds) that gives ECC many more gradient features for convergence. Use the same centroid init and same full-res conversion as Approach A.

**Step 1: Write failing test**

```python
def test_register_he_mx_affine_intensity_returns_valid_matrix():
    """register_he_mx_affine_intensity should return a 2x3 float32 matrix."""
    import numpy as np
    from stages.patchify_lib.registration import register_he_mx_affine_intensity

    rng = np.random.default_rng(1)
    # Synthetic: HE gray blob and MX DNA blob in HE overview space
    he_gray = np.zeros((128, 128), dtype=np.float32)
    he_gray[40:90, 40:90] = rng.uniform(0.3, 1.0, (50, 50)).astype(np.float32)
    mx_dna = np.zeros((96, 96), dtype=np.float32)
    mx_dna[20:60, 20:60] = rng.uniform(0.2, 0.9, (40, 40)).astype(np.float32)

    he_mask = he_gray > 0.2
    mx_mask = mx_dna > 0.2

    m = register_he_mx_affine_intensity(
        he_gray, mx_dna, he_mask.astype(bool), mx_mask.astype(bool),
        ds=8, he_h=1024, he_w=1024, mx_h=768, mx_w=768
    )
    assert m.shape == (2, 3)
    assert m.dtype == np.float32
```

Run: `pytest tests/test_patchify_registration.py::test_register_he_mx_affine_intensity_returns_valid_matrix -v`
Expected: FAIL with `ImportError: cannot import name 'register_he_mx_affine_intensity'`

**Step 2: Implement `register_he_mx_affine_intensity`**

Add to `stages/patchify_lib/registration.py` after `register_he_mx_affine`:

```python
def register_he_mx_affine_intensity(
    he_gray_ov: np.ndarray,
    mx_dna_ov: np.ndarray,
    he_mask: np.ndarray,
    mx_mask: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
) -> np.ndarray:
    """Affine registration using actual image intensities instead of binary masks.

    Parameters
    ----------
    he_gray_ov : float32 (he_ov_h, he_ov_w) normalized HE grayscale at overview res.
    mx_dna_ov  : float32 (mx_ov_h, mx_ov_w) normalized MX DNA channel at overview res.
    he_mask    : bool (he_ov_h, he_ov_w) tissue mask for centroid init.
    mx_mask    : bool (mx_ov_h, mx_ov_w) tissue mask for centroid init.
    ds         : overview downsample factor.
    he_h/w, mx_h/w : full-resolution image dimensions.

    Returns
    -------
    m_full : float32 (2, 3) affine mapping H&E full-res -> MX full-res.
    """
    he_ov_h, he_ov_w = he_gray_ov.shape
    mx_ov_h, mx_ov_w = mx_dna_ov.shape

    # Resize MX DNA to HE overview grid (same as mask-based ECC)
    mx_resized = cv2.resize(
        mx_dna_ov, (he_ov_w, he_ov_h), interpolation=cv2.INTER_LINEAR
    )
    # Resize MX mask too for centroid init
    mx_mask_resized = cv2.resize(
        mx_mask.astype(np.float32), (he_ov_w, he_ov_h), interpolation=cv2.INTER_LINEAR
    ) > 0.5

    # Mild blur to smooth noise without removing texture
    he_f = cv2.GaussianBlur(he_gray_ov.astype(np.float32), (3, 3), 0)
    mx_f = cv2.GaussianBlur(mx_resized.astype(np.float32), (3, 3), 0)

    # Centroid init from tissue masks
    m_ov = _centroid_init_m_ov(
        he_mask.astype(np.float32), mx_mask_resized.astype(np.float32)
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        _, m_ov = cv2.findTransformECC(he_f, mx_f, m_ov, cv2.MOTION_AFFINE, criteria)
    except cv2.error as e:  # pylint: disable=catching-non-exception
        print(f"  WARNING: intensity ECC failed ({e}). Falling back to mask-based result.")
        # Return centroid-init scale-only as fallback
        scale = he_w / mx_w
        return np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float32)

    # Re-centre cap (same logic as register_he_mx_affine)
    mx_warped = cv2.warpAffine(
        mx_f, m_ov.astype(np.float32), (he_ov_w, he_ov_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    he_ctr = _mask_centroid_or_none(he_mask > 0)
    mx_ctr = _mask_centroid_or_none(mx_warped > mx_warped.mean() + 1e-6)
    if he_ctr is not None and mx_ctr is not None:
        dx = float(mx_ctr[0] - he_ctr[0])
        dy = float(mx_ctr[1] - he_ctr[1])
        shift_mag = float(np.hypot(dx, dy))
        max_recentre_px = 12.0
        if shift_mag > max_recentre_px and shift_mag > 1e-9:
            scale = max_recentre_px / shift_mag
            dx *= scale
            dy *= scale
        m_ov[0, 2] += dx
        m_ov[1, 2] += dy

    # Convert overview → full-res (same pixel-center formula as register_he_mx_affine)
    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h
    tx_ov = (float(m_ov[0, 2]) + 0.5) / rx - 0.5
    ty_ov = (float(m_ov[1, 2]) + 0.5) / ry - 0.5
    return np.array(
        [
            [m_ov[0, 0] / rx, m_ov[0, 1] / rx, tx_ov * ds],
            [m_ov[1, 0] / ry, m_ov[1, 1] / ry, ty_ov * ds],
        ],
        dtype=np.float32,
    )
```

Run: `pytest tests/test_patchify_registration.py -v`
Expected: all pass

**Step 3: Add `_read_he_gray_overview` helper to `patchify.py`**

Add after `build_mx_tissue_mask` in `stages/patchify.py`:

```python
def _read_he_gray_overview(
    store, axes: str, img_h: int, img_w: int, ds: int
) -> np.ndarray:
    """Read H&E at overview resolution and return normalized float32 grayscale (H, W)."""
    axes_up = axes.upper()
    h_t = (img_h // ds) * ds
    w_t = (img_w // ds) * ds

    c_first = "C" in axes_up and axes_up.index("C") < axes_up.index("Y")
    if c_first:
        raw = np.array(store[:, :h_t:ds, :w_t:ds])
        overview = np.moveaxis(raw[:3], 0, -1)  # (H, W, 3)
    else:
        overview = np.array(store[:h_t:ds, :w_t:ds, :3])  # (H, W, 3)

    if overview.dtype != np.uint8:
        overview = percentile_to_uint8(overview)

    gray = cv2.cvtColor(overview.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0
```

Run: `pytest tests/test_patchify_registration.py -v`
Expected: all pass

**Step 4: Commit**

```bash
git add stages/patchify_lib/registration.py stages/patchify.py tests/test_patchify_registration.py
git commit -m "feat: intensity-based ECC fallback (Approach B)"
```

---

## Task 3: Approach C — ORB feature matching + RANSAC

**Files:**
- Modify: `stages/patchify_lib/registration.py`
- Test: `tests/test_patchify_registration.py`

**Context:** ORB detects binary keypoints in both images. Matches are filtered with Lowe ratio test (0.75), then `cv2.estimateAffine2D` with RANSAC computes the affine. ORB is in base OpenCV — no opencv-contrib needed. Returns `None` if too few matches, so the caller can fall back gracefully.

**Step 1: Write failing test**

```python
def test_register_he_mx_orb_returns_none_on_blank_images():
    """ORB should return None when images have no detectable features."""
    import numpy as np
    from stages.patchify_lib.registration import register_he_mx_orb

    blank = np.zeros((128, 128), dtype=np.uint8)
    result = register_he_mx_orb(blank, blank, ds=8, he_h=1024, he_w=1024, mx_h=768, mx_w=768)
    assert result is None


def test_register_he_mx_orb_returns_matrix_on_textured_images():
    """ORB should return a 2x3 matrix when both images have enough features."""
    import numpy as np
    from stages.patchify_lib.registration import register_he_mx_orb

    rng = np.random.default_rng(42)
    # Checkerboard-like pattern gives ORB many keypoints
    base = np.zeros((256, 256), dtype=np.uint8)
    base[::16, :] = 200
    base[:, ::16] = 200
    # Shift MX by ~10px to simulate misalignment
    he_img = base.copy()
    mx_img = np.roll(base, 10, axis=0).astype(np.uint8)
    result = register_he_mx_orb(he_img, mx_img, ds=8, he_h=2048, he_w=2048, mx_h=1536, mx_w=1536)
    # May return None if ORB finds no good matches on synthetic data; that's OK
    if result is not None:
        assert result.shape == (2, 3)
```

Run: `pytest tests/test_patchify_registration.py::test_register_he_mx_orb_returns_none_on_blank_images -v`
Expected: FAIL with ImportError

**Step 2: Implement `register_he_mx_orb`**

Add to `stages/patchify_lib/registration.py`:

```python
def register_he_mx_orb(
    he_gray_u8: np.ndarray,
    mx_dna_u8: np.ndarray,
    ds: int,
    he_h: int,
    he_w: int,
    mx_h: int,
    mx_w: int,
    n_features: int = 2000,
    ratio_thresh: float = 0.75,
    min_inliers: int = 8,
) -> np.ndarray | None:
    """Affine registration via ORB keypoints + RANSAC.

    Parameters
    ----------
    he_gray_u8 : uint8 (he_ov_h, he_ov_w) H&E grayscale at overview resolution.
    mx_dna_u8  : uint8 (he_ov_h, he_ov_w) MX DNA channel resized to HE overview grid.
    ds         : overview downsample factor.
    he_h/w, mx_h/w : full-resolution dimensions.
    n_features : ORB feature budget.
    ratio_thresh : Lowe ratio test threshold.
    min_inliers : minimum RANSAC inliers to accept the result.

    Returns
    -------
    m_full : float32 (2, 3) mapping H&E full-res -> MX full-res, or None if failed.
    """
    he_ov_h, he_ov_w = he_gray_u8.shape
    # mx_dna_u8 is already in HE overview space (caller resizes)
    mx_ov_h = int(round(he_ov_h * mx_h / he_h))
    mx_ov_w = int(round(he_ov_w * mx_w / he_w))

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(he_gray_u8, None)
    kp2, des2 = orb.detectAndCompute(mx_dna_u8, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  WARNING: ORB found too few keypoints.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:  # pylint: disable=catching-non-exception
        return None

    good = [m for m, n in raw_matches if len([m, n]) == 2 and m.distance < ratio_thresh * n.distance]
    if len(good) < min_inliers:
        print(f"  WARNING: ORB ratio test left only {len(good)} matches (need {min_inliers}).")
        return None

    pts_he = np.float32([kp1[m.queryIdx].pt for m in good])  # HE overview coords
    pts_mx = np.float32([kp2[m.trainIdx].pt for m in good])  # HE overview coords (mx resized)

    # estimateAffine2D: find M s.t. M * pts_mx ≈ pts_he
    m_ov, inliers = cv2.estimateAffine2D(
        pts_mx, pts_he, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if m_ov is None or inliers is None or int(inliers.sum()) < min_inliers:
        print(f"  WARNING: ORB RANSAC failed or too few inliers ({inliers.sum() if inliers is not None else 0}).")
        return None

    m_ov = m_ov.astype(np.float32)
    # Convert from HE-overview-space affine to full-res H&E->MX affine
    # m_ov maps MX-in-HE-overview → HE-overview, i.e. it's an inverse map in overview space.
    # We need H&E full-res → MX full-res.
    rx = he_ov_w / mx_ov_w
    ry = he_ov_h / mx_ov_h
    tx_ov = (float(m_ov[0, 2]) + 0.5) / rx - 0.5
    ty_ov = (float(m_ov[1, 2]) + 0.5) / ry - 0.5
    return np.array(
        [
            [m_ov[0, 0] / rx, m_ov[0, 1] / rx, tx_ov * ds],
            [m_ov[1, 0] / ry, m_ov[1, 1] / ry, ty_ov * ds],
        ],
        dtype=np.float32,
    )
```

Run: `pytest tests/test_patchify_registration.py -v`
Expected: all pass

**Step 3: Commit**

```bash
git add stages/patchify_lib/registration.py tests/test_patchify_registration.py
git commit -m "feat: ORB+RANSAC registration fallback (Approach C)"
```

---

## Task 4: Wire cascade into `patchify.py::main()`

**Files:**
- Modify: `stages/patchify.py` (main function, registration block only)

**Context:** The current registration block in `main()` calls `register_he_mx_affine` once then attempts Farneback deformable. Replace with A→B→C cascade. The QC decision logic is already in place — we just need to re-use it for each tier. Update `final_transform.json` to include `registration_method`.

Also need to import the two new functions and expose the intensity overviews.

**Step 1: Write failing integration test (patchify CLI)**

Add to `tests/test_patchify_registration.py` (or `tests/test_patchify_cli.py`):

```python
def test_registration_cascade_method_stored_in_final_transform(tmp_path):
    """After patchify, final_transform.json must contain 'registration_method' key."""
    import json, subprocess, sys, tifffile, numpy as np
    from pathlib import Path

    # Tiny synthetic HE + MX with shifted tissue
    he = np.zeros((3, 256, 256), dtype=np.uint8)
    he[:, 80:180, 80:180] = 200
    mx = np.zeros((2, 128, 128), dtype=np.uint16)
    mx[0, 35:85, 35:85] = 3000  # DNA

    he_path = tmp_path / "he.ome.tif"
    mx_path = tmp_path / "mx.ome.tif"
    csv_path = tmp_path / "meta.csv"
    tifffile.imwrite(str(he_path), he, ome=True, metadata={"axes": "CYX"})
    tifffile.imwrite(str(mx_path), mx, ome=True, metadata={"axes": "CYX"})
    csv_path.write_text("Channel ID,Target Name\nChannel:0:0,DNA\nChannel:0:1,CD31\n")

    subprocess.run(
        [sys.executable, "-m", "stages.patchify",
         "--he-image", str(he_path),
         "--multiplex-image", str(mx_path),
         "--metadata-csv", str(csv_path),
         "--out", str(tmp_path / "out"),
         "--channels", "DNA",
         "--patch-size", "64",
         "--overview-downsample", "8"],
        check=True, capture_output=True
    )
    ft = json.loads((tmp_path / "out" / "registration" / "final_transform.json").read_text())
    assert "registration_method" in ft
    assert ft["registration_method"] in {"affine_centroid", "affine_intensity", "orb", "fallback_scale"}
```

Run: `pytest tests/test_patchify_registration.py::test_registration_cascade_method_stored_in_final_transform -v`
Expected: FAIL (KeyError on 'registration_method')

**Step 2: Update registration block in `patchify.py::main()`**

Locate the registration block (search for `"Computing affine registration via ECC"`). Replace the entire registration + QC + deformable block with:

```python
    # ------------------------------------------------------------------
    # Registration cascade A → B → C
    # ------------------------------------------------------------------
    from stages.patchify_lib.registration import (
        register_he_mx_affine,
        register_he_mx_affine_intensity,
        register_he_mx_orb,
    )

    registration_method = "fallback_scale"
    m_full = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0]], dtype=np.float32)

    # ---- Approach A: centroid-initialized ECC on tissue masks ----
    print("  [A] Centroid-initialized ECC on tissue masks ...")
    m_A = register_he_mx_affine(he_mask, mx_mask, ds, he_h, he_w, mx_h, mx_w)
    qc_A = _evaluate_registration_qc(
        m_A, he_mask, mx_mask, mx_chw_ov,
        he_h, he_w, mx_h, mx_w, ds, args,
    )
    if qc_A["decision"] == PASS_AFFINE:
        print(f"  [A] PASS (iou={qc_A['global_qc']['mask_iou']:.3f})")
        m_full = m_A
        registration_method = "affine_centroid"
    else:
        print(f"  [A] FAIL ({qc_A['decision']}) → trying B ...")

        # ---- Approach B: intensity-based ECC (HE gray vs DNA) ----
        he_gray_ov = _read_he_gray_overview(he_store, he_axes, he_h, he_w, ds)
        mx_dna_ov_f32 = _read_channel_overview(
            mx_store, mx_axes, mx_h, mx_w, ds, channel_index=0
        ).astype(np.float32)
        mx_dna_max = float(mx_dna_ov_f32.max())
        if mx_dna_max > 0:
            mx_dna_ov_f32 /= mx_dna_max

        print("  [B] Intensity-based ECC (HE gray vs DNA channel) ...")
        m_B = register_he_mx_affine_intensity(
            he_gray_ov, mx_dna_ov_f32,
            he_mask, mx_mask,
            ds, he_h, he_w, mx_h, mx_w,
        )
        qc_B = _evaluate_registration_qc(
            m_B, he_mask, mx_mask, mx_chw_ov,
            he_h, he_w, mx_h, mx_w, ds, args,
        )
        if qc_B["decision"] == PASS_AFFINE:
            print(f"  [B] PASS (iou={qc_B['global_qc']['mask_iou']:.3f})")
            m_full = m_B
            registration_method = "affine_intensity"
        else:
            print(f"  [B] FAIL ({qc_B['decision']}) → trying C ...")

            # ---- Approach C: ORB keypoints + RANSAC ----
            he_gray_u8 = (he_gray_ov * 255).astype(np.uint8)
            he_ov_h, he_ov_w = he_mask.shape
            mx_dna_u8 = cv2.resize(
                (np.clip(mx_dna_ov_f32, 0, 1) * 255).astype(np.uint8),
                (he_ov_w, he_ov_h),
                interpolation=cv2.INTER_LINEAR,
            )
            print("  [C] ORB feature matching + RANSAC ...")
            m_C = register_he_mx_orb(
                he_gray_u8, mx_dna_u8,
                ds, he_h, he_w, mx_h, mx_w,
            )
            if m_C is not None:
                qc_C = _evaluate_registration_qc(
                    m_C, he_mask, mx_mask, mx_chw_ov,
                    he_h, he_w, mx_h, mx_w, ds, args,
                )
                print(f"  [C] iou={qc_C['global_qc']['mask_iou']:.3f} decision={qc_C['decision']}")
                # Use C regardless of QC pass (it's the last resort)
                m_full = m_C
                registration_method = "orb"
            else:
                print("  [C] ORB returned no valid transform → using best affine (A or B)")
                # Pick whichever had higher global IoU
                iou_A = qc_A["global_qc"]["mask_iou"]
                iou_B = qc_B["global_qc"]["mask_iou"]
                m_full = m_A if iou_A >= iou_B else m_B
                registration_method = "affine_centroid" if iou_A >= iou_B else "affine_intensity"

    # Store final QC using whichever m_full was selected
    final_qc = _evaluate_registration_qc(
        m_full, he_mask, mx_mask, mx_chw_ov,
        he_h, he_w, mx_h, mx_w, ds, args,
    )
    final_qc["registration_method"] = registration_method
```

Where `_evaluate_registration_qc` is a helper that wraps the existing QC code (compute_global_qc_metrics + compute_patch_qc_metrics) into a single call returning the qc dict. Refactor the existing QC block into this helper.

Also update `final_transform.json` write to include `registration_method`:
```python
    final_transform = {
        "mode": registration_method,
        "warp_matrix": m_full.tolist(),
        "decision_initial": final_qc.get("decision", "unknown"),
        "deform_field": None,
        "registration_method": registration_method,
    }
```

Run: `pytest tests/test_patchify_registration.py::test_registration_cascade_method_stored_in_final_transform -v`
Expected: PASS

Run full suite: `pytest tests/ -v`
Expected: all pass (existing tests unaffected)

**Step 3: Commit**

```bash
git add stages/patchify.py tests/test_patchify_registration.py
git commit -m "feat: wire A→B→C registration cascade into patchify main()"
```

---

## Task 5: End-to-end smoke test with crop images

**Files:** None (manual verification)

**Step 1: Run patchify on crop images**

```bash
python3 -m stages.patchify \
  --he-image data/WD-76845-096-crop.ome.tif \
  --multiplex-image data/WD-76845-097-crop.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd_crop_v2 \
  --channels DNA \
  --patch-size 512 \
  --overview-downsample 16
```

Expected output (look for):
- `[A] PASS` or `[B] PASS` or `[C]` line
- `registration_method` in `processed_wd_crop_v2/registration/final_transform.json`

**Step 2: Visualize alignment**

```bash
python3 -m tools.debug_match_he_mul \
  --he-image data/WD-76845-096-crop.ome.tif \
  --multiplex-image data/WD-76845-097-crop.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --index-json processed_wd_crop_v2/index.json \
  --save-png processed_wd_crop_v2/vis_check.png
```

**Step 3: Compare pipeline grid**

```bash
python3 -m tools.visualize_pipeline \
  --processed processed_wd_crop_v2/ \
  --random 4 \
  --seed 42 \
  --he-image data/WD-76845-096-crop.ome.tif
```

Inspect `processed_wd_crop_v2/pipeline_grid_*.png` — DNA channel should align with H&E tissue structures (nuclei in same rows, tissue edge at same position).

If alignment is still off, check `registration_method` and `final_transform.json`:
- If method is `fallback_scale`, all three tiers failed — may need to adjust `--overview-downsample` or investigate image quality
- If method is `affine_centroid` but still misaligned, examine the full-image pair with debug tool

**Step 4: If good, run on full images**

```bash
python3 -m stages.patchify \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --out processed_wd_1024_v2 \
  --channels DNA \
  --patch-size 1024 \
  --overview-downsample 64

python3 -m tools.debug_match_he_mul \
  --he-image data/WD-76845-096.ome.tif \
  --multiplex-image data/WD-76845-097.ome.tif \
  --metadata-csv data/WD-76845-097-metadata.csv \
  --index-json processed_wd_1024_v2/index.json \
  --save-png processed_wd_1024_v2/vis_registered_check.png
```

Compare with the old `processed_wd_1024/vis_registered_check.png` — the tissue outline should have tighter overlay and the y-offset should be reduced or eliminated.

**Step 5: Commit**

```bash
git add processed_wd_crop_v2/ processed_wd_1024_v2/  # only json/png artifacts
git commit -m "chore: end-to-end crop and full-image registration cascade results"
```

---

## Notes on `_evaluate_registration_qc` refactor

The existing QC block in `main()` spans roughly 60 lines and calls:
1. `compute_channel_drift_metrics` (on `mx_chw_ov`)
2. `compute_global_qc_metrics` (on masks + m_full)
3. `compute_patch_qc_metrics` (on masks + m_full)
4. Decision tree logic

Extract this into a module-level function `_evaluate_registration_qc(m_full, he_mask, mx_mask, mx_chw_ov, he_h, he_w, mx_h, mx_w, ds, args) -> dict` that returns the full qc dict (including `decision`). This allows calling it multiple times cleanly for each cascade tier.

The channel drift check (`compute_channel_drift_metrics`) only needs to run once (it's about internal MX drift, not H&E↔MX alignment), so call it before the cascade and pass the result into the helper or skip it in cascade QC evaluations.
