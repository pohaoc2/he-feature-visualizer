import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2

from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize


HE_PATH = "data/WD-76845-096-small.ome.tif"
CYCIF_PATH = "data/WD-76845-097-small.ome.tif"
OUT_DIR = "registration_output"
os.makedirs(OUT_DIR, exist_ok=True)


def read_ome_first_series(path):
    """Read first OME-TIFF series as numpy array."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        arr = series.asarray()
        axes = series.axes
        ome_metadata = tif.ome_metadata
    return arr, axes, ome_metadata


def normalize01(x, pmin=1, pmax=99):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [pmin, pmax])
    x = np.clip(x, lo, hi)
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x


def to_uint8(x):
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


def resize_to_match(moving, fixed):
    if moving.shape != fixed.shape:
        moving = resize(
            moving,
            fixed.shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)
    return moving


def parse_he_rgb(arr, axes):
    """
    Convert H&E image to RGB in YXC layout.
    Supports common layouts:
      - YXS
      - SYX
      - CYX with 3 or 4 channels
      - YXC
    """
    axes = axes.upper()

    if axes in ("YXS", "YXC"):
        rgb = arr
    elif axes == "SYX":
        rgb = np.moveaxis(arr, 0, -1)
    elif axes == "CYX" and arr.shape[0] in (3, 4):
        rgb = np.moveaxis(arr[:3], 0, -1)
    else:
        raise ValueError(f"Unsupported H&E axes/shape: axes={axes}, shape={arr.shape}")

    rgb = rgb[..., :3]

    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
        maxv = rgb.max()
        rgb /= maxv if maxv > 0 else 1.0

    rgb = np.clip(rgb, 0, 1)
    return rgb


def extract_he_nuclei(he_arr, he_axes):
    """
    Use hematoxylin channel from HED color deconvolution as nuclei proxy.
    """
    rgb = parse_he_rgb(he_arr, he_axes)
    hed = rgb2hed(rgb)

    # H channel in HED often becomes more negative for stronger stain
    hematoxylin = -hed[..., 0]
    hematoxylin = normalize01(hematoxylin)
    hematoxylin = gaussian(hematoxylin, sigma=1.0)

    return rgb, hematoxylin


def extract_cycif_channel(cycif_arr, cycif_axes, channel_index=0):
    """
    Extract one channel from CyCIF image.
    Supports CYX or YXC.
    """
    cycif_axes = cycif_axes.upper()

    if cycif_axes == "CYX":
        ch = cycif_arr[channel_index]
    elif cycif_axes in ("YXC", "YXS"):
        ch = cycif_arr[..., channel_index]
    else:
        raise ValueError(f"Unsupported CyCIF axes/shape: axes={cycif_axes}, shape={cycif_arr.shape}")

    ch = normalize01(ch)
    ch = gaussian(ch, sigma=1.0)
    return ch


def preview_cycif_channels(cycif_arr, cycif_axes, out_path, max_channels=12):
    """
    Save a grid preview of CyCIF channels to help find DAPI.
    """
    cycif_axes = cycif_axes.upper()

    if cycif_axes == "CYX":
        n_channels = cycif_arr.shape[0]
        getter = lambda i: cycif_arr[i]
    elif cycif_axes in ("YXC", "YXS"):
        n_channels = cycif_arr.shape[-1]
        getter = lambda i: cycif_arr[..., i]
    else:
        raise ValueError(f"Unsupported CyCIF axes/shape: axes={cycif_axes}, shape={cycif_arr.shape}")

    n_show = min(n_channels, max_channels)
    ncols = 4
    nrows = int(np.ceil(n_show / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axs = np.array(axs).ravel()

    for i in range(n_show):
        ch = normalize01(getter(i))
        axs[i].imshow(ch, cmap="gray")
        axs[i].set_title(f"channel {i}")
        axs[i].axis("off")

    for j in range(n_show, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def make_tissue_mask(img, blur_sigma=3.0, min_fraction=0.001):
    """
    Build a simple tissue mask from a grayscale image.
    """
    img = gaussian(normalize01(img), sigma=blur_sigma)

    try:
        th = threshold_otsu(img)
    except ValueError:
        th = 0.1

    mask = img > th

    # fallback if mask too small
    if mask.mean() < min_fraction:
        th = np.percentile(img, 70)
        mask = img > th

    return mask.astype(np.uint8)


def centroid_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return np.array([w / 2, h / 2], dtype=np.float32)
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def translation_init_from_masks(moving, fixed):
    """
    Create an initial translation warp by matching tissue centroids.
    """
    moving_mask = make_tissue_mask(moving)
    fixed_mask = make_tissue_mask(fixed)

    c_m = centroid_from_mask(moving_mask)
    c_f = centroid_from_mask(fixed_mask)

    dx, dy = (c_f - c_m).astype(np.float32)

    warp = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
    ], dtype=np.float32)

    return warp, moving_mask, fixed_mask


def edge_map(img):
    """
    Convert grayscale image to edge map for more stable ECC.
    """
    u8 = to_uint8(normalize01(img))
    edges = cv2.Canny(u8, 50, 150).astype(np.float32) / 255.0
    return edges


def warp_image(image, warp, out_shape, warp_mode=cv2.MOTION_AFFINE):
    h, w = out_shape

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(
            image,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        return cv2.warpAffine(
            image,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


def overlay_rgb(fixed_gray, moving_gray):
    """
    fixed -> green, moving -> magenta
    """
    fixed_gray = normalize01(fixed_gray)
    moving_gray = normalize01(moving_gray)
    out = np.zeros((*fixed_gray.shape, 3), dtype=np.float32)
    out[..., 1] = fixed_gray
    out[..., 0] = moving_gray
    out[..., 2] = moving_gray
    return np.clip(out, 0, 1)


def ecc_register(moving, fixed, warp_mode=cv2.MOTION_EUCLIDEAN, n_iters=1000, eps=1e-6, init_warp=None):
    """
    Register moving -> fixed using OpenCV ECC.
    Returns (cc, warp). If ECC fails, cc is None.
    """
    moving = resize_to_match(moving, fixed).astype(np.float32)
    fixed = fixed.astype(np.float32)

    moving_u8 = to_uint8(moving)
    fixed_u8 = to_uint8(fixed)

    if init_warp is not None:
        warp = init_warp.astype(np.float32).copy()
    else:
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp = np.eye(3, 3, dtype=np.float32)
        else:
            warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iters,
        eps,
    )

    try:
        cc, warp = cv2.findTransformECC(
            templateImage=fixed_u8,
            inputImage=moving_u8,
            warpMatrix=warp,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        return cc, warp
    except cv2.error as e:
        print("ECC failed:", e)
        return None, warp


def try_register_variant(moving_img, fixed_img, name):
    """
    Robust two-stage registration:
      1) edge maps + centroid init + Euclidean ECC
      2) affine refinement
    """
    moving_reg = edge_map(moving_img)
    fixed_reg = edge_map(fixed_img)

    init_warp, moving_mask, fixed_mask = translation_init_from_masks(moving_reg, fixed_reg)

    cc1, warp1 = ecc_register(
        moving=moving_reg,
        fixed=fixed_reg,
        warp_mode=cv2.MOTION_EUCLIDEAN,
        n_iters=1000,
        eps=1e-6,
        init_warp=init_warp,
    )

    if cc1 is None:
        return {
            "name": name,
            "success": False,
            "cc_euclidean": None,
            "cc_final": None,
            "warp": warp1,
            "moving_mask": moving_mask,
            "fixed_mask": fixed_mask,
        }

    cc2, warp2 = ecc_register(
        moving=moving_reg,
        fixed=fixed_reg,
        warp_mode=cv2.MOTION_AFFINE,
        n_iters=1500,
        eps=1e-6,
        init_warp=warp1,
    )

    if cc2 is not None:
        final_cc = cc2
        final_warp = warp2
    else:
        final_cc = cc1
        final_warp = warp1

    return {
        "name": name,
        "success": True,
        "cc_euclidean": cc1,
        "cc_final": final_cc,
        "warp": final_warp,
        "moving_mask": moving_mask,
        "fixed_mask": fixed_mask,
    }


def apply_variant_transform(variant_name, img):
    """
    Apply the same pre-transform used during registration candidate testing.
    """
    if variant_name == "orig":
        return img
    elif variant_name == "flip_lr":
        return np.fliplr(img)
    elif variant_name == "flip_ud":
        return np.flipud(img)
    elif variant_name == "rot180":
        return np.rot90(img, 2)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")


def save_registration_summary(
    out_path,
    he_rgb_rs,
    he_nuc_rs,
    cy_nuc,
    he_nuc_variant,
    he_nuc_aligned,
    he_rgb_aligned,
    moving_mask,
    fixed_mask,
    variant_name,
    cc,
):
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    axes[0, 0].imshow(he_rgb_rs)
    axes[0, 0].set_title("H&E resized")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(he_nuc_rs, cmap="gray")
    axes[0, 1].set_title("H&E hematoxylin")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cy_nuc, cmap="gray")
    axes[0, 2].set_title("CyCIF nuclei-like channel")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(he_nuc_variant, cmap="gray")
    axes[1, 0].set_title(f"H&E variant: {variant_name}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_rgb(cy_nuc, he_nuc_variant))
    axes[1, 1].set_title("Before alignment")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlay_rgb(cy_nuc, he_nuc_aligned))
    axes[1, 2].set_title(f"After alignment (cc={cc:.4f})" if cc is not None else "After alignment")
    axes[1, 2].axis("off")

    axes[2, 0].imshow(moving_mask, cmap="gray")
    axes[2, 0].set_title("Moving tissue mask")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(fixed_mask, cmap="gray")
    axes[2, 1].set_title("Fixed tissue mask")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(np.clip(he_rgb_aligned, 0, 1))
    axes[2, 2].set_title("Aligned H&E RGB")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    print("Loading images...")
    he_arr, he_axes, he_ome = read_ome_first_series(HE_PATH)
    cy_arr, cy_axes, cy_ome = read_ome_first_series(CYCIF_PATH)

    print("H&E axes:", he_axes, "shape:", he_arr.shape)
    print("CyCIF axes:", cy_axes, "shape:", cy_arr.shape)

    # Save channel preview to help identify DAPI
    print("Saving CyCIF channel preview...")
    preview_cycif_channels(
        cy_arr,
        cy_axes,
        out_path=os.path.join(OUT_DIR, "cycif_channel_preview.png"),
        max_channels=12
    )

    # -------- user may need to change this after inspecting preview --------
    dapi_index = 0
    # ----------------------------------------------------------------------

    print(f"Using CyCIF channel {dapi_index} as nuclei-like channel")

    # Extract nuclei-like images
    he_rgb, he_nuc = extract_he_nuclei(he_arr, he_axes)
    cy_nuc = extract_cycif_channel(cy_arr, cy_axes, channel_index=dapi_index)

    # Resize H&E images into CyCIF space size
    he_nuc_rs = resize_to_match(he_nuc, cy_nuc)

    he_rgb_rs = resize(
        he_rgb,
        (*cy_nuc.shape, 3),
        order=1,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    # Try several orientation variants
    candidates = {
        "orig": he_nuc_rs,
        "flip_lr": np.fliplr(he_nuc_rs),
        "flip_ud": np.flipud(he_nuc_rs),
        "rot180": np.rot90(he_nuc_rs, 2),
    }

    print("Trying registration variants...")
    results = []
    for name, moving_img in candidates.items():
        print(f"  - testing variant: {name}")
        result = try_register_variant(moving_img, cy_nuc, name=name)
        results.append(result)

    successful = [r for r in results if r["success"] and r["cc_final"] is not None]

    if not successful:
        raise RuntimeError(
            "All registration attempts failed. "
            "Most likely causes: wrong CyCIF channel, different crop, or stronger local deformation."
        )

    best = max(successful, key=lambda r: r["cc_final"])
    best_variant = best["name"]
    best_warp = best["warp"]
    best_cc = best["cc_final"]

    print("\nBest result")
    print("Variant:", best_variant)
    print("ECC correlation:", best_cc)
    print("Warp matrix:\n", best_warp)

    # Apply best pre-transform to H&E
    he_nuc_variant = apply_variant_transform(best_variant, he_nuc_rs)
    he_rgb_variant = apply_variant_transform(best_variant, he_rgb_rs)

    # Warp into CyCIF space
    he_nuc_aligned = warp_image(
        he_nuc_variant,
        best_warp,
        cy_nuc.shape,
        warp_mode=cv2.MOTION_AFFINE
    )

    he_rgb_aligned = np.stack([
        warp_image(
            he_rgb_variant[..., c],
            best_warp,
            cy_nuc.shape,
            warp_mode=cv2.MOTION_AFFINE
        )
        for c in range(3)
    ], axis=-1)

    # Save outputs
    np.save(os.path.join(OUT_DIR, "affine_matrix.npy"), best_warp)

    plt.imsave(os.path.join(OUT_DIR, "he_rgb_resized.png"), np.clip(he_rgb_rs, 0, 1))
    plt.imsave(os.path.join(OUT_DIR, "he_nuclei.png"), he_nuc_rs, cmap="gray")
    plt.imsave(os.path.join(OUT_DIR, "cycif_nuclei_channel.png"), cy_nuc, cmap="gray")
    plt.imsave(os.path.join(OUT_DIR, "he_variant_before_alignment.png"), he_nuc_variant, cmap="gray")
    plt.imsave(os.path.join(OUT_DIR, "he_nuclei_aligned.png"), he_nuc_aligned, cmap="gray")
    plt.imsave(os.path.join(OUT_DIR, "he_rgb_aligned.png"), np.clip(he_rgb_aligned, 0, 1))
    plt.imsave(os.path.join(OUT_DIR, "overlay_before.png"), overlay_rgb(cy_nuc, he_nuc_variant))
    plt.imsave(os.path.join(OUT_DIR, "overlay_after.png"), overlay_rgb(cy_nuc, he_nuc_aligned))
    plt.imsave(os.path.join(OUT_DIR, "moving_mask.png"), best["moving_mask"], cmap="gray")
    plt.imsave(os.path.join(OUT_DIR, "fixed_mask.png"), best["fixed_mask"], cmap="gray")

    save_registration_summary(
        out_path=os.path.join(OUT_DIR, "registration_summary.png"),
        he_rgb_rs=he_rgb_rs,
        he_nuc_rs=he_nuc_rs,
        cy_nuc=cy_nuc,
        he_nuc_variant=he_nuc_variant,
        he_nuc_aligned=he_nuc_aligned,
        he_rgb_aligned=he_rgb_aligned,
        moving_mask=best["moving_mask"],
        fixed_mask=best["fixed_mask"],
        variant_name=best_variant,
        cc=best_cc,
    )

    # Save a text summary
    with open(os.path.join(OUT_DIR, "registration_report.txt"), "w") as f:
        f.write(f"H&E path: {HE_PATH}\n")
        f.write(f"CyCIF path: {CYCIF_PATH}\n")
        f.write(f"H&E axes: {he_axes}, shape: {he_arr.shape}\n")
        f.write(f"CyCIF axes: {cy_axes}, shape: {cy_arr.shape}\n")
        f.write(f"Chosen CyCIF channel: {dapi_index}\n")
        f.write(f"Best variant: {best_variant}\n")
        f.write(f"ECC correlation: {best_cc}\n")
        f.write(f"Warp matrix:\n{best_warp}\n")

    print("\nDone.")
    print(f"Outputs saved in: {OUT_DIR}")
    print("Inspect these first:")
    print("  - cycif_channel_preview.png")
    print("  - overlay_before.png")
    print("  - overlay_after.png")
    print("  - registration_summary.png")


if __name__ == "__main__":
    main()