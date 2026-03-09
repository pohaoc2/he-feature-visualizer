"""Visualize H&E + multiplex channel overlay for a set of patches.

Usage
-----
python -m tools.viz_patch_overlay --processed processed_wd_1024/ --out overlay.png
python -m tools.viz_patch_overlay --processed processed_wd_1024/ --n 4 --seed 42
python -m tools.viz_patch_overlay --processed processed_wd_1024/ --patches 20736_512 20480_512 --channels 0 1
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_patch(processed: Path, patch_id: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (he_rgb uint8, multiplex float32 CHW or None)."""
    x, y = patch_id.split("_")
    he = np.array(Image.open(processed / "he" / f"{x}_{y}.png"))
    mx_path = processed / "multiplex" / f"{x}_{y}.npy"
    mx = np.load(mx_path).astype(np.float32) if mx_path.exists() else None
    return he, mx


def norm_channel(arr: np.ndarray, plo: float = 2.0, phi: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(arr, [plo, phi])
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def make_overlay(he: np.ndarray, dna: np.ndarray) -> np.ndarray:
    """Blend H&E (gray) with DNA channel (green tint)."""
    gray = he.mean(axis=2) / 255.0
    overlay = np.stack([gray, gray + dna * 0.6, gray], axis=2)
    return np.clip(overlay, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="H&E + multiplex patch overlay grid")
    ap.add_argument("--processed", required=True, help="Path to processed output dir")
    ap.add_argument(
        "--patches",
        nargs="*",
        default=None,
        help="Patch IDs to visualize (e.g. 20736_512). Default: random sample.",
    )
    ap.add_argument("--n", type=int, default=4, help="Number of random patches (if --patches not given)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--channels",
        nargs="*",
        type=int,
        default=[0],
        help="Multiplex channel indices to show (default: 0 = DNA)",
    )
    ap.add_argument("--out", default=None, help="Output PNG path (default: <processed>/patch_overlay.png)")
    args = ap.parse_args()

    processed = Path(args.processed)
    out_path = Path(args.out) if args.out else processed / "patch_overlay.png"

    # Resolve patch list
    with open(processed / "index.json") as f:
        index = json.load(f)
    all_patches = [f"{e['x0']}_{e['y0']}" for e in index["patches"] if e.get("has_multiplex")]

    if args.patches:
        patch_ids = args.patches
    else:
        rng = random.Random(args.seed)
        patch_ids = rng.sample(all_patches, min(args.n, len(all_patches)))

    channels = args.channels
    n_ch = len(channels)
    # columns: HE | ch_overlay... | (per channel: raw + overlay)
    n_cols = 1 + n_ch * 2  # HE + (raw + overlay) per channel
    n_rows = len(patch_ids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Load channel names from index if available
    ch_names = index.get("channels", [f"ch{i}" for i in range(64)])

    for row, pid in enumerate(patch_ids):
        he, mx = load_patch(processed, pid)

        axes[row, 0].imshow(he)
        axes[row, 0].set_title(f"H&E  {pid}", fontsize=8)
        axes[row, 0].axis("off")

        for col_offset, ch_idx in enumerate(channels):
            col_raw = 1 + col_offset * 2
            col_ov = col_raw + 1

            if mx is not None and ch_idx < mx.shape[0]:
                ch_data = norm_channel(mx[ch_idx])
                ch_label = ch_names[ch_idx] if ch_idx < len(ch_names) else f"ch{ch_idx}"

                axes[row, col_raw].imshow(ch_data, cmap="hot", vmin=0, vmax=1)
                axes[row, col_raw].set_title(f"MX ch{ch_idx} ({ch_label})", fontsize=8)
                axes[row, col_raw].axis("off")

                axes[row, col_ov].imshow(make_overlay(he, ch_data))
                axes[row, col_ov].set_title(f"H&E + {ch_label} overlay", fontsize=8)
                axes[row, col_ov].axis("off")
            else:
                axes[row, col_raw].axis("off")
                axes[row, col_ov].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
