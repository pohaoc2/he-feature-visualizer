"""paperpq_official_20261122 (PREREG.md §7.2): appearance metrics with the PixCell cb62fa7 Fig 2 functions.

  python pxscore.py pp_appear.py --cache-dir G --orion-root ROOT --ids IDS --variants V [V ...] [--device cuda]
(cwd = the PixCell shadow root; pxscore.py = the hfv_retrain UNI-2h transform patch + the NVIDIA_TF32_OVERRIDE=0 check)
Per tile: LPIPS(generated, real H&E of the same tile) with metrics_io._load_lpips / _lpips_pair, and style_hed with the HED
block of metrics_io.compute_and_update_metrics (as hfv_retrain eval/fig2tiles.py repeats it). Per variant: FUD with
metrics_io._compute_fud_by_variant (real = ROOT/features/<t>_uni.npy stored features, generated = UNI-2h recomputed).
Writes G/appear_per_tile.csv and G/fud.json.
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from tools.ablation_a1_a2.metrics_io import _compute_fud_by_variant, _image_path, _load_lpips, _lpips_pair
from tools.stage3.hed_utils import masked_mean_std, rgb_to_hed, tissue_mask_from_rgb

a = argparse.ArgumentParser()
a.add_argument("--cache-dir", required=True, type=Path)
a.add_argument("--orion-root", required=True, type=Path)
a.add_argument("--ids", required=True, type=Path)
a.add_argument("--variants", nargs="+", required=True)
a.add_argument("--device", default="cuda")
A = a.parse_args()
tiles = [t.strip() for t in open(A.ids) if t.strip()]
tile_dir = A.cache_dir / "tiles"
model, dev = _load_lpips(A.device)
rows = []
for v in A.variants:
    for t in tiles:
        gen, ref = _image_path(tile_dir, v, t), A.orion_root / "he" / f"{t}.png"
        lp = _lpips_pair(model, dev, ref, gen)
        ref_img, gen_img = Image.open(ref).convert("RGB"), Image.open(gen).convert("RGB")
        ref_hed, gen_hed = rgb_to_hed(ref_img), rgb_to_hed(gen_img)
        tm = tissue_mask_from_rgb(ref_img) | tissue_mask_from_rgb(gen_img)
        hed = 0.0
        for ch in (0, 1):
            rm, rs = masked_mean_std(ref_hed[..., ch], tm)
            gm, gs = masked_mean_std(gen_hed[..., ch], tm)
            hed += abs(gm - rm) + abs(gs - rs)
        rows.append(dict(variant=v, tile=t, lpips=float(lp), style_hed=float(hed)))
with open(A.cache_dir / "appear_per_tile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
fud = _compute_fud_by_variant(tile_dir=tile_dir, tile_ids=tiles, variants=tuple(A.variants), orion_root=A.orion_root,
                              uni_model=Path("pretrained_models/uni-2h"), device=A.device, batch_size=16)
json.dump({k: float(v) for k, v in fud.items()}, open(A.cache_dir / "fud.json", "w"), indent=1)
print(f"[pp_appear] {len(tiles)} tiles x {len(A.variants)} variants; FUD written", flush=True)
