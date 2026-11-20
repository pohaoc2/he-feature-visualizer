"""Per-tile Fig 2 metrics (LPIPS, HED, PQ, DICE) for the Fig 2 group, written beside cache.json.

  python pxscore.py fig2tiles.py --cache-dir G --orion-root D [--device cuda]      (cwd = PixCell shadow root)
PixCell's metrics_io.py compute keeps only the mean and std of each metric in cache.json. The README text definition of
ΔLPIPS (rule 2: per tile, LPIPS(Vanilla, real H&E) - LPIPS(variant, real H&E)) and its confidence interval need the
per-tile values. This script repeats metrics_io.compute_and_update_metrics' per-tile loop with the same PixCell functions
(_load_lpips, _lpips_pair, the HED block as written there, run_cellvit, _load_gt_instance_mask, _compute_pq,
_compute_binary_segmentation_metrics) on the same tile list (_metric_tile_ids) and writes G/fig2_per_tile.csv. It then
compares the per-tile means with the cache.json means metrics_io wrote and records the largest absolute difference in
G/fig2_per_tile_check.json (a separate GPU run of the same code; reported, not a pass/fail gate).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from tools.ablation_a1_a2.cache_io import load_cache
from tools.ablation_a1_a2.metrics_io import (_compute_binary_segmentation_metrics, _compute_pq, _image_path,
                                             _load_gt_instance_mask, _load_lpips, _lpips_pair, _metric_tile_ids,
                                             run_cellvit)
from tools.stage3.hed_utils import masked_mean_std, rgb_to_hed, tissue_mask_from_rgb

V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
a = argparse.ArgumentParser()
a.add_argument("--cache-dir", required=True, type=Path)
a.add_argument("--orion-root", required=True, type=Path)
a.add_argument("--device", default="cuda")
A = a.parse_args()
cache = load_cache(A.cache_dir / "cache.json")
tile_dir = A.cache_dir / "tiles"
tiles = _metric_tile_ids(cache, tile_dir, tuple(V5))
model, dev = _load_lpips(A.device)
rows = []
for v in V5:
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
        pred = run_cellvit(gen)
        gt = _load_gt_instance_mask(A.orion_root, t, shape=pred.shape)
        _, _, pq = _compute_pq(gt, pred)
        dice, _iou, _acc = _compute_binary_segmentation_metrics(gt, pred)
        rows.append(dict(variant=v, tile=t, lpips=lp, style_hed=float(hed), pq=float(pq), dice=float(dice)))
with open(A.cache_dir / "fig2_per_tile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
import json
diffs = {}
for v in V5:
    m = cache["metrics"][v]
    for k in ("lpips", "style_hed", "pq", "dice"):
        vals = [r[k] for r in rows if r["variant"] == v and r[k] is not None and math.isfinite(r[k])]
        mine, theirs = float(np.mean(vals)), m.get(k)
        diffs[f"{v}/{k}"] = dict(per_tile_mean=mine, cache=theirs, abs_diff=None if theirs is None else abs(mine - theirs))
        print(f"[fig2tiles] {v} {k} per-tile mean {mine:.6f} cache {theirs}", flush=True)
mx = max((d["abs_diff"] for d in diffs.values() if d["abs_diff"] is not None), default=None)
json.dump(dict(n_tiles=len(tiles), max_abs_diff=mx, metrics=diffs), open(A.cache_dir / "fig2_per_tile_check.json", "w"), indent=1)
print(f"[fig2tiles] {len(tiles)} tiles x {len(V5)} variants -> {A.cache_dir / 'fig2_per_tile.csv'}; max abs diff {mx}")
