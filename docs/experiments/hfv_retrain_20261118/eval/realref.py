"""hfv_retrain_20261118: PQ / DICE against the paper-comparable reference (README deviation "PQ 參考遮罩", 2026-11-20).

The reference is CellViT-256 run on the REAL H&E of the same tile (eval/boot_realcv.sh -> S3 outputs/realcv_<pack>.tar),
drawn as the paper's cell_masks were drawn: the CellViT contours filled with PixCell's own
_cellvit_json_to_instance_mask, foreground > 0, written as an L-mode 0/255 PNG. (Check on the paper's 100 CRC33 Fig 2
tiles: this raster of processed_crc33/cellvit equals the paper's orion-crc33 cell_masks pixel for pixel, 100/100.)
PixCell then reads it exactly as it reads any cell_masks channel (_load_gt_instance_mask: > 0, 8-connected labels).
CPU only; nothing is generated and CellViT is not re-run on generated images (their *_cellvit_instances.json exist).
The scored trees are never written: each step works on a NEW mirror (PNG / npy symlinked, every other file copied).

  PYTHONPATH=<PixCell cb62fa7> python realref.py masks REALCV_DIR OUT_ROOT
      REALCV_DIR = realcv_<pack>/ with <id>.json; OUT_ROOT/exp_channels/cell_masks/<id>.png (a minimal orion root:
      only cell_masks, so any step that needs another channel fails instead of silently reading the Lin one)
  ... realref.py fig2 GROUP_DIR REF_ROOT OUT_DIR
      Fig 2 group: same PQ / DICE code path as tools/ablation_a1_a2/metrics_io.py compute (run_cellvit on tiles/<v>/<t>.png,
      _load_gt_instance_mask, _compute_pq, _compute_binary_segmentation_metrics; _mean / _std; merge_metrics) on the
      cache's metric_tile_ids; only pq / pq_std / dice / dice_std change in cache.json, and only the pq / dice columns
      in fig2_per_tile.csv. FUD, LPIPS, HED and the sensitivity block are copied unchanged.
  ... realref.py tree KIND GROUP_DIR REF_ROOT OUT_DIR [JOBS]
      KIND abl / decomp / crc34 / loo: PixCell's own scripts, unchanged, on the mirror with --orion-root REF_ROOT:
        abl     tools/compute_ablation_metrics.py --metrics aji pq dice iou accuracy --device cpu (paired, unpaired)
        decomp  tools/compute_ablation_metrics.py --metrics pq dice --device cpu; src.a2_decomposition.metrics --summarize
        crc34   tools/compute_ablation_metrics.py --metrics pq dice --device cpu; tools/stage3/aggregate_all_only_metrics.py
        loo     tools.stage3.aggregate_subchannel_layout_metrics
      (the metric lists are the cell metrics of the evalscore.py call for that kind; LPIPS / HED / FUD stay as scored).
  ... realref.py line PACK_TAR PACK_NAME REALCV_DIR IDS_FILE OUT_JSON
      Reference line for the secondary (Lin-mask) PQ axis: CellViT on the real H&E scored against the Lin cell_masks
      of the eval pack, per tile, mean with a 95 % percentile bootstrap over tiles (10,000 resamples, seed 42).
"""
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from tools.compute_ablation_metrics import (_cellvit_json_to_instance_mask, _compute_binary_segmentation_metrics,
                                            _compute_pq, _label_binary_instances, _load_gt_instance_mask, run_cellvit)

PIXCELL = Path(os.environ.get("PIXCELL", os.environ.get("PYTHONPATH", "").split(os.pathsep)[0])).resolve()
LINK_SUFFIX = {".png", ".npy", ".pt", ".npz"}


def masks(realcv_dir, out_root):
    d = Path(out_root) / "exp_channels" / "cell_masks"
    if d.exists():
        sys.exit(f"STOP {d} exists (new dirs only)")
    d.mkdir(parents=True)
    js = sorted(Path(realcv_dir).glob("*.json"))
    for j in js:
        inst = _cellvit_json_to_instance_mask(j, image_hw=(256, 256))
        Image.fromarray(((inst > 0) * 255).astype(np.uint8), mode="L").save(d / f"{j.stem}.png")
    print(f"{len(js)} reference masks -> {d}")


def mirror(src, dst):
    src, dst = Path(src).resolve(), Path(dst)
    if dst.exists():
        sys.exit(f"STOP {dst} exists (new dirs only)")
    n_link = n_copy = 0
    for root, dirs, files in os.walk(src):
        r = Path(root)
        o = dst / r.relative_to(src)
        o.mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(f).suffix in LINK_SUFFIX:
                os.symlink(r / f, o / f)
                n_link += 1
            else:
                shutil.copy2(r / f, o / f)
                n_copy += 1
    print(f"mirror {src} -> {dst}: {n_link} linked, {n_copy} copied")


def fig2(group, ref_root, out):
    from tools.ablation_a1_a2.metrics_io import _image_path, _mean, _metric_tile_ids, _std, load_cache, merge_metrics, \
        save_cache
    group, out = Path(group), Path(out)
    mirror(group, out)
    cache = load_cache(out / "cache.json")
    variants = [v for v in ("a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf")
                if v in cache["metrics"]]
    tiles = _metric_tile_ids(cache, out / "tiles", tuple(variants))
    per = {}
    for v in variants:
        pq_v, dice_v = [], []
        for t in tiles:
            pred = run_cellvit(_image_path(out / "tiles", v, t))
            gt = _load_gt_instance_mask(Path(ref_root), t, shape=pred.shape)
            pq_v.append(float(_compute_pq(gt, pred)[2]))
            dice_v.append(float(_compute_binary_segmentation_metrics(gt, pred)[0]))
            per[(v, t)] = (pq_v[-1], dice_v[-1])
        merge_metrics(cache, v, {"pq": _mean(pq_v), "pq_std": _std(pq_v), "dice": _mean(dice_v), "dice_std": _std(dice_v)})
        print(f"{v:20s} n {len(tiles)} PQ {_mean(pq_v):.4f} DICE {_mean(dice_v):.4f}", flush=True)
    cache["pq_reference"] = "CellViT-256 on the real H&E (hfv_retrain_20261118 eval/realref.py)"
    save_cache(cache, out / "cache.json")
    rows = list(csv.DictReader(open(out / "fig2_per_tile.csv")))
    for r in rows:
        if (r["variant"], r["tile"]) not in per:
            sys.exit(f"STOP fig2_per_tile.csv row {r['variant']} {r['tile']} has no recomputed PQ")
        r["pq"], r["dice"] = per[(r["variant"], r["tile"])]
    with open(out / "fig2_per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def px(args, log):
    env = dict(os.environ, PYTHONPATH=str(PIXCELL))
    print("+", " ".join(map(str, args)), flush=True)
    with open(log, "a") as f:
        rc = subprocess.run([sys.executable, *map(str, args)], cwd=PIXCELL, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        sys.exit(f"STOP rc {rc}: {' '.join(map(str, args))} (log {log})")


def tree(kind, group, ref_root, out, jobs):
    group, out, ref = Path(group), Path(out), Path(ref_root).resolve()
    mirror(group, out)
    out = out.resolve()
    log = out / "realref.log"
    cam = "tools/compute_ablation_metrics.py"
    if kind == "abl":
        for mode in ("paired", "unpaired"):
            px([cam, "--cache-dir", out / f"{mode}_ablation" / "ablation_results", "--orion-root", ref, "--metrics", "aji",
                "pq", "dice", "iou", "accuracy", "--device", "cpu", "--jobs", jobs], log)
    elif kind == "decomp":
        M = out / "decomposition_metrics"
        px([cam, "--cache-dir", M, "--orion-root", ref, "--metrics", "pq", "dice", "--device", "cpu", "--jobs", jobs], log)
        px(["-m", "src.a2_decomposition.metrics", "--summarize", "--metrics-root", M, "--fud-json", out / "fud_scores.json",
            "--out-csv", out / "decomposition_summary.csv", "--representative-json", out / "representative_tile.json"], log)
    elif kind == "crc34":
        for mode in ("paired", "unpaired"):
            C = out / f"{mode}_cache"
            px([cam, "--cache-dir", C, "--orion-root", ref, "--metrics", "pq", "dice", "--device", "cpu", "--jobs", jobs], log)
            px(["tools/stage3/aggregate_all_only_metrics.py", "--cache-dir", C, "--label", f"CRC34 {mode}"], log)
    elif kind == "loo":
        px(["-m", "tools.stage3.aggregate_subchannel_layout_metrics", "--out-dir", out, "--orion-root", ref], log)
    else:
        sys.exit(f"STOP unknown kind {kind}")
    (out / "PQ_REFERENCE.txt").write_text("PQ / DICE (and AJI / IoU / accuracy where present) in this tree are against "
                                          "CellViT-256 on the real H&E (eval/realref.py); every other metric as scored.\n")


def line(pack_tar, pack, realcv_dir, ids_file, out_json):
    ids = [l.strip() for l in open(ids_file) if l.strip()]
    want = {f"orion-{pack}/exp_channels/cell_masks/{t}.png": t for t in ids}
    lin = {}
    with tarfile.open(pack_tar) as tf:
        for m in tf:
            if m.name in want:
                lin[want[m.name]] = np.asarray(Image.open(io.BytesIO(tf.extractfile(m).read())))
    miss = [t for t in ids if t not in lin]
    if miss:
        sys.exit(f"STOP {len(miss)} ids without a Lin cell_masks PNG in {pack_tar}, e.g. {miss[:3]}")
    pq, dice = [], []
    for t in ids:
        pred = _cellvit_json_to_instance_mask(Path(realcv_dir) / f"{t}.json", image_hw=lin[t].shape[:2])
        gt = _label_binary_instances(lin[t] > (0.5 if lin[t].max() <= 1.0 else 0))
        pq.append(_compute_pq(gt, pred)[2])
        dice.append(_compute_binary_segmentation_metrics(gt, pred)[0])
    idx = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    res = {"pack": pack, "ids": str(ids_file), "n_tiles": len(ids),
           "what": "CellViT-256 on the real H&E vs the Lin cell_masks (reference line for the Lin-mask PQ axis)"}
    for k, x in (("pq", np.array(pq)), ("dice", np.array(dice))):
        lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
        res[k] = {"mean": float(x.mean()), "std": float(x.std()), "ci95": [float(lo), float(hi)]}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "masks":
        masks(*a)
    elif cmd == "fig2":
        fig2(*a)
    elif cmd == "tree":
        tree(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else "4")
    elif cmd == "line":
        line(*a)
    else:
        sys.exit(__doc__)
