"""paperpq_official_20261122 (PREREG.md §7.1): mask-following readouts per tile, generator and pipeline.

  python pp_score.py D NPROC
Per (root, variant, tile), with root in P_run, P_off, R_run, R_off (pipeline = run|off from the root name):
  reference (paper procedure) = PixCell _load_gt_instance_mask(root, tile): 8-connected components of the input mask PNG
  pred      = PixCell _cellvit_json_to_instance_mask of CellViT on the generated image, same post-processing as the pipeline
  pq        = _compute_pq(reference, pred)[2]          (paper-style PQ; primary)
  pq_inst   = official pipeline: reference = the official instance map itself (real-H&E JSON filled, not binarised)
  dice      = _compute_binary_segmentation_metrics(reference, pred)[0]
  pq_x      = pq with the other post-processing on the generated image (descriptive)
  a, b, c   = truenuc pix(hematoxylin(generated), input mask > 0, t)  (hfv stages.lin_typing.compare.hematoxylin, float64)
  n_ref, n_pred
Real-H&E reference line: the same with the generated image replaced by the real H&E (pred from cellvit_official_20261122
full/cv/F2_P_gt (official) and full_runner/cv/F2_P_gt (runner)); one row per pipeline (P roots; real H&E and masks are
identical for the R roots, gated in pp_prep.py).
Writes D/score/per_tile.csv. Generated images are read from D/gen/<root>.tar.
"""
import csv
import io
import json
import multiprocessing as mp
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

D = Path(sys.argv[1])
NP = int(sys.argv[2])
CVO = Path("/gscratch/cheme/chiu/cellvit_official_20261122")
T = 0.02243768021416608
ROOTS = {"P_run": ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"],
         "P_off": ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"],
         "R_run": ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel"],
         "R_off": ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel"]}
PATHS = {"run": "runner", "off": "official"}
OTHER = {"runner": "official", "official": "runner"}
REAL_CV = {"official": CVO / "full/cv/F2_P_gt", "runner": CVO / "full_runner/cv/F2_P_gt"}
ids = [t.strip() for t in open(D / "prep/ids100.txt") if t.strip()]


def pix(hod, fg, t):   # mask_truenuc_20261122 truenuc.py pix(), unchanged (a, b, c only)
    pos = hod >= t
    nf, npos = int(fg.sum()), int(pos.sum())
    both = int((fg & pos).sum())
    a = both / nf if nf else np.nan
    b = both / npos if npos else np.nan
    c = float(hod[fg].mean() - hod[~fg].mean()) if 0 < nf < fg.size else np.nan
    return dict(a=a, b=b, c=c)


def score(job):
    from stages.lin_typing import compare as CP
    from tools.compute_ablation_metrics import (_cellvit_json_to_instance_mask, _compute_binary_segmentation_metrics,
                                                _compute_pq, _load_gt_instance_mask)
    root, variant, t, img_bytes, pp = job
    rdir = D / "roots" / root
    ref = _load_gt_instance_mask(rdir, t)
    fg = ref > 0
    if variant == "real":
        pred = _cellvit_json_to_instance_mask(REAL_CV[pp] / f"{t}.json", image_hw=(256, 256))
        predx = _cellvit_json_to_instance_mask(REAL_CV[OTHER[pp]] / f"{t}.json", image_hw=(256, 256))
    else:
        pred = _cellvit_json_to_instance_mask(D / "cv" / pp / f"{root}__{variant}" / f"{t}.json", image_hw=(256, 256))
        predx = _cellvit_json_to_instance_mask(D / "cv" / OTHER[pp] / f"{root}__{variant}" / f"{t}.json",
                                               image_hw=(256, 256))
    ref_inst = (_cellvit_json_to_instance_mask(REAL_CV["official"] / f"{t}.json", image_hw=(256, 256))
                if pp == "official" else ref)
    img = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    hod = CP.hematoxylin(img).astype(np.float64)
    r = dict(root=root, pipeline=root.split("_")[1], variant=variant, tile=t,
             pq=float(_compute_pq(ref, pred)[2]), pq_inst=float(_compute_pq(ref_inst, pred)[2]),
             dice=float(_compute_binary_segmentation_metrics(ref, pred)[0]), pq_x=float(_compute_pq(ref, predx)[2]),
             n_ref=int(len(np.unique(ref)) - 1), n_pred=int(len(np.unique(pred)) - 1))
    r.update(pix(hod, fg, T))
    return r


def jobs():
    with __import__("zipfile").ZipFile(CVO / "zips/F2_P_gt.zip") as z:
        real = {t: z.read(f"{t}.png") for t in ids}
    for root in ("P_run", "P_off"):
        for t in ids:
            yield root, "real", t, real[t], PATHS[root.split("_")[1]]
    for root, vs in ROOTS.items():
        with tarfile.open(D / "gen" / f"{root}.tar") as tf:
            mem = {m.name: m for m in tf if m.isfile()}
            for v in vs:
                for t in ids:
                    yield root, v, t, tf.extractfile(mem[f"out/tiles/{v}/{t}.png"]).read(), PATHS[root.split("_")[1]]


if __name__ == "__main__":
    out = D / "score"
    if out.exists():
        sys.exit(f"STOP {out} exists")
    out.mkdir()
    with mp.get_context("fork").Pool(NP) as pool:
        rows = list(pool.imap(score, jobs(), chunksize=8))
    n_exp = 2 * len(ids) + sum(len(v) for v in ROOTS.values()) * len(ids)
    if len(rows) != n_exp:
        sys.exit(f"STOP {len(rows)} rows, expected {n_exp}")
    with open(out / "per_tile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(rows=len(rows), t=T), open(out / "DONE", "w"))
    print(f"[pp_score] {len(rows)} rows -> {out / 'per_tile.csv'}", flush=True)
