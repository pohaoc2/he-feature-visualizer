"""hfv_retrain_20261118: two PQ robustness readouts (README deviation "PQ 的穩健性檢查", 2026-11-20). CPU only.

Readout 1, StarDist (symmetric): StarDist 2D '2D_versatile_he' (Schmidt 2018; Weigert 2020), the model's own default
  thresholds (read from the model and written to sd_model.json), input normalised as in the StarDist examples
  (csbdeep normalize, percentiles 1 and 99.8, over all three colour channels together). Run on the real H&E and on every
  generated image; PQ / DICE with PixCell's _compute_pq / _compute_binary_segmentation_metrics, reference = StarDist on
  the real H&E of the same tile (the reference tile of PixCell's compute_cell_metrics: the manifest tile_id, i.e. the
  layout tile in unpaired mode). Both sides use StarDist's instance labels as they are (no binarise + relabel).
Readout 2, detection F1 vs the Lin nuclei (IoU-free): one point per Lin cell (cells_labels.npz x, y: the Lin
  quantification centroid, mask pixel grid = tile grid; tile id = "x0_y0"); one point per CellViT nucleus (centre of mass
  of PixCell's _cellvit_json_to_instance_mask labels of the *_cellvit_instances.json sidecar, or of the real-H&E realcv
  JSON). One-to-one matching (Hungarian, scipy linear_sum_assignment) of points closer than the match radius r.
  r = median over the set's Lin cells (centroid inside the set's evaluated tiles) of the equivalent radius sqrt(area/pi)
  of the Lin nucleus (cells_nuclei.npz loose_px: the Lin cell mask eroded as for the cell_masks channel), in pixels.
  Edge rule (r only, no other constant): Lin points up to r outside the tile take part in the matching; TP, FN and FP
  are counted only for points at least r inside the tile (TP / FN by the Lin point, FP by the detection).
  F1 = 2 TP / (2 TP + FP + FN); count ratio = detections / Lin points in that inner region.

  python robust.py sdlist SET SCORE_ROOT REAL_ROOT OUT_LIST      (image list for StarDist: src \\t dst)
  python robust.py stardist LIST SD_ROOT TASK NTASK             (sdenv; SD_ROOT/sd_<task>.npz, uint16 labels)
  python robust.py metrics SET SCORE_ROOT REAL_ROOT REALCV_DIR SD_ROOT LIN_DIR OUT_DIR NPROC   (PixCell on PYTHONPATH)
  python robust.py stats OUT_DIR                                (bootstrap, the evalstats.py definitions)
SCORE_ROOT holds the scored groups <kind>_<SET>/ (score_<group>.tar extracted); REAL_ROOT/<SET>/<id>.png the real H&E.
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

KINDS = ("fig2", "abl", "loo", "decomp")
V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
B, SEED = 10000, 42


# ---------------------------------------------------------------- image enumeration (no PixCell needed for fig2 / loo)
def images(kind, gdir):
    """yield (mode, tile_dir_or_tile, condition, ref_tile, image_path)"""
    gdir = Path(gdir)
    if kind == "fig2":
        tiles = sorted({r["tile"] for r in csv.DictReader(open(gdir / "fig2_per_tile.csv")) if r["variant"] == "a2_off_shelf"})
        for v in V5:
            for t in tiles:
                yield "fig2", t, v, t, gdir / "tiles" / v / f"{t}.png"
    elif kind == "loo":
        from tools.stage3.aggregate_subchannel_layout_metrics import SUBCHANNELS
        for td in sorted(p for p in gdir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            if not (td / "all_baseline.png").is_file():
                continue
            yield "loo", td.name, "all_baseline", td.name, td / "all_baseline.png"
            for s in SUBCHANNELS:
                if (td / s / "generated_he.png").is_file():
                    yield "loo", td.name, s, td.name, td / s / "generated_he.png"
    else:
        from tools.compute_ablation_metrics import _iter_condition_images, _tile_id_from_manifest
        roots = ([("paired", gdir / "paired_ablation" / "ablation_results"),
                  ("unpaired", gdir / "unpaired_ablation" / "ablation_results")] if kind == "abl"
                 else [("decomp", gdir / "decomposition_metrics")])
        for mode, r in roots:
            for td in sorted(p for p in r.iterdir() if (p / "manifest.json").is_file()):
                ref = _tile_id_from_manifest(td)
                for cond, ip in _iter_condition_images(td).items():
                    yield mode, td.name, cond, ref, ip


def groups(score_root, set_):
    for k in KINDS:
        g = Path(score_root) / f"{k}_{set_}" / f"{k}_{set_}"
        if g.is_dir():
            yield k, g


# ---------------------------------------------------------------- StarDist
def image_jobs(set_, score_root):
    score_root = Path(score_root).resolve()
    for k, g in groups(score_root, set_):
        for mode, tile, cond, ref, ip in images(k, g):
            yield k, mode, tile, cond, ref, str(ip), str(Path("gen") / Path(ip).relative_to(score_root))


def sdlist(set_, score_root, real_root, out_list):
    """every image the metrics step reads: the real H&E of the reference tiles + every evaluated generated image"""
    jobs = list(image_jobs(set_, score_root))
    real_root = Path(real_root).resolve()
    refs = sorted({j[4] for j in jobs})
    rows = [(real_root / set_ / f"{t}.png", f"real/{set_}/{t}.png") for t in refs]
    miss = [str(a) for a, _ in rows if not a.is_file()] + [j[5] for j in jobs if not Path(j[5]).is_file()]
    if miss:
        sys.exit(f"STOP {len(miss)} images missing, e.g. {miss[:3]}")
    rows += [(j[5], j[6]) for j in jobs]
    with open(out_list, "w") as f:
        for a, b in rows:
            f.write(f"{a}\t{b}\n")
    print(f"{set_}: {len(refs)} real + {len(jobs)} generated images -> {out_list}")


def stardist(lst, sd_root, task, ntask):
    import tensorflow as tf
    from csbdeep.utils import normalize
    from PIL import Image
    from stardist.models import StarDist2D
    n = int(os.environ.get("SD_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    task, ntask = int(task), int(ntask)
    rows = [l.rstrip("\n").split("\t") for l in open(lst) if l.strip()]
    mine = rows[task::ntask]
    import fcntl
    # serialise the first-use download / unpack of the pretrained weights across parallel processes and jobs
    with open(os.path.join(os.environ.get("KERAS_HOME", str(Path.home())), "sd_load.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        model = StarDist2D.from_pretrained("2D_versatile_he")
        fcntl.flock(lk, fcntl.LOCK_UN)
    th = dict(prob=float(model.thresholds.prob), nms=float(model.thresholds.nms))
    sd_root = Path(sd_root)
    if task == 0 and not (sd_root / "sd_model.json").exists():
        import stardist
        json.dump(dict(model="2D_versatile_he", thresholds=th, normalize=[1, 99.8, "axis (0, 1, 2)"],
                       stardist=stardist.__version__, tensorflow=tf.__version__), open(sd_root / "sd_model.json", "w"), indent=1)
    out = sd_root / f"sd_{task:04d}.npz"
    if out.exists():
        sys.exit(f"STOP {out} exists (new files only)")
    labs, rels = {}, []
    for src, rel in mine:
        img = np.asarray(Image.open(src).convert("RGB"))
        lab, _ = model.predict_instances(normalize(img, 1, 99.8, axis=(0, 1, 2)))
        if lab.max() >= 2 ** 16:
            sys.exit(f"STOP {src}: {lab.max()} labels do not fit uint16")
        labs[f"i{len(rels)}"] = lab.astype(np.uint16)
        rels.append(rel)
    # one file per task (gscratch file creation is slow): labels under i<k>, rels[k] = the image's relative name
    np.savez_compressed(out, rels=np.array(rels), **labs)
    print(f"task {task}/{ntask}: {len(rels)} images, thresholds {th}", flush=True)


# ---------------------------------------------------------------- metrics
G = {}


def sd_index(sd_root):
    idx = {}
    for f in sorted(Path(sd_root).glob("sd_*.npz")):
        for k, rel in enumerate(np.load(f)["rels"]):
            idx[str(rel)] = (str(f), f"i{k}")
    return idx


_NPZ = {}


def sd_labels(rel):
    f, k = G["sd_idx"][str(rel)]
    if f not in _NPZ:
        _NPZ[f] = np.load(f)
    return _NPZ[f][k].astype(np.int32)


def centroids(inst):
    from scipy import ndimage
    ids = np.unique(inst[inst > 0])
    if ids.size == 0:
        return np.zeros((0, 2))
    c = ndimage.center_of_mass(np.ones_like(inst), inst, ids)
    return np.array([(x, y) for y, x in c], dtype=float)   # (x, y) in tile pixels


def lin_points(t):
    x0, y0 = map(int, t.split("_"))
    r = G["r"]
    x, y = G["lin_x"], G["lin_y"]
    m = (x >= x0 - r) & (x < x0 + 256 + r) & (y >= y0 - r) & (y < y0 + 256 + r)
    return np.stack([x[m] - x0, y[m] - y0], axis=1)


def det_f1(det, lin):
    from scipy.optimize import linear_sum_assignment
    r = G["r"]
    inner = lambda p: (p[:, 0] >= r) & (p[:, 0] < 256 - r) & (p[:, 1] >= r) & (p[:, 1] < 256 - r)
    din, lin_in = inner(det), inner(lin)
    md = np.zeros(len(det), bool)
    ml = np.zeros(len(lin), bool)
    if len(det) and len(lin):
        d = np.sqrt(((det[:, None, :] - lin[None, :, :]) ** 2).sum(-1))
        cost = np.where(d <= r, d, 1e6)
        ri, ci = linear_sum_assignment(cost)
        ok = d[ri, ci] <= r
        md[ri[ok]] = True
        ml[ci[ok]] = True
    tp = int((ml & lin_in).sum())
    fn = int((~ml & lin_in).sum())
    fp = int((~md & din).sum())
    den = 2 * tp + fp + fn
    n_lin = int(lin_in.sum())
    return dict(f1=(2 * tp / den) if den else 1.0, count_ratio=(int(din.sum()) / n_lin) if n_lin else float("nan"),
                tp=tp, fp=fp, fn=fn, n_det=int(din.sum()), n_lin=n_lin)


def one(job):
    from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_binary_segmentation_metrics, \
        _compute_pq
    kind, mode, tile, cond, ref, ip, rel = job
    ip = Path(ip)
    side = ip.with_name(ip.stem + "_cellvit_instances.json")
    row = dict(kind=kind, mode=mode, tile=tile, cond=cond, ref=ref)
    sd_ref = sd_labels(f"real/{G['set']}/{ref}.png")
    pred = sd_labels(rel)
    row["sd_pq"] = _compute_pq(sd_ref, pred)[2]
    row["sd_dice"] = _compute_binary_segmentation_metrics(sd_ref, pred)[0]
    cv = _cellvit_json_to_instance_mask(side, image_hw=pred.shape)
    row.update(det_f1(centroids(cv), lin_points(ref)))
    return row


def real_row(t):
    from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq, \
        _compute_binary_segmentation_metrics
    sd = sd_labels(f"real/{G['set']}/{t}.png")
    cv = _cellvit_json_to_instance_mask(G["realcv"] / f"{t}.json", image_hw=sd.shape)
    lin = lin_points(t)
    a = det_f1(centroids(cv), lin)
    b = det_f1(centroids(sd), lin)
    return dict(tile=t, cv_f1=a["f1"], cv_count_ratio=a["count_ratio"], sd_f1=b["f1"], sd_count_ratio=b["count_ratio"],
                n_lin=a["n_lin"], sd_vs_cv_pq=_compute_pq(cv, sd)[2], sd_vs_cv_dice=_compute_binary_segmentation_metrics(cv, sd)[0])


def metrics(set_, score_root, real_root, realcv, sd_root, lin_dir, out_dir, nproc):
    import multiprocessing as mp
    out = Path(out_dir)
    if out.exists():
        sys.exit(f"STOP {out} exists (new dirs only)")
    out.mkdir(parents=True)
    jobs = list(image_jobs(set_, score_root))
    refs = sorted({j[4] for j in jobs})
    c = np.load(Path(lin_dir) / "cells_labels.npz")
    nu = np.load(Path(lin_dir) / "cells_nuclei.npz")
    if not np.array_equal(c["cell_id"], nu["cell_id"]):
        sys.exit("STOP cells_labels / cells_nuclei cell_id order differs")
    x, y = c["x"].astype(float), c["y"].astype(float)
    inside = np.zeros(len(x), bool)
    for t in refs:
        x0, y0 = map(int, t.split("_"))
        inside |= (x >= x0) & (x < x0 + 256) & (y >= y0) & (y < y0 + 256)
    req = np.sqrt(nu["loose_px"][inside].astype(float) / math.pi)
    r = float(np.median(req))
    G.update(set=set_, sd_idx=sd_index(sd_root), realcv=Path(realcv), lin_x=x, lin_y=y, r=r)
    miss = [j[6] for j in jobs if j[6] not in G["sd_idx"]] + [t for t in refs if f"real/{set_}/{t}.png" not in G["sd_idx"]]
    if miss:
        sys.exit(f"STOP {len(miss)} images without StarDist labels, e.g. {miss[:3]}")
    info = dict(set=set_, n_images=len(jobs), n_ref_tiles=len(refs), n_lin_cells_in_tiles=int(inside.sum()),
                match_radius_px=r, radius_rule="median sqrt(loose_px / pi) of the Lin cells with centroid inside the "
                                              "evaluated tiles", lin_dir=str(lin_dir))
    json.dump(info, open(out / "info.json", "w"), indent=1)
    print(json.dumps(info), flush=True)
    with mp.get_context("fork").Pool(int(nproc)) as pool:
        real = pool.map(real_row, refs, chunksize=8)
        rows = pool.map(one, jobs, chunksize=32)
    for name, rr in (("real.csv", real), ("images.csv", rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    print(f"{len(real)} real tiles, {len(rows)} images -> {out}", flush=True)


# ---------------------------------------------------------------- stats (definitions of eval/evalstats.py)
def boot(arrs):
    if len({len(x) for x in arrs.values()}) > 1:   # unequal tile counts: separate resamples per quantity
        return {k: boot({k: x})[k] for k, x in arrs.items()}
    n = len(next(iter(arrs.values())))
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    res = {}
    for k, x in arrs.items():
        x = np.asarray(x, dtype=np.float64)
        lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
        res[k] = dict(mean=float(x.mean()), ci95=[float(lo), float(hi)], n=int(n), excludes_0=bool(lo > 0 or hi < 0))
    return res


def stats(out_dir):
    from tools.ablation_report.data import summarize_added_group_effect_stats
    out = Path(out_dir)
    rows = list(csv.DictReader(open(out / "images.csv")))
    real = list(csv.DictReader(open(out / "real.csv")))
    M = ("sd_pq", "sd_dice", "f1", "count_ratio")
    val = lambda r, m: float(r[m])
    res = dict(info=json.load(open(out / "info.json")))
    res["real"] = boot({k: [float(r[k]) for r in real if not math.isnan(float(r[k]))]
                        for k in ("cv_f1", "sd_f1", "sd_vs_cv_pq", "sd_vs_cv_dice")})
    res["real"].update({k: boot({k: [float(r[k]) for r in real if not math.isnan(float(r[k]))]})[k]
                        for k in ("cv_count_ratio", "sd_count_ratio")})
    by = {}
    for r in rows:
        by.setdefault((r["kind"], r["mode"]), {}).setdefault(r["tile"], {})[r["cond"]] = r
    if ("fig2", "fig2") in by:
        T = by[("fig2", "fig2")]
        tiles = sorted(T)
        f = dict(n_tiles=len(tiles), means={v: {m: float(np.nanmean([val(T[t][v], m) for t in tiles])) for m in M} for v in V5})
        arr = {f"{v}/{m}": [val(T[t][v], m) - val(T[t]["a2_off_shelf"], m) for t in tiles] for v in V5[:4] for m in M
               if m != "count_ratio" or all(not math.isnan(val(T[t][v], m)) for t in tiles)}
        f["vs_vanilla"] = boot(arr)
        f["ranking"] = {m: sorted(V5, key=lambda v: -f["means"][v][m]) for m in M}
        res["fig2"] = f
    for mode in ("paired", "unpaired"):
        if ("abl", mode) not in by:
            continue
        T = by[("abl", mode)]
        per = {}
        # PixCell's added-effect definition; its METRIC_SPEC table knows "pq" / "dice" (higher is better), so each readout
        # is passed under that key (count ratio has no better direction and is not summarised here)
        for t, conds in T.items():
            for out_key, src in (("sd_pq", "sd_pq"), ("sd_dice", "sd_dice"), ("f1", "f1")):
                tm = {c: {"pq": val(r, src)} for c, r in conds.items() if not math.isnan(val(r, src))}
                for g, mm in summarize_added_group_effect_stats(tm, ["pq"]).items():
                    per.setdefault(f"{g}/{out_key}", []).append(mm["pq"][0])
        res[f"abl_{mode}"] = dict(n_tiles=len(T), added_effect=boot(per))
    if ("loo", "loo") in by:
        T = by[("loo", "loo")]
        subs = sorted({c for conds in T.values() for c in conds if c != "all_baseline"})
        drop = {}
        for s in subs:
            for m in ("sd_pq", "f1"):
                drop[f"{s}/{m}"] = [val(c["all_baseline"], m) - val(c[s], m) for c in T.values() if s in c]
        res["loo"] = dict(n_tiles=len(T), drop=boot(drop))
    if ("decomp", "decomp") in by:
        T = by[("decomp", "decomp")]
        modes = ["uni_plus_tme", "uni_only", "tme_only", "neither"]
        tl = [t for t, c in T.items() if all(m in c for m in modes)]
        res["decomp"] = dict(n_tiles=len(tl),
                             modes=boot({f"{md}/{m}": [val(T[t][md], m) for t in tl] for md in modes for m in ("sd_pq", "f1")}),
                             vs_full=boot({f"{md}/{m}": [val(T[t][md], m) - val(T[t]["uni_plus_tme"], m) for t in tl]
                                           for md in modes[1:] for m in ("sd_pq", "f1")}))
    json.dump(res, open(out / "stats_robust.json", "w"), indent=1)
    print(f"wrote {out / 'stats_robust.json'}")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    {"sdlist": sdlist, "stardist": stardist, "metrics": metrics, "stats": stats}.get(cmd, lambda *_: sys.exit(__doc__))(*a)
