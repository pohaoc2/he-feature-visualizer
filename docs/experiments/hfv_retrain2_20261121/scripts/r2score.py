"""hfv_retrain2 stage 3 (PREREG §9.2, §9.5), Hyak CPU: readouts 1-2 (mask fidelity, appearance) of model N vs model K on
the Fig 2 tiles, and the §5 / §9.5 decision. Written before any N image exists.

  python -I r2score.py zip G WHICH      (WHICH = KR: the K and real-H&E zips, before N exists; N: the N zips + ND.zip)
      G/zips/{N,K,R}_<set>.zip (<tile>.png) for crc36ho (300) and crc33sec (100): N = G/gen/fig2c_<set>/tiles/a1_concat,
      K = G/kfig2/fig2_<set>/tiles/a1_concat (2026-11-19 L4 Fig 2 Concat images), R = real H&E of the R0 evaluation pack
      (G/packs/R0/orion-<set>/he); and G/zips/ND.zip (ND__<w>__<tile>__<seed>.png; every image's sha256 checked against
      G/gen/r2ND/rows.json; 384). Gates: every id present in all three sources; N / R evaluation-pack H&E bytes identical.
  NVIDIA_TF32_OVERRIDE=0 python pxscore.py r2score.py metrics G UNI      (cwd = PixCell cb62fa7 root)
      per tile and model m in {N, K} (CellViT JSON in G/cv/<src>_<set>/<tile>.json, PixCell _cellvit_json_to_instance_mask):
        pq2_m  = _compute_pq(own mask, CellViT(m))   own mask = _load_gt_instance_mask of the model's training-style
                 cell_masks (N: G/packs/N = L1; K: G/packs/R0 = R0), i.e. binarised, connected components
        ref2_m = _compute_pq(own mask, CellViT(real H&E))                          fid_pq_m = pq2_m - ref2_m
        f1_m   = robust.py readout 2 (Hungarian, radius r of hfv_robust_20261120/res2/<set>/info.json, edge rule);
                 f1_R the same for the real H&E                                    fid_f1_m = f1_m - f1_R
        pq1_m  = _compute_pq(CellViT(real H&E), CellViT(m))  (primary PQ, descriptive)
        lpips_m (PixCell _load_lpips / _lpips_pair, alex), hed_m (fig2tiles.py block), n_det_m, count_ratio_m
      FUD per (set, model): PixCell compute_statistics / compute_fid_from_stats; real = stored UNI of the R0 pack
      (features/<t>_uni.npy), generated = UNI-2h (UNI = model folder) with the pxscore.py transform, CPU.
      -> G/score/tiles_<set>.csv, G/score/fud.json
  python -I r2score.py stats G
      tile bootstrap B = 10,000, default_rng(42).integers(0, n, (B, n)), one matrix per set for every quantity (paired);
      95 % percentile interval of the mean. -> G/score/stats_r1.json
  python -I r2score.py decide G APP8_JSON
      §9.5: "N higher" = the four N - K fidelity intervals (fid_pq, fid_f1 x crc36ho, crc33sec) all > 0; row 2 = the
      NP - KP or ND - KD delta interval upper bound < 0; order row 2 -> row 1 (N higher and neither delta interval < 0)
      -> row 3. -> G/score/decision.json
"""
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

SETS = {"crc36ho": ("crc36ho_p300_ids.txt", "/gscratch/cheme/chiu/hfv_crc36_20261118/out"),
        "crc33sec": ("crc33sec_fig2_ids.txt", "/gscratch/cheme/chiu/hfv_lintype_20261118/out")}
INFO = "/gscratch/cheme/chiu/hfv_robust_20261120/res2/{s}/info.json"
B, SEED = 10000, 42


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


def ids_of(G, s):
    return [t.strip() for t in open(Path(G) / "inputs/lists" / SETS[s][0]) if t.strip()]


def src_png(G, src, s, t):
    G = Path(G)
    return {"N": G / f"gen/fig2c_{s}/tiles/a1_concat/{t}.png", "K": G / f"kfig2/fig2_{s}/tiles/a1_concat/{t}.png",
            "R": G / f"packs/R0/orion-{s}/he/{t}.png"}[src]


# ------------------------------------------------------------------------------------------------------------------ zip
def zips(G, which):
    G = Path(G)
    Z = G / "zips"
    Z.mkdir(parents=True, exist_ok=True)
    srcs = {"KR": ("K", "R"), "N": ("N",)}[which]
    rep = {}
    for s in SETS:
        ids = ids_of(G, s)
        for t in ids:
            a, b = (G / f"packs/{m}/orion-{s}/he/{t}.png" for m in ("R0", "N"))
            if a.read_bytes() != b.read_bytes():
                stop(f"{s} {t}: N / R0 pack H&E differ")
        for src in srcs:
            if (Z / f"{src}_{s}.zip").exists():
                stop(f"{src}_{s}.zip exists")
            fs = [src_png(G, src, s, t) for t in ids]
            miss = [f for f in fs if not f.is_file()]
            if miss:
                stop(f"{src} {s}: {len(miss)} missing, e.g. {miss[0]}")
            with zipfile.ZipFile(Z / f"{src}_{s}.zip", "w", zipfile.ZIP_STORED) as z:
                for t, f in zip(ids, fs):
                    z.writestr(f"{t}.png", f.read_bytes())
            rep[f"{src}_{s}"] = len(fs)
    if which == "KR":
        (Z / "zips_KR.json").write_text(json.dumps(rep, indent=1) + "\n")
        print(rep)
        return
    d = G / "gen/r2ND"
    st = json.loads((d / "status.json").read_text())
    if st["state"] != "complete" or st["images"] != 384:
        stop(f"r2ND status {st}")
    items = []
    for r in json.loads((d / "rows.json").read_text()):
        b = (d / r["image"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != r["image_sha256"]:
            stop(f"sha256 {r['image']}")
        items.append((f"ND__{r['window']}__{r['tile']}__{r['seed']}.png", b))
    if len(items) != 384 or len({n for n, _ in items}) != 384:
        stop(f"ND {len(items)}")
    with zipfile.ZipFile(Z / "ND.zip", "w", zipfile.ZIP_STORED) as z:
        for n, b in sorted(items):
            z.writestr(n, b)
    rep["ND"] = 384
    (Z / "zips_N.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(rep)


# -------------------------------------------------------------------------------------------------------------- metrics
def centroids(inst):
    from scipy import ndimage
    ids = np.unique(inst[inst > 0])
    if ids.size == 0:
        return np.zeros((0, 2))
    c = ndimage.center_of_mass(np.ones_like(inst), inst, ids)
    return np.array([(x, y) for y, x in c], dtype=float)


def det_f1(det, lin, r):
    """robust.py det_f1, r passed in"""
    from scipy.optimize import linear_sum_assignment
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
    return dict(f1=(2 * tp / den) if den else 1.0, count_ratio=(int(din.sum()) / n_lin) if n_lin else float("nan"))


def metrics(G, uni):
    import torch
    from PIL import Image
    from tools.ablation_a1_a2.metrics_io import _load_lpips, _lpips_pair
    from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq, _load_gt_instance_mask
    from tools.compute_fid import (ImageFeatureRecord, compute_fid_from_stats, compute_statistics, extract_uni_features,
                                   load_uni_extractor)
    from tools.stage3.hed_utils import masked_mean_std, rgb_to_hed, tissue_mask_from_rgb
    G = Path(G)
    O = G / "score"
    O.mkdir(parents=True)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    lp_model, lp_dev = _load_lpips("cpu")
    inst = lambda src, s, t: _cellvit_json_to_instance_mask(G / f"cv/{src}_{s}/{t}.json", image_hw=(256, 256))
    fud = {}
    extractor = load_uni_extractor(uni_model=Path(uni), device="cpu")
    for s, (_, lin_dir) in SETS.items():
        ids = ids_of(G, s)
        info = json.loads(Path(INFO.format(s=s)).read_text())
        r = float(info["match_radius_px"])
        c = np.load(Path(lin_dir) / "cells_labels.npz")
        lx, ly = c["x"].astype(float), c["y"].astype(float)
        rows = []
        for i, t in enumerate(ids):
            x0, y0 = map(int, t.split("_"))
            m = (lx >= x0 - r) & (lx < x0 + 256 + r) & (ly >= y0 - r) & (ly < y0 + 256 + r)
            lin = np.stack([lx[m] - x0, ly[m] - y0], axis=1)
            pr = inst("R", s, t)
            fr = det_f1(centroids(pr), lin, r)
            row = dict(tile=t, f1_R=fr["f1"], count_ratio_R=fr["count_ratio"], n_det_R=int(len(np.unique(pr)) - 1))
            ref_png = src_png(G, "R", s, t)
            ref_img = Image.open(ref_png).convert("RGB")
            ref_hed = rgb_to_hed(ref_img)
            for mdl, pack in (("N", "N"), ("K", "R0")):
                gt = _load_gt_instance_mask(G / f"packs/{pack}/orion-{s}", t, shape=(256, 256))
                pm = inst(mdl, s, t)
                f = det_f1(centroids(pm), lin, r)
                gen_png = src_png(G, mdl, s, t)
                gen_img = Image.open(gen_png).convert("RGB")
                gen_hed = rgb_to_hed(gen_img)
                tm = tissue_mask_from_rgb(ref_img) | tissue_mask_from_rgb(gen_img)
                hed = 0.0
                for ch in (0, 1):
                    rm, rs = masked_mean_std(ref_hed[..., ch], tm)
                    gm, gs = masked_mean_std(gen_hed[..., ch], tm)
                    hed += abs(gm - rm) + abs(gs - rs)
                pq2, ref2 = _compute_pq(gt, pm)[2], _compute_pq(gt, pr)[2]
                row.update({f"pq2_{mdl}": pq2, f"ref2_{mdl}": ref2, f"fid_pq_{mdl}": pq2 - ref2,
                            f"f1_{mdl}": f["f1"], f"fid_f1_{mdl}": f["f1"] - fr["f1"],
                            f"count_ratio_{mdl}": f["count_ratio"], f"pq1_{mdl}": _compute_pq(pr, pm)[2],
                            f"lpips_{mdl}": float(_lpips_pair(lp_model, lp_dev, ref_png, gen_png)),
                            f"hed_{mdl}": float(hed), f"n_det_{mdl}": int(len(np.unique(pm)) - 1),
                            f"n_gt_{mdl}": int(len(np.unique(gt)) - 1)})
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"{s} {i + 1}/{len(ids)}", flush=True)
        with open(O / f"tiles_{s}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        real = [ImageFeatureRecord(image_path=src_png(G, "R", s, t),
                                   feature_path=G / f"packs/R0/orion-{s}/features/{t}_uni.npy") for t in ids]
        rmu, rsig = compute_statistics(extract_uni_features(real, extractor=None, batch_size=16))
        for mdl in ("N", "K"):
            recs = [ImageFeatureRecord(image_path=src_png(G, mdl, s, t), feature_path=None) for t in ids]
            feats = extract_uni_features(recs, extractor=extractor, batch_size=16)
            np.save(O / f"uni_{mdl}_{s}.npy", np.asarray(feats, np.float32))
            gmu, gsig = compute_statistics(feats)
            fud[f"{mdl}_{s}"] = float(compute_fid_from_stats(rmu, rsig, gmu, gsig))
        print(s, {k: v for k, v in fud.items() if k.endswith(s)}, flush=True)
    (O / "fud.json").write_text(json.dumps(dict(fud=fud, n={s: len(ids_of(G, s)) for s in SETS},
                                                real="stored UNI of the R0 evaluation pack",
                                                generated="UNI-2h, pxscore.py transform, CPU"), indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------------------- stats
Q = ["fid_pq", "fid_f1", "pq2", "ref2", "f1", "pq1", "lpips", "hed", "count_ratio", "n_det"]


def stats(G):
    G = Path(G)
    out = dict(rule="hfv_retrain2 PREREG §9.2 (r2score.py stats)", B=B, seed=SEED, sets={})
    for s in SETS:
        rows = list(csv.DictReader(open(G / f"score/tiles_{s}.csv")))
        n = len(rows)
        idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
        col = lambda k: np.array([float(r[k]) for r in rows])

        def summ(x):
            ok = ~np.isnan(x)
            if ok.all():
                bm = x[idx].mean(1)
            else:   # count_ratio may be NaN on a tile without inner Lin points: nan-mean on the same resamples
                bm = np.nanmean(x[idx], 1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            return dict(mean=float(np.nanmean(x)), ci95=[float(lo), float(hi)], n=int(ok.sum()),
                        excludes_0=bool(lo > 0 or hi < 0))
        d = dict(n_tiles=n, per_model={}, n_minus_k={}, real={})
        for m in ("N", "K"):
            d["per_model"][m] = {q: summ(col(f"{q}_{m}")) for q in Q}
        for q in Q:
            d["n_minus_k"][q] = summ(col(f"{q}_N") - col(f"{q}_K"))
        d["real"] = {q: summ(col(f"{q}_R")) for q in ("f1", "count_ratio", "n_det")}
        out["sets"][s] = d
    out["fud"] = json.loads((G / "score/fud.json").read_text())
    (G / "score/stats_r1.json").write_text(json.dumps(out, indent=1) + "\n")
    for s, d in out["sets"].items():
        for q in ("fid_pq", "fid_f1", "pq1", "lpips", "hed"):
            v = d["n_minus_k"][q]
            print(s, q, "N", round(d["per_model"]["N"][q]["mean"], 4), "K", round(d["per_model"]["K"][q]["mean"], 4),
                  "N-K", round(v["mean"], 4), [round(x, 4) for x in v["ci95"]])
    print("FUD", out["fud"]["fud"])


def decide(G, app8):
    G = Path(G)
    r1 = json.loads((G / "score/stats_r1.json").read_text())
    a8 = json.loads(Path(app8).read_text())
    fid = {f"{q}:{s}": r1["sets"][s]["n_minus_k"][f"fid_{q}"]["ci95"] for s in SETS for q in ("pq", "f1")}
    dl = {k: a8["n_minus_k"][f"{k}:delta"]["ci95"] for k in ("NP-KP", "ND-KD")}
    n_higher = all(ci[0] > 0 for ci in fid.values())
    below = {k: ci[1] < 0 for k, ci in dl.items()}
    if any(below.values()):
        row = "2: a delta N - K interval is entirely below 0 -> keep K, report why"
    elif n_higher:
        row = "1: fidelity N higher on all four intervals and neither delta interval below 0 -> sim2real uses N"
    else:
        row = "3: fidelity not higher (an interval includes 0 or is below) and delta not below 0 -> report; mask " \
              "choice back to the user"
    res = dict(rule="hfv_retrain2 PREREG §5, §9.5 (r2score.py decide)", fidelity_ci95=fid, fidelity_n_higher=n_higher,
               delta_ci95=dl, delta_below_0=below, decision_row=row)
    (G / "score/decision.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else ""
    if m == "zip":
        zips(*sys.argv[2:4])
    elif m == "metrics":
        metrics(*sys.argv[2:4])
    elif m == "stats":
        stats(sys.argv[2])
    elif m == "decide":
        decide(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)
