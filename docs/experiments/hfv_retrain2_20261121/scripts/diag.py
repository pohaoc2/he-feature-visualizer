"""hfv_retrain2_20261121 stage 1 diagnosis (PREREG.md §2). CPU only; reuses existing real-H&E segmentations.

  python diag.py tiles SET MASK_TIF LIN_DIR REALHE_DIR REALCV_DIR SD_DIR PACK_DIR R_PX OUT_DIR NPROC
      REALHE_DIR/<tile>.png real H&E; REALCV_DIR/<tile>.json CellViT on it; SD_DIR/sd_*.npz StarDist (rel real/SET/<tile>.png);
      PACK_DIR/exp_channels/cell_masks/<tile>.png the training-style mask (R0); R_PX = robust.py match radius of the set.
      -> OUT_DIR/{tiles.csv, lin_scores.npz, det_unmatched.csv, sample_<tile>.npz}
  python diag.py stats OUT_ROOT FIG_DIR     (both sets: Youden t, Rogan-Gladen, cross PQ, rule, contact sheet)

PQ = PixCell cb62fa7 tools.compute_ablation_metrics._compute_pq (PixCell on PYTHONPATH). Every candidate mask is
binarised and relabelled with 8-connectivity exactly as PixCell's _label_binary_instances reads cell_masks.
New candidates are drawn as instances with a separation line: a pixel of label a is cleared when an 8-neighbour carries
a lower nonzero label, so no two labels stay 8-adjacent and binarise + relabel keeps one component per object.
"""
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

TILE, M = 256, 8
POP = ("healthy", "cancer", "immune")
B, SEED = 10000, 42
G = {}


# ------------------------------------------------------------------------------------------------ helpers
def separate(inst):
    """Clear pixels of label a that touch (8-neighbourhood) a lower nonzero label."""
    import cv2
    f = inst.astype(np.float32)
    big = np.float32(2 ** 24)
    lo = cv2.erode(np.where(f > 0, f, big), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=float(big))
    out = inst.copy()
    out[(inst > 0) & (lo < f)] = 0
    return out


def relabel(binary):
    from scipy import ndimage
    lab, _ = ndimage.label(np.asarray(binary, bool), structure=np.ones((3, 3), np.uint8))
    return lab.astype(np.int32)


def centroids(inst):
    from scipy import ndimage
    ids = np.unique(inst)
    ids = ids[ids > 0]
    if not len(ids):
        return ids, np.zeros((0, 2))
    c = ndimage.center_of_mass(np.ones_like(inst), inst, ids)
    return ids, np.array([(cx, cy) for cy, cx in c], float)


def match(det, lin, r):
    """robust.py det_f1 matching: Hungarian on centroid distance <= r. -> (matched_det, matched_lin) bool arrays."""
    from scipy.optimize import linear_sum_assignment
    md = np.zeros(len(det), bool)
    ml = np.zeros(len(lin), bool)
    if len(det) and len(lin):
        d = np.sqrt(((det[:, None, :] - lin[None, :, :]) ** 2).sum(-1))
        ri, ci = linear_sum_assignment(np.where(d <= r, d, 1e6))
        ok = d[ri, ci] <= r
        md[ri[ok]] = True
        ml[ci[ok]] = True
    return md, ml


def inner(p, r):
    return (p[:, 0] >= r) & (p[:, 0] < TILE - r) & (p[:, 1] >= r) & (p[:, 1] < TILE - r)


# ------------------------------------------------------------------------------------------------ per tile
def init(set_, tif, lin_dir, realhe, realcv, sd_dir, pack, r):
    import tifffile
    sys.path.insert(0, os.environ["HFV"])
    G.update(set=set_, realhe=Path(realhe), realcv=Path(realcv), pack=Path(pack), r=float(r))
    c = np.load(Path(lin_dir) / "cells_labels.npz")
    o = np.argsort(c["x"], kind="stable")
    G.update(cid=c["cell_id"][o], x=c["x"][o], y=c["y"][o], ty=c["ty"][o].astype(int), st=c["st"][o].astype(int))
    G["lab"] = tifffile.imread(tif)
    sd = {}
    for f in sorted(Path(sd_dir).glob("sd_*.npz")):
        z = np.load(f)
        for k, rel in enumerate(z["rels"]):
            if rel.startswith(f"real/{set_}/"):
                sd[Path(rel).stem] = z[f"i{k}"].astype(np.int32)
    G["sd"] = sd


def window(x0, y0):
    """Lin labels of the tile plus an M-px margin; outside the slide = background (zero padding)."""
    lab = G["lab"]
    out = np.zeros((TILE + 2 * M, TILE + 2 * M), np.int64)
    ya, yb = max(y0 - M, 0), min(y0 + TILE + M, lab.shape[0])
    xa, xb = max(x0 - M, 0), min(x0 + TILE + M, lab.shape[1])
    out[ya - (y0 - M):yb - (y0 - M), xa - (x0 - M):xb - (x0 - M)] = lab[ya:yb, xa:xb]
    return out


def lin_near(x0, y0, pad):
    a, b = np.searchsorted(G["x"], [x0 - pad, x0 + TILE + pad])
    i = np.arange(a, b)
    return i[(G["y"][i] >= y0 - pad) & (G["y"][i] < y0 + TILE + pad)]


def one(t):
    from PIL import Image
    from stages.lin_typing import compare as CP
    from stages.lin_typing import nuclei as NU
    from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq
    import hashlib
    r = G["r"]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little"))
    x0, y0 = map(int, t.split("_"))
    he = np.asarray(Image.open(G["realhe"] / f"{t}.png").convert("RGB"))
    hod = CP.hematoxylin(he)
    labw = window(x0, y0)
    _, loose = NU.erode_labels(labw)
    cellm = labw[M:-M, M:-M]
    L = loose[M:-M, M:-M].astype(np.int64)
    cv = _cellvit_json_to_instance_mask(G["realcv"] / f"{t}.json", image_hw=(TILE, TILE))
    sd = G["sd"].get(t)
    if sd is None:
        return None
    pm = G["pack"] / "exp_channels" / "cell_masks" / f"{t}.png"
    if not pm.exists():
        return None
    r0 = np.asarray(Image.open(pm))
    r0 = r0[..., 0] if r0.ndim == 3 else r0
    R0 = relabel(r0 > (0.5 if r0.max() <= 1 else 0))

    # ---- D1 centroid matching (Lin quantification centroids, robust.py rule)
    idx = lin_near(x0, y0, r)
    lp = np.stack([G["x"][idx] - x0, G["y"][idx] - y0], 1)
    lin_in = inner(lp, r)
    ids_cv, pc = centroids(cv)
    ids_sd, ps = centroids(sd)
    mdc, mlc = match(pc, lp, r)
    mds, mls = match(ps, lp, r)
    row = dict(set=G["set"], tile=t, n_lin=int(lin_in.sum()),
               cv_tp=int((mlc & lin_in).sum()), cv_fn=int((~mlc & lin_in).sum()), cv_fp=int((~mdc & inner(pc, r)).sum()),
               sd_tp=int((mls & lin_in).sum()), sd_fn=int((~mls & lin_in).sum()), sd_fp=int((~mds & inner(ps, r)).sum()))

    # ---- D2 scores: mean hematoxylin OD over each Lin loose nucleus (cells with centroid in the inner region)
    occupied = (cellm > 0) | (cv > 0) | (sd > 0)
    from scipy import ndimage
    occ = ndimage.binary_dilation(occupied, np.ones((3, 3), bool))
    sc, cls, neg = [], [], []
    for k in np.flatnonzero(lin_in):
        cid = int(G["cid"][idx[k]])
        pix = L == cid
        if not pix.any():
            continue
        s = float(hod[pix].mean())
        both, none = mlc[k] and mls[k], (not mlc[k]) and (not mls[k])
        cl = 2 if both else (0 if none else 1)   # 2 matched by both, 1 by one, 0 by neither
        sc.append(s)
        cls.append(cl)
        if cl == 2:   # negative: same footprint moved to a free spot of the tile
            yy, xx = np.nonzero(pix)
            yy, xx = yy - yy.min(), xx - xx.min()
            h, w = yy.max() + 1, xx.max() + 1
            for _ in range(50):
                oy, ox = rng.integers(0, TILE - h + 1), rng.integers(0, TILE - w + 1)
                if not occ[yy + oy, xx + ox].any():
                    neg.append(float(hod[yy + oy, xx + ox].mean()))
                    break

    # ---- D3 unmatched detections: centroid inside a Lin cell?
    def d3(p, md):
        u = (~md) & inner(p, r)
        q = np.clip(np.rint(p[u]).astype(int), 0, TILE - 1)
        return int(u.sum()), int((cellm[q[:, 1], q[:, 0]] > 0).sum()) if u.any() else 0
    row["cv_fp_in_lin"] = d3(pc, mdc)[1]
    row["sd_fp_in_lin"] = d3(ps, mds)[1]

    # ---- D4 masks (binarise + relabel as PixCell), PQ vs real CellViT and StarDist
    L1 = relabel(separate(L) > 0)
    H1 = relabel(separate(sd) > 0)
    H2 = relabel(separate(cv) > 0)
    for name, m in (("R0", R0), ("L1", L1), ("H1", H1), ("H2", H2)):
        row[f"pq_{name}_cv"] = _compute_pq(cv, m)[2]
        row[f"pq_{name}_sd"] = _compute_pq(sd, m)[2]
        row[f"n_{name}"] = int(m.max())
    row["pq_sd_cv"] = _compute_pq(cv, sd)[2]
    row["n_cv"], row["n_sd"] = int(len(ids_cv)), int(len(ids_sd))
    row["n_lin_tile"] = int(len(np.unique(L)) - (1 if (L == 0).any() else 0))
    row["n_R0_lincells"] = int(len(np.unique(cellm)) - (1 if (cellm == 0).any() else 0))

    # ---- H-series type transfer: H object centroid in a Lin cell -> that cell; else nearest Lin centroid within r
    for name, inst in (("H1", sd), ("H2", cv)):
        ids, p = centroids(inst)
        hit = np.zeros(3, int)
        typed = 0
        lin_hit = set()
        for (px, py) in p:
            q = cellm[min(int(py), TILE - 1), min(int(px), TILE - 1)]
            j = None
            if q > 0:
                j = np.flatnonzero(G["cid"][idx] == q)
                j = j[0] if len(j) else None
            if j is None and len(lp):
                d = np.hypot(lp[:, 0] - px, lp[:, 1] - py)
                if d.min() <= r:
                    j = int(np.argmin(d))
            if j is not None:
                typed += 1
                lin_hit.add(int(j))
        for j in lin_hit:
            if lin_in[j]:
                hit[G["ty"][idx[j]]] += 1
        row[f"{name}_typed"] = typed
        row[f"{name}_obj"] = int(len(ids))
        for k, nm in enumerate(POP):
            row[f"{name}_hit_{nm}"] = int(hit[k])
    for k, nm in enumerate(POP):
        row[f"lin_{nm}"] = int((G["ty"][idx[lin_in]] == k).sum())
    return row, np.array(sc), np.array(cls), np.array(neg), dict(he=he, R0=R0, L=L, sd=sd, cv=cv)


def two(job):
    """Phase 2: L2 = Lin loose nuclei with mean hematoxylin OD >= the set's Youden t (cells inside the tile; cells
    whose centroid is outside the inner region are kept as in L1, since they were not scored)."""
    from PIL import Image
    from stages.lin_typing import compare as CP
    from stages.lin_typing import nuclei as NU
    from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq
    t, thr = job
    x0, y0 = map(int, t.split("_"))
    he = np.asarray(Image.open(G["realhe"] / f"{t}.png").convert("RGB"))
    hod = CP.hematoxylin(he)
    labw = window(x0, y0)
    _, loose = NU.erode_labels(labw)
    L = loose[M:-M, M:-M].astype(np.int64)
    r = G["r"]
    idx = lin_near(x0, y0, r)
    lp = np.stack([G["x"][idx] - x0, G["y"][idx] - y0], 1)
    scored = set(int(G["cid"][idx[k]]) for k in np.flatnonzero(inner(lp, r)))
    L2 = L.copy()
    for cid in np.unique(L):
        if cid == 0 or int(cid) not in scored:
            continue
        pix = L == cid
        if hod[pix].mean() < thr:
            L2[pix] = 0
    cv = _cellvit_json_to_instance_mask(G["realcv"] / f"{t}.json", image_hw=(TILE, TILE))
    sd = G["sd"][t]
    m = relabel(separate(L2) > 0)
    return dict(tile=t, pq_L2_cv=_compute_pq(cv, m)[2], pq_L2_sd=_compute_pq(sd, m)[2], n_L2=int(m.max())), m


def cmd_tiles(set_, tif, lin_dir, realhe, realcv, sd_dir, pack, r, out, nproc):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    tiles = [r_["tile"] for r_ in csv.DictReader(open(Path(sd_dir) / "real.csv"))]
    if os.environ.get("DIAG_N"):
        tiles = tiles[:int(os.environ["DIAG_N"])]
    init(set_, tif, lin_dir, realhe, realcv, sd_dir, pack, r)
    print(set_, len(tiles), "tiles; lab", G["lab"].shape, flush=True)
    ctx = mp.get_context("fork")
    rows, S, C, T, N = [], [], [], [], []
    keep = {}
    with ctx.Pool(int(nproc)) as pool:
        for k, res in enumerate(pool.imap(one, tiles, chunksize=4)):
            if res is None:
                print("skip", tiles[k], flush=True)
                continue
            row, sc, cl, neg, arr = res
            rows.append(row)
            S.append(sc); C.append(cl); T.append(np.full(len(sc), len(rows) - 1)); N.append(neg)
            keep[row["tile"]] = (row["n_lin"], arr)
            if k % 100 == 0:
                print(k, flush=True)
    with open(out / "tiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    negt = np.concatenate([np.full(len(n), i) for i, n in enumerate(N)]) if N else np.zeros(0)
    np.savez_compressed(out / "lin_scores.npz", score=np.concatenate(S), cls=np.concatenate(C), tile=np.concatenate(T),
                        neg=np.concatenate(N), neg_tile=negt, tiles=np.array([r_["tile"] for r_ in rows]))
    # contact-sheet sample: 3 tiles with Lin count above the set median, rng(42)
    nl = np.array([keep[t][0] for t in keep])
    cand = sorted(t for t in keep if keep[t][0] > np.median(nl))
    pick = np.random.default_rng(SEED).choice(len(cand), 3, replace=False)
    for i in pick:
        t = cand[i]
        np.savez_compressed(out / f"sample_{t}.npz", **keep[t][1])
    z = np.load(out / "lin_scores.npz")
    y = youden(z["score"][z["cls"] == 2], z["neg"])
    print("youden", y, flush=True)
    l2 = []
    samp = {cand[i] for i in pick}
    with ctx.Pool(int(nproc)) as pool:
        for row2, m in pool.imap(two, [(t, y["t"]) for t in z["tiles"]], chunksize=4):
            l2.append(row2)
            if row2["tile"] in samp:
                np.save(out / f"sample_{row2['tile']}_L2.npy", m)
    with open(out / "l2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(l2[0]))
        w.writeheader()
        w.writerows(l2)
    json.dump(dict(set=set_, n_tiles=len(rows), r=float(r), youden=y, sample=[cand[i] for i in pick]), open(out / "info.json", "w"))
    print("DONE", len(rows), flush=True)


# ------------------------------------------------------------------------------------------------ stats
def youden(pos, neg):
    v = np.unique(np.concatenate([pos, neg]))
    best = (-1, None, None, None)
    for t in v:
        tpr, fpr = (pos >= t).mean(), (neg >= t).mean()
        if tpr - fpr > best[0]:
            best = (tpr - fpr, t, tpr, fpr)
    return dict(t=float(best[1]), tpr=float(best[2]), fpr=float(best[3]), j=float(best[0]))


def cmd_stats(root, figdir):
    root, figdir = Path(root), Path(figdir)
    res = {}
    allrows = []
    vis = {}
    for set_ in ("crc33sec", "crc36ho"):
        d = root / set_
        rows = list(csv.DictReader(open(d / "tiles.csv")))
        z = np.load(d / "lin_scores.npz")
        sc, cl = z["score"], z["cls"]
        y = youden(sc[cl == 2], z["neg"])
        f = float((sc[cl == 0] >= y["t"]).mean())
        rg = float(np.clip((f - y["fpr"]) / (y["tpr"] - y["fpr"]), 0, 1))
        f1 = float((sc[cl == 1] >= y["t"]).mean())
        rg1 = float(np.clip((f1 - y["fpr"]) / (y["tpr"] - y["fpr"]), 0, 1))
        S = lambda k: int(sum(int(r_[k]) for r_ in rows))
        res[set_] = dict(
            n_tiles=len(rows), n_lin=S("n_lin"),
            D1=dict(cv=dict(tp=S("cv_tp"), fn=S("cv_fn"), fp=S("cv_fp")), sd=dict(tp=S("sd_tp"), fn=S("sd_fn"), fp=S("sd_fp"))),
            D2=dict(youden=y, n_pos=int((cl == 2).sum()), n_neg=int(len(z["neg"])), n_one=int((cl == 1).sum()),
                    n_none=int((cl == 0).sum()), frac_none_above_t=f, rogan_gladen_none=rg,
                    frac_one_above_t=f1, rogan_gladen_one=rg1),
            D3=dict(cv_fp_in_lin=S("cv_fp_in_lin") / max(S("cv_fp"), 1), sd_fp_in_lin=S("sd_fp_in_lin") / max(S("sd_fp"), 1)),
            merge=dict(R0_components_per_lin_cell=S("n_R0") / max(S("n_R0_lincells"), 1),
                       L1_components_per_lin_cell=S("n_L1") / max(S("n_lin_tile"), 1)),
            types={nm: dict(lin=S(f"lin_{nm}"), H1_hit=S(f"H1_hit_{nm}"), H2_hit=S(f"H2_hit_{nm}")) for nm in POP},
            H_typed=dict(H1=S("H1_typed") / max(S("H1_obj"), 1), H2=S("H2_typed") / max(S("H2_obj"), 1)))
        vis[set_] = y["t"]
        for r_ in rows:
            allrows.append(r_)
    # L2 PQ comes from the tiles step's phase 2 (l2.csv, same Youden t)
    for set_ in ("crc33sec", "crc36ho"):
        p = root / set_ / "l2.csv"
        if p.exists():
            l2 = {r_["tile"]: r_ for r_ in csv.DictReader(open(p))}
            for r_ in allrows:
                if r_["set"] == set_ and r_["tile"] in l2:
                    r_.update({k: v for k, v in l2[r_["tile"]].items() if k != "tile"})
    cross = {"R0": lambda r_: (float(r_["pq_R0_cv"]) + float(r_["pq_R0_sd"])) / 2,
             "L1": lambda r_: (float(r_["pq_L1_cv"]) + float(r_["pq_L1_sd"])) / 2,
             "H1": lambda r_: float(r_["pq_H1_cv"]),
             "H2": lambda r_: float(r_["pq_H2_sd"]),
             "SDvsCV": lambda r_: float(r_["pq_sd_cv"])}
    if all("pq_L2_cv" in r_ for r_ in allrows):
        cross["L2"] = lambda r_: (float(r_["pq_L2_cv"]) + float(r_["pq_L2_sd"])) / 2
    V = {k: np.array([fn(r_) for r_ in allrows]) for k, fn in cross.items()}
    idx = np.random.default_rng(SEED).integers(0, len(allrows), size=(B, len(allrows)))
    ci = {k: dict(point=float(v.mean()), ci95=[float(np.quantile(v[idx].mean(1), .025)), float(np.quantile(v[idx].mean(1), .975))])
          for k, v in V.items()}
    per_set = {s: {k: float(np.mean([cross[k](r_) for r_ in allrows if r_["set"] == s])) for k in cross} for s in ("crc33sec", "crc36ho")}
    ov = lambda a, b: not (ci[a]["ci95"][1] < ci[b]["ci95"][0] or ci[b]["ci95"][1] < ci[a]["ci95"][0])
    bestH = "H1" if ov("H1", "H2") or ci["H1"]["point"] >= ci["H2"]["point"] else "H2"
    Ls = [k for k in ("L1", "L2") if k in ci]
    if any(ov(k, bestH) or ci[k]["ci95"][0] > ci[bestH]["ci95"][1] for k in Ls):
        choice = "L2" if ("L2" in ci and ci["L2"]["ci95"][0] > ci["L1"]["ci95"][1]) else "L1"
        why = f"an L candidate's cross-PQ CI overlaps (or exceeds) the best H ({bestH}) -> L series; L2 only if its CI is above L1's"
    else:
        choice = bestH
        why = f"no L candidate overlaps the best H -> {bestH}"
    out = dict(rule="PREREG.md §2.3", per_set_diag=res, cross_pq=ci, cross_pq_per_set=per_set, choice=choice, why=why,
               B=B, seed=SEED, n_tiles=len(allrows))
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir.parent / "data").mkdir(exist_ok=True)
    (figdir.parent / "data" / "diag_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(cross_pq=ci, choice=choice, why=why), indent=1))
    sheet(root, figdir, choice, vis)


def sheet(root, figdir, choice, vis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    smp = []
    for set_ in ("crc33sec", "crc36ho"):
        info = json.load(open(root / set_ / "info.json"))
        smp += [(set_, t, root / set_ / f"sample_{t}.npz") for t in info["sample"]]
    cols = ["H&E", "R0 current mask", "L1 Lin true shapes", f"{choice} chosen"]
    fig, ax = plt.subplots(len(smp), 4, figsize=(7.1, 1.85 * len(smp)))
    for i, (s, t, p) in enumerate(smp):
        z = np.load(p)
        he = z["he"]
        L1 = relabel(separate(z["L"]) > 0)
        ch = {"L1": L1, "H1": relabel(separate(z["sd"]) > 0), "H2": relabel(separate(z["cv"]) > 0)}.get(choice, L1)
        if choice == "L2":
            ch = np.load(root / s / f"sample_{t}_L2.npy")
        panels = [he, (z["R0"] > 0), L1, ch]
        for j, pnl in enumerate(panels):
            a = ax[i, j]
            if j == 0:
                a.imshow(he)
            elif j == 1:
                a.imshow(pnl, cmap="gray")
            else:
                ov = he.copy()
                b = find_boundaries(pnl, mode="outer")
                ov[b] = (0, 230, 0)
                a.imshow(ov)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(cols[j], fontsize=7)
        ax[i, 0].set_ylabel(f"{s}\n{t}", fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(figdir / "contact_sheet.png", dpi=200)
    print("sheet", figdir / "contact_sheet.png")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "tiles":
        cmd_tiles(*sys.argv[2:])
    elif m == "stats":
        cmd_stats(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
