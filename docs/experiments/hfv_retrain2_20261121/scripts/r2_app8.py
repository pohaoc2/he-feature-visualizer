"""hfv_retrain2 stage 3 (PREREG §9.3), Hyak CPU: readout 3 (app 8 readouts on CRC16's 12 windows) for model N vs model K.
Written before any N image exists.

  python -I r2_app8.py SEV_APPS L1 S3B_CODE S3B_CELLS CV_REAL CV8 CV10 CVND APP8_JSON OUT_JSON
    SEV_APPS   sim-exp-validation scripts/analysis/sim2real_apps (app8_stats.py, app10_stats.py; imported, unchanged)
    L1         hfv_retrain2 eval/crc16_L1                      CV_REAL  app 2 cv/RAW (real__<w>__<tile>__0.json)
    CV8        app 8 cv (<arm>/<arm>__<w>__<tile>__<seed>.json; KP, KD)
    CV10       app 10 cv (KP = model N + own style on L1 = arm NP)          CVND  stage3 cv/ND (ND__...json)
    APP8_JSON  app 8 results_app8.json (gate: the K-side per-window counts must equal it)

Sources and truth units (§9.3): model N arms NP (= app 10 KP) and ND on the CRC16 L1 instance units (app10_stats.load_truth,
type = the L1 type channels); model K arms KP, KD (app 8) on the s3b units (app8_stats.py: msi_s3b_edits.load_block +
msi_s1_edits.units). Real H&E (seed 0) is matched on both: R_L1 and R_s3b. Matching (both): a detection whose centroid pixel
lies on a unit pixel is matched to it; seeds 42 + 43 pooled.
Readouts = app10_stats metr: delta = canc_neo - heal_neo, canc_neo, median nucleus area (contour polygon, all detections),
within-tile delta (Mantel-Haenszel risk difference, Greenland-Robins weights, tiles with n_c >= 1 and n_h >= 1; app 10
prereg §6.2), det_neo, detections per image.
Intervals: one (2000 x 12) window index matrix, default_rng(20261128).integers(0, 12, (2000, 12)) (app 8), every source;
N - K = NP - KP and ND - KD paired on it. Decision inputs (§9.5): the delta intervals of NP - KP and ND - KD.
"""
import json
import sys
from pathlib import Path

import numpy as np

SEV, L1, S3C, CELLS, CVR, CV8, CV10, CVND, A8J, OUT = sys.argv[1:11]
sys.path.insert(0, SEV)
sys.path.insert(0, S3C)
import app8_stats as A8S  # noqa: E402
import app10_stats as A10S  # noqa: E402
import msi_s3b_edits as S3B  # noqa: E402

NB, SEED, HELD, area = A8S.NB, A8S.SEED, A8S.HELD, A8S.area
E = S3B.E
L1, CVR, CV8, CV10, CVND = map(Path, (L1, CVR, CV8, CV10, CVND))
# 2026-11-22: after the ch6 STOP, NP is its own stage-3 job; pass CV10 = stage3 cv (holds NP/ with NP__...json)
NPSRC = (CV10 / "NP", "NP") if (CV10 / "NP").is_dir() else (CV10 / "KP", "KP")
SRC_N = {"R_L1": (CVR / "RAW", "real", (0,)), "NP": (*NPSRC, (42, 43)), "ND": (CVND, "ND", (42, 43))}
SRC_K = {"R_s3b": (CVR / "RAW", "real", (0,)), "KP": (CV8 / "KP", "KP", (42, 43)), "KD": (CV8 / "KD", "KD", (42, 43))}


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


def read(d, pre, w, tid, seed):
    f = d / f"{pre}__{w}__{tid}__{seed}.json"
    if not f.exists():
        stop(f"missing {f}")
    return json.loads(f.read_text())["cells"]


W = {}
for w in HELD:
    # model N side: L1 units (app10_stats.load_truth, gated against labels / exp_channels)
    central, lab, utype = A10S.load_truth(L1, w)
    WIN = lab.shape[0]
    for s, (d, pre, seeds) in SRC_N.items():
        conf_t = np.zeros((len(central), 3, 6), int)
        tot = np.zeros(6, int)
        a_all = []
        for seed in seeds:
            for k, (tid, g) in enumerate(central.items()):
                for c in read(d, pre, w, tid, seed):
                    tc = int(c["type_cellvit"])
                    tot[tc] += 1
                    a_all.append(area(c["contour"]))
                    x, y = c["centroid"]
                    u = lab[min(int(g["oy"] + y), WIN - 1), min(int(g["ox"] + x), WIN - 1)]
                    if u > 0:
                        conf_t[k, utype[u], tc] += 1
        W.setdefault(s, {})[w] = dict(conf_t=conf_t, tot=tot, n_img=len(seeds) * len(central), a_all=np.array(a_all))
    # model K side: s3b units (app8_stats.py rule, per tile kept)
    arr, X, labk, sl = S3B.load_block(CELLS, w)
    U = E.units(X, labk, w)
    ty = np.argmax(X[1:4], 0)
    key = np.where(labk > 0, labk * 3 + ty, -1)
    uid = {u["label"] * 3 + u["type"]: i for i, u in enumerate(U)}
    ut = np.array([u["type"] for u in U])
    ids = arr["central_ids"].astype(str).tolist()
    for s, (d, pre, seeds) in SRC_K.items():
        conf_t = np.zeros((len(ids), 3, 6), int)
        tot = np.zeros(6, int)
        a_all = []
        for seed in seeds:
            for k, tid in enumerate(ids):
                oy, ox = sl[k][0].start, sl[k][1].start
                for c in read(d, pre, w, tid, seed):
                    tc = int(c["type_cellvit"])
                    tot[tc] += 1
                    a_all.append(area(c["contour"]))
                    x, y = c["centroid"]
                    kk = key[min(int(oy + y), E.N - 1), min(int(ox + x), E.N - 1)]
                    if kk >= 0:
                        conf_t[k, ut[uid[int(kk)]], tc] += 1
        W.setdefault(s, {})[w] = dict(conf_t=conf_t, tot=tot, n_img=len(seeds) * len(ids), a_all=np.array(a_all))

# gate: K side reproduces app 8's per-window confusion and totals
a8 = json.loads(Path(A8J).read_text())["per_window"]
for s, s8 in (("R_s3b", "R"), ("KP", "KP"), ("KD", "KD")):
    for w in HELD:
        if W[s][w]["conf_t"].sum(0).tolist() != a8[s8][w]["conf"] or W[s][w]["tot"].tolist() != a8[s8][w]["tot"]:
            stop(f"{s} {w} differs from results_app8.json")
print("gate: K side = results_app8.json per-window counts (R, KP, KD x 12 windows)", flush=True)


def metr(s, ws):
    ct = np.concatenate([W[s][w]["conf_t"] for w in ws])
    conf = ct.sum(0)
    tot = sum(W[s][w]["tot"] for w in ws)
    r = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    nc, kc = ct[:, 1].sum(1), ct[:, 1, 1]
    nh, kh = ct[:, 0].sum(1), ct[:, 0, 1]
    ok = (nc >= 1) & (nh >= 1)
    n = nc + nh
    num = np.sum((kc * nh - kh * nc)[ok] / n[ok])
    den = np.sum((nc * nh)[ok] / n[ok])
    aa = np.concatenate([W[s][w]["a_all"] for w in ws])
    o = dict(delta=r[1, 1] - r[0, 1], canc_neo=r[1, 1], heal_neo=r[0, 1],
             area_med=float(np.median(aa)) if len(aa) else np.nan, within_delta=num / den if den > 0 else np.nan,
             tiles_eligible=int(ok.sum()), det_neo=tot[1] / max(tot.sum(), 1),
             n_det_per_img=tot.sum() / sum(W[s][w]["n_img"] for w in ws),
             n_truth_cancer_matched=int(nc.sum()), n_truth_healthy_matched=int(nh.sum()))
    return {k: float(v) for k, v in o.items()}


SRC = list(SRC_N) + list(SRC_K)
pt = {s: metr(s, HELD) for s in SRC}
idx = np.random.default_rng(SEED).integers(0, len(HELD), size=(NB, len(HELD)))
bs = {s: [metr(s, [HELD[j] for j in row]) for row in idx] for s in SRC}
q = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]
ci = {s: {k: q([b[k] for b in bs[s]]) for k in pt[s]} for s in SRC}


def diff(a, b, k):
    v = np.array([x[k] for x in bs[a]]) - np.array([x[k] for x in bs[b]])
    return dict(point=pt[a][k] - pt[b][k], ci95=q(v))


KS = ("delta", "canc_neo", "within_delta", "area_med", "det_neo", "n_det_per_img")
nk = {f"{a}-{b}:{k}": diff(a, b, k) for a, b in (("NP", "KP"), ("ND", "KD")) for k in KS}
desc = {f"{a}-{b}:{k}": diff(a, b, k)
        for a, b in (("R_L1", "NP"), ("R_L1", "ND"), ("NP", "ND"), ("R_s3b", "KP"), ("R_s3b", "KD"), ("KP", "KD"),
                     ("R_L1", "R_s3b"))
        for k in KS}
res = dict(rule="hfv_retrain2 PREREG §9.3 (r2_app8.py)", B=NB, seed=SEED, point=pt, ci95=ci, n_minus_k=nk,
           descriptive=desc,
           per_window={s: {w: dict(conf=W[s][w]["conf_t"].sum(0).tolist(), tot=W[s][w]["tot"].tolist()) for w in HELD}
                       for s in SRC})
Path(OUT).write_text(json.dumps(res, indent=1) + "\n")
for s in SRC:
    print(s, {k: round(v, 3) for k, v in pt[s].items()})
for k, v in (nk | desc).items():
    print(k, round(v["point"], 3), [round(x, 3) for x in v["ci95"]])
