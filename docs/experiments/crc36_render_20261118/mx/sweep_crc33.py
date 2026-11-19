"""Find the hfv Stage-4 settings that produced orion-crc33 exp_channels/{vasculature,oxygen,glucose} (step 1a).

  HFV=<hfv code> sweep_crc33.py MX_TIFF MARKERS_CSV INDEX_JSON TRAIN_TAR OUT_JSON

Candidates (every value appears in hfv code / docs / git history, see README):
  vasculature : CD31 only  vs  CD31 + SMA refine (--vasc-sma-refine)
  oxygen/glucose : --wsi-pde-ds in {2, 4 (hfv default), 8 (PixCell manuscript_outline)}
                   x SMA refine of the WSI vessel mask {off, on}
                   x e-folding length L (um) in {60, 100, 120, 160, 200, 450}
                   (160/450 = --oxygen/--glucose-krogh-um defaults, used by wsi-pde before commit ce3bb3e;
                    200/120 = --oxygen/--glucose-pde-krogh-um defaults after ce3bb3e; 60/100 = the alternative run
                    in docs/plans/2026-03-21-oxygen-glucose-pde-diagnostics.md)
Each candidate field is compared with BOTH training channels on a fixed random sample of tiles
(500 tile ids, numpy default_rng(20261118)).
"""
import json
import os
import sys
import tarfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mx_lib as L  # noqa: E402

TIFF, MARKERS, INDEX, TRAIN_TAR, OUT = sys.argv[1:6]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def metrics(pred, ref):
    p = np.stack(pred).astype(np.float64)
    r = np.stack(ref).astype(np.float64)
    d = np.abs(p - r)
    eq = [np.array_equal(a, b) for a, b in zip(pred, ref)]
    pf, rf = p.ravel(), r.ravel()
    corr = float(np.corrcoef(pf, rf)[0, 1]) if pf.std() > 0 and rf.std() > 0 else None
    return dict(tile_equal_frac=float(np.mean(eq)), max_abs=float(d.max()), mean_abs=float(d.mean()),
                p99_abs=float(np.percentile(d, 99)), pearson=corr)


idx = json.load(open(INDEX))
mpp = float(idx["mx_mpp"])
assert float(idx["he_mpp"]) == mpp and idx["scale_he_to_mx"] == 1.0, "STOP: index not 1:1 H&E/MX"
ids = sorted(f"{p['x0']}_{p['y0']}" for p in idx["patches"])
rng = np.random.default_rng(20261118)
sample = sorted(rng.choice(ids, 500, replace=False).tolist())
ci, names = L.channel_index(MARKERS)
assert names == idx["channels"], ("STOP: markers.csv order differs from hfv index channels", names, idx["channels"])
info = L.ome_info(TIFF)
log("ome", info, "channels", ci)
assert info["shape"][-2:] == [idx["img_h"], idx["img_w"]], "STOP: multiplex size differs from hfv index img size"

ref = {c: {} for c in ("vasculature", "oxygen", "glucose")}
with tarfile.open(TRAIN_TAR) as tf:
    want = {f"orion-crc33/exp_channels/{c}/{t}.npy": (c, t) for c in ref for t in sample}
    for m in tf:
        if m.name in want:
            c, t = want[m.name]
            raw = tf.extractfile(m).read()
            ref[c][t] = (raw, np.load(__import__("io").BytesIO(raw), allow_pickle=False))
for c in ref:
    assert len(ref[c]) == len(sample), (c, len(ref[c]))
    a = next(iter(ref[c].values()))[1]
    log("train", c, a.dtype, a.shape)
R = dict(tiff=TIFF, ome=info, sample_n=len(sample), sample_seed=20261118,
         train_dtypes={c: str(next(iter(ref[c].values()))[1].dtype) for c in ref})

# ---- vasculature (level-0 crops) ----
cd31 = L.read_plane(TIFF, ci["CD31"])
sma = L.read_plane(TIFF, ci["SMA"])
log("planes read", cd31.shape)
R["vasculature"] = {}
for tag, use_sma in (("cd31_only", False), ("cd31_sma", True)):
    pred, byte_eq = [], 0
    for t in sample:
        x0, y0 = map(int, t.split("_"))
        m = L.vessel_mask(L.crop(cd31, x0, y0), L.crop(sma, x0, y0) if use_sma else None)
        pred.append(m)
        byte_eq += L.npy_bytes(m) == ref["vasculature"][t][0]
    mm = metrics(pred, [ref["vasculature"][t][1] for t in sample])
    mm["byte_equal_frac"] = byte_eq / len(sample)
    R["vasculature"][tag] = mm
    log("vasc", tag, mm)
del cd31, sma

# ---- oxygen / glucose (whole-slide coarse WKB) ----
R["fields"] = []
Ls = [60.0, 100.0, 120.0, 160.0, 200.0, 450.0]
for ds in (2, 4, 8):
    for use_sma in (False, True):
        t1 = time.time()
        f = L.wsi_fields(TIFF, ci, ds, {str(v): v for v in Ls}, mpp, use_sma)
        log("solved ds", ds, "sma", use_sma, f["_shape"], f"{time.time() - t1:.0f}s")
        for v in Ls:
            pred = [L.coarse_tile(f[str(v)], *map(int, t.split("_")), ds, mpp) for t in sample]
            for c in ("oxygen", "glucose"):
                mm = metrics(pred, [ref[c][t][1] for t in sample])
                mm["byte_equal_frac"] = float(np.mean([L.npy_bytes(p) == ref[c][t][0] for p, t in zip(pred, sample)]))
                row = dict(channel=c, ds=ds, sma=use_sma, L_um=v, coarse_shape=f["_shape"], **mm)
                R["fields"].append(row)
                log(row)
        del f
        json.dump(R, open(OUT, "w"), indent=1)
best = {}
for c in ("oxygen", "glucose"):
    rows = [r for r in R["fields"] if r["channel"] == c]
    best[c] = min(rows, key=lambda r: (r["max_abs"], -(r["pearson"] or -1)))
R["best"] = best
R["wall_s"] = time.time() - T0
json.dump(R, open(OUT, "w"), indent=1)
log("best", best)
