"""fidelity_variants_20261122 (PREREG.md §1, §4): segmenter-free hematoxylin readouts of the retrain-1 Fig 2 images
(five variants) and the real H&E, each against the same input mask (R0 cell_masks). Hyak CPU, read-only inputs.

  python fv_hema.py OUT NPROC        (HFV on sys.path: stages.lin_typing.compare)
-> OUT/hema_{36,33}.csv (one row per tile x arm), OUT/info.json; exit 1 = STOP (any gate)

Readouts (mask_truenuc_20261122 truenuc.py pix(), copied): hod = rgb2hed hematoxylin OD; fg = cell_masks > 0
  a = |fg & hod>=t| / |fg|; b = |fg & hod>=t| / |hod>=t|; c = mean(hod | fg) - mean(hod | ~fg)
  t16 = 0.02243768021416608 (rule t); tS = stage-1 Youden t of the slide (crc36ho / crc33sec, sensitivity)
  descriptive: auc = compare.nucleus_auc(hod, fg); hmean = mean(hod); frac = |fg| / size
Gates: zip sha256 == cellvit_official zips_manifest.json; tile ids equal across the five zips and retgt_<set>;
R0 cell_masks == evalpack_<set>.tar cell_masks (pixels); truenuc real-H&E-vs-L1 a/b/c (t16) reproduced on 8 CRC16 tiles
(float32 hematoxylin as truenuc part A; readouts here use float64 as truenuc part B; job 43466429 STOPPED on a float64
comparison, max diff 2.6e-9).
"""
import csv
import hashlib
import io
import json
import multiprocessing as mp
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from stages.lin_typing import compare as CP

CC = Path("/gscratch/cheme/chiu")
D = CC / "cellvit_official_20261122"
S3 = CC / "hfv_retrain2_20261121/stage3"
PACKS = CC / "hfv_eval_20261118/packs"
TN = CC / "mask_truenuc_20261122/full"
SETS = {"36": "crc36ho", "33": "crc33sec"}
VARS = ("van", "grp", "cat", "gbp", "pch")
T16 = 0.02243768021416608
TS = {"crc36ho": 0.021964938593192773, "crc33sec": 0.02025237538031727}
MANIFEST = {  # cellvit_official_20261122 data/fig2/zips_manifest.json
    "F2_36_van": "15e62dbf0facbced00bdfb919afe2d615bde6b986928e058a99da11d1485f8b6",
    "F2_36_grp": "5b1fd660cc6825018b15ce734f14e4a8058ef8a0368dbb68a1ed1de7e1f6c2fc",
    "F2_36_cat": "419b928214666073890e7deb92f718c9b10b027f91f1fd703b9d545fa5bde2b7",
    "F2_36_gbp": "aeb1fb5a23f4202c232d464b13de9bcbe07382867a8dfadebec158e0fadc890f",
    "F2_36_pch": "f73c1712a9be5555f813ff6e3c91c2b92db3ba34219fc3630295a57305528d7f",
    "F2_33_van": "b966fcb3e53f3b39be7d188980766b851a5fd6082b5d88587d9f24ec23244385",
    "F2_33_grp": "05751215b137bc736c26c5ca378b77922ec4d0a84272c76f7c0c1e810e9f79cf",
    "F2_33_cat": "b5e45094bdfc52a86f9b38c2af255a975c9c75aab7ce65aac5aecc46d92cf4c5",
    "F2_33_gbp": "57ed1bd6a01d3e93c88db2b927c0018a3357439ad39838a770e52932203e40c1",
    "F2_33_pch": "33aac51b5a0605ecf1bd07bd24c59d5da72bf3950808b86ef7e8b7e47d1473cb",
}


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rgb_bytes(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def raw_bytes(b):
    return np.asarray(Image.open(io.BytesIO(b)))


def png_bin(a):
    """truenuc.py png_bin on an array."""
    a = a[..., 0] if a.ndim == 3 else a
    return a > (0.5 if a.max() <= 1 else 0)


def hema(img):
    return CP.hematoxylin(img).astype(np.float64)


def pix(hod, fg, t):
    """truenuc.py pix() (a, b, c only)."""
    pos = hod >= t
    nf, npos = int(fg.sum()), int(pos.sum())
    both = int((fg & pos).sum())
    a = both / nf if nf else np.nan
    b = both / npos if npos else np.nan
    c = float(hod[fg].mean() - hod[~fg].mean()) if 0 < nf < fg.size else np.nan
    return a, b, c


def readouts(img, fg, ts):
    hod = hema(img)
    a, b, c = pix(hod, fg, T16)
    a_s, b_s, _ = pix(hod, fg, ts)
    auc = CP.nucleus_auc(hod, fg) if 0 < fg.sum() < fg.size else np.nan
    return dict(a=a, b=b, c=c, a_tS=a_s, b_tS=b_s, auc=auc, hmean=float(hod.mean()), frac=float(fg.mean()))


def job(args):
    k, s, t = args
    mask_raw = np.asarray(Image.open(S3 / f"packs/R0/orion-{s}/exp_channels/cell_masks/{t}.png"))
    fg = png_bin(mask_raw)
    out = []
    real = np.asarray(Image.open(D / f"zips/retgt_{s}/{t}.png").convert("RGB"))
    out.append(dict(group=k, tile=t, arm="real", **readouts(real, fg, TS[s])))
    for v in VARS:
        with zipfile.ZipFile(D / f"zips/F2_{k}_{v}.zip") as zf:
            img = rgb_bytes(zf.read(f"{t}.png"))
        out.append(dict(group=k, tile=t, arm=v, **readouts(img, fg, TS[s])))
    return out


def main(out, nproc):
    out = Path(out)
    out.mkdir(parents=True)
    info = dict(t16=T16, tS=TS, gates={})
    # gate 1: sha256
    for name, h in MANIFEST.items():
        if sha(D / f"zips/{name}.zip") != h:
            stop(f"sha256 {name}")
    info["gates"]["sha256"] = f"{len(MANIFEST)}/{len(MANIFEST)}"
    print("gate sha256 ok", flush=True)
    # gate 2: tile ids
    ids = {}
    for k, s in SETS.items():
        sets = [sorted(Path(n).stem for n in zipfile.ZipFile(D / f"zips/F2_{k}_{v}.zip").namelist()) for v in VARS]
        if any(x != sets[0] for x in sets):
            stop(f"{k}: tile ids differ across variants")
        if sorted(p.stem for p in (D / f"zips/retgt_{s}").glob("*.png")) != sets[0]:
            stop(f"{k}: retgt ids differ")
        ids[k] = sets[0]
    info["gates"]["tile_ids"] = {k: len(v) for k, v in ids.items()}
    print("gate tile ids ok", info["gates"]["tile_ids"], flush=True)
    # gate 3: R0 mask == retrain-1 evalpack cell_masks
    for k, s in SETS.items():
        want = {f"orion-{s}/exp_channels/cell_masks/{t}.png": t for t in ids[k]}
        seen = 0
        with tarfile.open(PACKS / f"evalpack_{s}.tar") as tf:
            for m in tf:
                if m.name in want:
                    a = raw_bytes(tf.extractfile(m).read())
                    b = np.asarray(Image.open(S3 / f"packs/R0/orion-{s}/exp_channels/cell_masks/{want[m.name]}.png"))
                    if a.shape != b.shape or not np.array_equal(a, b):
                        stop(f"R0 mask != evalpack {s} {want[m.name]}")
                    seen += 1
        if seen != len(want):
            stop(f"evalpack {s}: found {seen} of {len(want)} masks")
    info["gates"]["r0_eq_evalpack"] = {k: len(v) for k, v in ids.items()}
    print("gate R0 == evalpack ok", flush=True)
    # gate 4: hematoxylin code path reproduces truenuc (real H&E vs L1, t16) on 8 CRC16 tiles
    cfg = json.load(open(TN / "cfg.json"))
    rows = list(csv.DictReader(open(TN / "tiles.csv")))[:8]
    # truenuc part A evaluated pix() on hod.astype(float32) (tile_one returns float32); its part B (generated images)
    # and this script use float64. The gate replicates part A exactly (float32); the float64 difference is reported.
    md, md64 = 0.0, 0.0
    for r in rows:
        t = r["tile"]
        he = np.asarray(Image.open(Path(cfg["he"]) / f"{t}.png").convert("RGB"))
        fg = png_bin(np.asarray(Image.open(Path(cfg["l1"]) / f"{t}.png")))
        h64 = hema(he)
        for got, acc in ((pix(h64.astype(np.float32), fg, T16), "32"), (pix(h64, fg, T16), "64")):
            for x, key in zip(got, ("a", "b", "c")):
                d = abs(x - float(r[f"L1_{key}_t16"]))
                if acc == "32":
                    md = max(md, d)
                else:
                    md64 = max(md64, d)
    if not md <= 1e-9:
        stop(f"truenuc reproduction (float32, as truenuc part A) max diff {md}; float64 {md64}")
    info["gates"]["truenuc_repro_maxdiff_float32"] = md
    info["gates"]["truenuc_repro_maxdiff_float64"] = md64
    print("gate truenuc reproduction ok", md, "float64 diff", md64, flush=True)
    # readouts
    for k, s in SETS.items():
        with mp.get_context("fork").Pool(int(nproc)) as p:
            res = [r for chunk in p.imap(job, [(k, s, t) for t in ids[k]], chunksize=4) for r in chunk]
        with open(out / f"hema_{k}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
        print(k, len(res), "rows", flush=True)
    import skimage
    info["versions"] = dict(skimage=skimage.__version__, numpy=np.__version__)
    json.dump(info, open(out / "info.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
