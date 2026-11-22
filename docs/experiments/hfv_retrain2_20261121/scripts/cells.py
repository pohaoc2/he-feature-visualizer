"""hfv_retrain2_20261121 stage 2: the 7 cell channels of the new training masks (PREREG.md §7-§8), Hyak CPU.

  HFV=.. SEV_SRC=.. python cells.py MODE SLIDE MASK_TIF LIN_OUT LIB TRAIN_TILES PREFIX ID_PREFIX OUT NPROC \
        [--eval SET TILES_CSV CHECK_CSV CHECK_COL]...
MODE
  L1   every Lin cell drawn with its own loose nucleus (stages/lin_typing/nuclei.erode_labels on the tile + 8 px margin,
       zero padding outside the slide = diag.py window), then separate_instances (1 px line where nuclei touch);
       type / state from cells_labels.npz (ty, st) of each CellID
  L1b  the sim2real renderer path of R0 (render.tile_agents: Lin centroids, library outlines picked by
       NucleusShapes.pick(type, target_px, CellID)) with render_crop(allow_contact=True, separate_touching=True)
Channels = channels.CHANNEL_NAMES[:7], PNG 0/255 as rasterize.write_channel_dir writes them.
Outputs: OUT/cells_<SLIDE>_<MODE>.tar (PREFIX/<ch>/<ID_PREFIX><tile>.png, training tiles), OUT/eval_<SET>_<MODE>.tar
(exp_channels/<ch>/<tile>.png), OUT/cells_<SLIDE>_<MODE>.json, OUT/tiles_<SLIDE>_<MODE>.npz (per-tile fractions).
Gates (exit 1, nothing renamed into place):
  every label in a tile is a CellID of cells_labels; type and state channels are disjoint covers of cell_masks;
  every training tile written with 7 channels;
  eval tiles: 8-connected component count of cell_masks == CHECK_COL of CHECK_CSV (diag.py n_L1 / l1b.py n_L1b), i.e.
  the builder draws exactly the masks that were scored.
"""
import csv
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time

import cv2
import numpy as np
from scipy import ndimage

MODE, SLIDE, TIF, LIN, LIB, TRAIN, PREFIX, IDP, OUT = sys.argv[1:10]
NPROC = int(sys.argv[10])
EVALS = []
for i, a in enumerate(sys.argv):
    if a == "--eval":
        EVALS.append(sys.argv[i + 1:i + 5])
assert MODE in ("L1", "L1b")
TILE, M = 256, 8
sys.path.insert(0, os.environ["HFV"])
from stages.lin_typing import nuclei as NU  # noqa: E402
from stages.lin_typing import render as RD  # noqa: E402
from stages.lin_typing.labels import POP_NAMES, STATE_NAMES  # noqa: E402

rasterize, mr, channels = RD.renderer(os.environ["SEV_SRC"])
NAMES = list(channels.CHANNEL_NAMES[:7])
T0 = time.time()
G = {}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def stop(m):
    print("STOP", m, flush=True)
    sys.exit(1)


def window(x0, y0):
    lab = G["lab"]
    out = np.zeros((TILE + 2 * M, TILE + 2 * M), np.int64)
    ya, yb = max(y0 - M, 0), min(y0 + TILE + M, lab.shape[0])
    xa, xb = max(x0 - M, 0), min(x0 + TILE + M, lab.shape[1])
    out[ya - (y0 - M):yb - (y0 - M), xa - (x0 - M):xb - (x0 - M)] = lab[ya:yb, xa:xb]
    return out


def draw(t):
    x0, y0 = map(int, t.split("_"))
    bad = 0
    if MODE == "L1":
        _, loose = NU.erode_labels(window(x0, y0))
        sep = mr.separate_instances(loose[M:-M, M:-M].astype(np.int64))
        ids = np.unique(sep[sep > 0])
        bad = int(np.sum((ids < 1) | (ids > G["n"])))
        out = {n: np.zeros((TILE, TILE), np.float32) for n in NAMES}
        out["cell_masks"][sep > 0] = 1.0
        if not bad:
            k = np.where(sep > 0, sep - 1, 0)
            fg = sep > 0
            ty, st = G["ty"][k], G["st"][k]
            for j, nm in enumerate(POP_NAMES):
                out[f"cell_type_{nm}"][fg & (ty == j)] = 1.0
            for j, nm in enumerate(STATE_NAMES):
                out[f"cell_state_{nm}"][fg & (st == j)] = 1.0
        ten = np.stack([out[n] for n in NAMES])
    else:
        agents, pos = RD.tile_agents(G["table"], x0, y0, G["lib"])
        ten, _ = mr.render_crop(agents, pos, y0, x0, shape="outline", placement="jitter", allow_contact=True,
                                separate_touching=True)
    part = bool(np.array_equal(ten[1] + ten[2] + ten[3], ten[0]) and np.array_equal(ten[4] + ten[5] + ten[6], ten[0])
                and set(np.unique(ten).tolist()) <= {0.0, 1.0})
    ncomp = ndimage.label(ten[0] > 0, np.ones((3, 3)))[1]
    pngs = [cv2.imencode(".png", (a * 255).astype(np.uint8))[1].tobytes() for a in ten]
    # Lin cells whose centroid is inside the tile, and how many of them have >= 1 pixel in cell_masks
    i = G["table"].near(x0, y0, reach=0)
    cx = np.clip(np.round(G["table"].x[i]).astype(int) - x0, 0, TILE - 1)
    cy = np.clip(np.round(G["table"].y[i]).astype(int) - y0, 0, TILE - 1)
    return t, pngs, ten.reshape(7, -1).mean(1), ncomp, bad, part, len(i), int((ten[0][cy, cx] > 0).sum())


def build(tiles, tarpath, pre, idp):
    rows = {}
    with tarfile.open(tarpath + ".part", "w") as tar, mp.get_context("fork").Pool(NPROC) as p:
        for t, pngs, frac, ncomp, bad, part, nin, nhit in p.imap(draw, tiles, chunksize=8):
            if bad or not part:
                stop(f"tile {t}: labels outside cells_labels {bad}, partition {part}")
            for n, by in zip(NAMES, pngs):
                ti = tarfile.TarInfo(f"{pre}/{n}/{idp}{t}.png")
                ti.size, ti.mtime = len(by), 0
                tar.addfile(ti, io.BytesIO(by))
            rows[t] = (frac, ncomp, nin, nhit)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    base = f"{OUT}/cells_{SLIDE}_{MODE}"
    if os.path.exists(base + ".json"):
        stop(f"{base}.json exists")
    c = np.load(f"{LIN}/cells_labels.npz")
    if os.path.exists(f"{LIN}/cells_nuclei.npz"):
        nuc = dict(np.load(f"{LIN}/cells_nuclei.npz"))
    elif MODE == "L1":   # L1 draws from the mask; target_px only feeds L1b (e.g. msi s3b_crc16b has no cells_nuclei.npz)
        nuc = dict(cell_id=c["cell_id"], target_px=np.ones(len(c["cell_id"])))
    else:
        stop("L1b needs cells_nuclei.npz")
    if not (np.array_equal(c["cell_id"], np.arange(1, len(c["cell_id"]) + 1)) and np.array_equal(c["cell_id"], nuc["cell_id"])):
        stop("cell_id is not 1..N in label order")
    G.update(n=len(c["cell_id"]), ty=c["ty"].astype(int), st=c["st"].astype(int),
             table=RD.CellTable(c["cell_id"], c["x"], c["y"], c["ty"], c["st"], nuc["target_px"]))
    if MODE == "L1":
        import tifffile
        G["lab"] = tifffile.imread(TIF)
        log("mask", G["lab"].shape, G["lab"].dtype, int(G["lab"].max()))
        if int(G["lab"].max()) > G["n"]:
            stop("mask labels above N")
    else:
        G["lib"] = rasterize.NucleusShapes.load(LIB)
    train = open(TRAIN).read().split()
    rep = dict(mode=MODE, slide=SLIDE, sev=open(os.path.join(os.environ["SEV_SRC"], "..", "COMMIT")).read().strip(),
               n_train=len(train), margin_px=M if MODE == "L1" else None, lib=LIB if MODE == "L1b" else None)
    rows = build(train, base + ".tar", PREFIX, IDP)
    if len(rows) != len(train):
        stop("train tiles missing")
    os.replace(base + ".tar.part", base + ".tar")
    fr = np.array([rows[t][0] for t in train])
    rep["train"] = dict(tar=base + ".tar", tar_bytes=os.path.getsize(base + ".tar"),
                        mean_frac=dict(zip(NAMES, fr.mean(0).round(5).tolist())),
                        components_per_tile_median=float(np.median([rows[t][1] for t in train])),
                        lin_centroids_in_tile=int(sum(rows[t][2] for t in train)),
                        lin_centroids_on_mask_frac=sum(rows[t][3] for t in train) / max(sum(rows[t][2] for t in train), 1))
    log("train", json.dumps(rep["train"]))
    np.savez_compressed(f"{OUT}/tiles_{SLIDE}_{MODE}.npz", tile=np.array(train), frac=fr, names=np.array(NAMES),
                        ncomp=np.array([rows[t][1] for t in train]))
    rep["eval"] = {}
    for set_, tcsv, ccsv, col in EVALS:
        tiles = [r["tile"] for r in csv.DictReader(open(tcsv))]
        ref = {r["tile"]: int(r[col]) for r in csv.DictReader(open(ccsv))}
        et = f"{OUT}/eval_{set_}_{MODE}"
        er = build(tiles, et + ".tar", "exp_channels", "")
        mism = [t for t in tiles if er[t][1] != ref[t]]
        efr = np.array([er[t][0] for t in tiles])
        rep["eval"][set_] = dict(n=len(tiles), check=f"{os.path.basename(ccsv)}:{col}", component_mismatch=len(mism),
                                 examples=mism[:10], mean_frac=dict(zip(NAMES, efr.mean(0).round(5).tolist())),
                                 lin_centroids_on_mask_frac=sum(er[t][3] for t in tiles) / max(sum(er[t][2] for t in tiles), 1))
        log("eval", set_, json.dumps(rep["eval"][set_]))
        if mism:
            stop(f"eval {set_} component gate")
        os.replace(et + ".tar.part", et + ".tar")
    rep["t_total_s"] = round(time.time() - T0)
    json.dump(rep, open(base + ".json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
