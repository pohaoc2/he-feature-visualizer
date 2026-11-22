"""hfv_retrain3_cellvit_20261122 (PREREG §3.1-§3.2, §3.6 gate 3): the 7 cell channels from official CellViT nuclei, Hyak CPU.

  HFV=.. SEV_SRC=.. PYTHONPATH=<PixCell cb62fa7> python cells_oc.py SLIDE MASK_TIF LIN_OUT 'CVGLOB' TRAIN_TILES PREFIX \
        ID_PREFIX OUT NPROC [--eval SET TILES]...
Per tile t (x0_y0, 256 px):
  inst  = PixCell _cellvit_json_to_instance_mask(official CellViT JSON of the real H&E tile)   (contours filled; the
          function that also rasterises generated images when they are scored)
  sep   = sim-exp-validation morphology_renderer.separate_instances(inst)  (1 px line where nuclei touch; hfv_retrain2 L1)
  (D1) each nucleus of sep keeps only its largest 8-connected piece (pixels the gap cut off are cleared, counted)
  cell_masks = sep > 0, PNG 0/255
  Lin cell of a nucleus = the Lin whole-cell label (MASK_TIF) with the most pixels under the nucleus (background not
  counted; ties -> lower CellID); that cell's population / state from LIN_OUT/cells_labels.npz (ty, st: CELESTA + Lin
  Fig. 3d, labels.py dead-first state) -> cell_type_* / cell_state_*. Nuclei on no Lin cell: cell_masks only.
  CellViT's own type output is never read.
Gates (STOP, nothing renamed into place): every tile has a JSON; 8-connected components of cell_masks == surviving
instances of sep; type channels disjoint and their union == matched nuclei; same for state; values in {0, 1}.
Outputs: OUT/cells_<SLIDE>_OC.tar (PREFIX/<ch>/<ID_PREFIX><t>.png), OUT/eval_<SET>_OC.tar (exp_channels/<ch>/<t>.png),
OUT/tiles_<SLIDE>_OC.csv (per-tile counts), OUT/cells_<SLIDE>_OC.json (summary).
"""
import csv
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

SLIDE, TIF, LIN, CVROOT, TRAIN, PREFIX, IDP, OUT = sys.argv[1:9]
NPROC = int(sys.argv[9])
EVALS = [sys.argv[i + 1:i + 3] for i, a in enumerate(sys.argv) if a == "--eval"]
TILE = 256
sys.path.insert(0, os.environ["HFV"])
from stages.lin_typing import render as RD  # noqa: E402
from stages.lin_typing.labels import POP_NAMES, STATE_NAMES  # noqa: E402
from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask  # noqa: E402

rasterize, mr, channels = RD.renderer(os.environ["SEV_SRC"])
NAMES = list(channels.CHANNEL_NAMES[:7])
assert NAMES == ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
                 "cell_state_nonprolif", "cell_state_dead"], NAMES
T0 = time.time()
G = {}
COLS = ["tile", "n_json", "n_inst", "n_comp", "orphan_inst", "orphan_px", "n_unmatched", "px_inst", "px_unmatched", "lin_in", "lin_in_hit",
        "lin_in_multi"] + [f"lin_in_{p}" for p in POP_NAMES] + [f"lin_in_hit_{p}" for p in POP_NAMES] \
       + [f"nuc_{p}" for p in POP_NAMES] + [f"nuc_{s}" for s in STATE_NAMES]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def stop(m):
    print("STOP", m, flush=True)
    sys.exit(1)


def lin_window(x0, y0):
    lab = G["lab"]
    out = np.zeros((TILE, TILE), np.int64)
    ya, yb = max(y0, 0), min(y0 + TILE, lab.shape[0])
    xa, xb = max(x0, 0), min(x0 + TILE, lab.shape[1])
    out[ya - y0:yb - y0, xa - x0:xb - x0] = lab[ya:yb, xa:xb]
    return out


def draw(t):
    x0, y0 = map(int, t.split("_"))
    jp = G["json"][t]
    n_json = len(json.loads(Path(jp).read_text())["cells"])
    inst = _cellvit_json_to_instance_mask(jp, image_hw=(TILE, TILE))
    sep = mr.separate_instances(inst.astype(np.int64))
    # deviation D1 (2026-11-22, gate 3 failed on CRC33 14592_13824 and CRC36 18432_19968): the 1 px gap can cut a
    # few pixels of a nucleus off from the rest; each nucleus keeps only its largest 8-connected piece (ties -> the
    # piece found first), the cut-off pieces are cleared and counted (orphan_inst, orphan_px). Different nuclei are
    # never 8-adjacent after the gap, so every piece belongs to one nucleus.
    comp, nc = ndimage.label(sep > 0, np.ones((3, 3)))
    orphan_inst = orphan_px = 0
    if nc:
        size = np.bincount(comp.ravel())
        owner = np.zeros(nc + 1, np.int64)
        owner[comp.ravel()] = sep.ravel()
        keep = np.zeros(nc + 1, bool)
        for k in np.unique(owner[1:]):
            cs = np.flatnonzero(owner == k)
            cs = cs[cs > 0]
            keep[cs[np.argmax(size[cs])]] = True
            if len(cs) > 1:
                orphan_inst += 1
                orphan_px += int(size[cs].sum() - size[cs].max())
        sep = np.where(keep[comp], sep, 0)
    fg = sep > 0
    ids = np.unique(sep[fg])
    n_comp = int(ndimage.label(fg, np.ones((3, 3)))[1])
    lin = lin_window(x0, y0)
    # nucleus id -> Lin CellID with the most pixels (ties -> lower CellID); 0 = no Lin pixel under the nucleus
    both = fg & (lin > 0)
    match = {}
    if both.any():
        pairs, cnt = np.unique(np.stack([sep[both], lin[both]], 1), axis=0, return_counts=True)
        order = np.lexsort((pairs[:, 1], -cnt, pairs[:, 0]))   # by nucleus, then count desc, then CellID asc
        for k in order:
            match.setdefault(int(pairs[k, 0]), int(pairs[k, 1]))
    bad = sum(1 for v in match.values() if v < 1 or v > G["n"])
    out = {n: np.zeros((TILE, TILE), np.uint8) for n in NAMES}
    out["cell_masks"][fg] = 1
    nuc_pop, nuc_st = np.zeros(3, int), np.zeros(3, int)
    lut_ty = np.full(int(sep.max()) + 1, -1, np.int64)
    lut_st = np.full(int(sep.max()) + 1, -1, np.int64)
    if not bad:
        for k, c in match.items():
            lut_ty[k], lut_st[k] = G["ty"][c - 1], G["st"][c - 1]
            nuc_pop[lut_ty[k]] += 1
            nuc_st[lut_st[k]] += 1
        ty, st = lut_ty[sep], lut_st[sep]
        for j, nm in enumerate(POP_NAMES):
            out[f"cell_type_{nm}"][fg & (ty == j)] = 1
        for j, nm in enumerate(STATE_NAMES):
            out[f"cell_state_{nm}"][fg & (st == j)] = 1
    matched_px = fg & (lut_ty[sep] >= 0)
    tsum = out["cell_type_healthy"].astype(int) + out["cell_type_cancer"] + out["cell_type_immune"]
    ssum = out["cell_state_prolif"].astype(int) + out["cell_state_nonprolif"] + out["cell_state_dead"]
    part = bool(tsum.max(initial=0) <= 1 and ssum.max(initial=0) <= 1 and np.array_equal(tsum > 0, matched_px)
                and np.array_equal(ssum > 0, matched_px))
    # Lin cells whose centroid is inside the tile, and how many of them were chosen by >= 1 nucleus
    i = G["table"].near(x0, y0, reach=0)
    cid = G["table"].id[i] if len(i) else np.zeros(0, np.int64)
    chosen = np.array(list(match.values()), int)
    hits = np.array([(chosen == c).sum() for c in cid], int)
    pop_in = G["ty"][cid - 1] if len(cid) else np.zeros(0, int)
    row = dict(tile=t, n_json=n_json, n_inst=len(ids), n_comp=n_comp, orphan_inst=orphan_inst, orphan_px=orphan_px,
               n_unmatched=len(ids) - len(match),
               px_inst=int(fg.sum()), px_unmatched=int((fg & ~matched_px).sum()), lin_in=len(cid),
               lin_in_hit=int((hits > 0).sum()), lin_in_multi=int((hits > 1).sum()))
    for j, p in enumerate(POP_NAMES):
        row[f"lin_in_{p}"] = int((pop_in == j).sum())
        row[f"lin_in_hit_{p}"] = int(((pop_in == j) & (hits > 0)).sum())
        row[f"nuc_{p}"] = int(nuc_pop[j])
    for j, s in enumerate(STATE_NAMES):
        row[f"nuc_{s}"] = int(nuc_st[j])
    pngs = [cv2.imencode(".png", out[n] * 255)[1].tobytes() for n in NAMES]
    return t, pngs, row, bad, part


def build(tiles, tarpath, pre, idp):
    rows = {}
    with tarfile.open(tarpath + ".part", "w") as tar, mp.get_context("fork").Pool(NPROC) as p:
        for t, pngs, row, bad, part in p.imap(draw, tiles, chunksize=8):
            if bad or not part or row["n_comp"] != row["n_inst"]:
                stop(f"tile {t}: labels outside cells_labels {bad}, partition {part}, components {row['n_comp']} "
                     f"vs instances {row['n_inst']}")
            for n, by in zip(NAMES, pngs):
                ti = tarfile.TarInfo(f"{pre}/{n}/{idp}{t}.png")
                ti.size, ti.mtime = len(by), 0
                tar.addfile(ti, io.BytesIO(by))
            rows[t] = row
    return rows


def summary(rows):
    s = {k: int(sum(r[k] for r in rows)) for k in COLS[1:]}
    s["n_tiles"] = len(rows)
    s["unmatched_frac_count"] = s["n_unmatched"] / max(s["n_inst"], 1)
    s["unmatched_frac_px"] = s["px_unmatched"] / max(s["px_inst"], 1)
    s["lin_without_nucleus_frac"] = 1 - s["lin_in_hit"] / max(s["lin_in"], 1)
    s["lin_without_nucleus_frac_by_pop"] = {p: 1 - s[f"lin_in_hit_{p}"] / max(s[f"lin_in_{p}"], 1) for p in POP_NAMES}
    s["lin_multi_frac"] = s["lin_in_multi"] / max(s["lin_in"], 1)
    s["lost_in_gap"] = s["n_json"] - s["n_inst"]
    s["nuclei_per_tile_median"] = float(np.median([r["n_inst"] for r in rows]))
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = "_OC" if TRAIN != "-" else "_OC_evalonly"   # TRAIN "-" = evaluation sets only (D2: all CRC33 pack channel tiles)
    base = f"{OUT}/cells_{SLIDE}{tag}"
    if os.path.exists(base + ".json"):
        stop(f"{base}.json exists")
    import glob
    js = sorted(glob.glob(f"{CVROOT}/*.json"))   # CVROOT is a glob of chunk dirs of this slide only (x_y names repeat across slides)
    G["json"] = {Path(p).stem: p for p in js}
    if len(G["json"]) != len(js):
        stop("a tile has two CellViT JSONs")
    c = np.load(f"{LIN}/cells_labels.npz")
    if not np.array_equal(c["cell_id"], np.arange(1, len(c["cell_id"]) + 1)):
        stop("cell_id is not 1..N in label order")
    G.update(n=len(c["cell_id"]), ty=c["ty"].astype(np.int64), st=c["st"].astype(np.int64),
             table=RD.CellTable(c["cell_id"], c["x"], c["y"], c["ty"], c["st"], np.ones(len(c["cell_id"]))))
    import tifffile
    G["lab"] = tifffile.imread(TIF)
    log("mask", G["lab"].shape, G["lab"].dtype, int(G["lab"].max()), "json", len(G["json"]))
    if int(G["lab"].max()) > G["n"]:
        stop("mask labels above N")
    train = open(TRAIN).read().split() if TRAIN != "-" else []
    sets = ([("train", train, PREFIX, IDP, base)] if train else []) + [(s, open(f).read().split(), "exp_channels", "", f"{OUT}/eval_{s}_OC")
                                                   for s, f in EVALS]
    for name, tiles, _, _, _ in sets:
        miss = [t for t in tiles if t not in G["json"]]
        if miss:
            stop(f"{name}: {len(miss)} tiles without CellViT JSON, e.g. {miss[:3]}")
    rep = dict(slide=SLIDE, sev=open(os.path.join(os.environ["SEV_SRC"], "..", "COMMIT")).read().strip(), sets={})
    allrows = {}
    for name, tiles, pre, idp, tp in sets:
        rows = build(tiles, tp + ".tar", pre, idp)
        if len(rows) != len(tiles):
            stop(f"{name} tiles missing")
        os.replace(tp + ".tar.part", tp + ".tar")
        rep["sets"][name] = dict(tar=tp + ".tar", tar_bytes=os.path.getsize(tp + ".tar"), **summary(list(rows.values())))
        log(name, json.dumps(rep["sets"][name]))
        allrows |= rows
    with open(f"{OUT}/tiles_{SLIDE}{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for t in sorted(allrows):
            w.writerow(allrows[t])
    rep["t_total_s"] = round(time.time() - T0)
    json.dump(rep, open(base + ".json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
