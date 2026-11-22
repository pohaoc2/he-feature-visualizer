"""hfv_retrain2_20261121 PREREG §8: L1b = the sim2real renderer path of R0 (stages/lin_typing/render.tile_agents: Lin
centroids, library outlines NucleusShapes.pick(type, target_px, CellID)) with render_crop(separate_touching=True).

  python l1b.py SET LIN_OUT LIB REALCV_DIR SD_DIR PACK_DIR DIAG_TILES_CSV OUT NPROC
Gate per tile: the same call with separate_touching=False == the eval-pack R0 cell_masks png (every channel value of
cell_masks; the png is the 0/255 image write_channel_dir wrote). Any mismatch -> STOP.
Scores (diag.py rule): binarise, relabel 8-connected, PixCell _compute_pq against CellViT and StarDist of the real H&E.
"""
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag as D  # noqa: E402

SET, LIN, LIB, REALCV, SDD, PACK, TCSV, OUT = sys.argv[1:9]
NPROC = int(sys.argv[9])
G = {}


def init():
    sys.path.insert(0, os.environ["HFV"])
    from stages.lin_typing import render as RD
    rasterize, mr, _ = RD.renderer(os.environ["SEV_SRC"])
    c = np.load(f"{LIN}/cells_labels.npz")
    n = np.load(f"{LIN}/cells_nuclei.npz")
    if not np.array_equal(c["cell_id"], n["cell_id"]):
        D_stop("cells_nuclei order")
    G.update(RD=RD, mr=mr, lib=rasterize.NucleusShapes.load(LIB),
             table=RD.CellTable(c["cell_id"], c["x"], c["y"], c["ty"], c["st"], n["target_px"]))
    sd = {}
    for f in sorted(Path(SDD).glob("sd_*.npz")):
        z = np.load(f)
        for k, rel in enumerate(z["rels"]):
            if rel.startswith(f"real/{SET}/"):
                sd[Path(rel).stem] = z[f"i{k}"].astype(np.int32)
    G["sd"] = sd


def D_stop(m):
    print("STOP", m, flush=True)
    sys.exit(1)


def one(t):
    from PIL import Image
    from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq
    x0, y0 = map(int, t.split("_"))
    agents, pos = G["RD"].tile_agents(G["table"], x0, y0, G["lib"])
    kw = dict(shape="outline", placement="jitter", allow_contact=True)
    base, _ = G["mr"].render_crop(agents, pos, y0, x0, **kw)
    sep, inst = G["mr"].render_crop(agents, pos, y0, x0, separate_touching=True, **kw)
    r0 = np.asarray(Image.open(Path(PACK) / "exp_channels" / "cell_masks" / f"{t}.png"))
    r0 = r0[..., 0] if r0.ndim == 3 else r0
    gate = bool(np.array_equal(r0 > 0, base[0] > 0)) and bool(np.array_equal(np.unique(r0), np.unique((base[0] > 0) * 255)))
    cv = _cellvit_json_to_instance_mask(Path(REALCV) / f"{t}.json", image_hw=(256, 256))
    sd = G["sd"][t]
    m = D.relabel(sep[0] > 0)
    m0 = D.relabel(base[0] > 0)
    return dict(set=SET, tile=t, gate_R0=gate, pq_L1b_cv=_compute_pq(cv, m)[2], pq_L1b_sd=_compute_pq(sd, m)[2],
                pq_R0re_cv=_compute_pq(cv, m0)[2], pq_R0re_sd=_compute_pq(sd, m0)[2], n_L1b=int(m.max()), n_R0re=int(m0.max()),
                n_agents=len(inst), px_lost=int(base[0].sum() - sep[0].sum()))


def main():
    os.makedirs(OUT, exist_ok=False)
    tiles = [r["tile"] for r in csv.DictReader(open(TCSV))]
    init()
    with mp.get_context("fork").Pool(NPROC) as p:
        rows = p.map(one, tiles, chunksize=4)
    with open(f"{OUT}/l1b.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r["tile"] for r in rows if not r["gate_R0"]]
    print(SET, "tiles", len(rows), "R0 gate mismatches", len(bad), bad[:5], flush=True)
    if bad:
        D_stop("R0 reproduction gate")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
