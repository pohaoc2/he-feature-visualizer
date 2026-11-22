"""MSI stage 3 arm B (prereg_stage3_20261120.md §5): CRC16 and CRC01 window channels through ONE path, the hfv Lin-cell
renderer of the CRC01 run (hfv_crc01_20261118: tiles.py, labels_crc01.py, nuclei_render_crc01.py; the frozen hfv and
sim-exp-validation copies there). Every rule is copied from those scripts with slide paths as parameters; on CRC01 each step
must reproduce the existing CRC01 output exactly (gate), so the CRC16 output is the same program on another slide.

  windows SLIDE S                 S/windows.json: arm-B window list (name, arm-B index, central origin px (y, x), 6 x 6 tiles)
                                   CRC16 = w220-w231 (stage-1 tcmvp2cel windows, index 0-11); CRC01 = stage-0 windows
                                   3, 13, 30, 14, 18, 4, 17, 28, 27, 7, 26, 9 (descending TUM, no central-block overlap;
                                   index 12-23, names c<stage-0 index>)
  tiles SLIDE S NPROC             tiles.py rules: hfv patchify tissue tiles (overview every 64 px, tissue_mask_hsv,
                                   get_tissue_patches 256 / 256 / 0.1), H&E tile PNG = 256 px crop of Lin's registered H&E,
                                   ink = style_pool.pigment_fraction > 0 -> S/tiles.json, S/he/<tile>.png (window tiles)
                                   CRC01 gate: tile list and ink equal hfv_crc01 out/tiles_crc01.json; window PNGs equal the
                                   crc01_he_tiles.tar members byte for byte
  labels SLIDE S                  labels_crc01.py rules (envsmc python): CELESTA k-d tiles + Lin Fig. 3d fill-in at the Lin
                                   class of the centroid; Ki-67 > the slide's Lin gate -> prolif; no dead (no CellViT run on
                                   CRC01; same on CRC16 so both slides share the rule); kept tiles = tissue - ink
                                   -> S/cells_labels.npz, S/tiles_keep.txt; CRC01 gate: equal to out/cells_labels.npz
  render SLIDE S NPROC            nuclei_render_crc01.py rules (NUC loose): target area = loose 5 px erosion of the Lin cell
                                   (>= 1 px); library = Lin nuclei of 400 kept tiles drawn with default_rng(20261118);
                                   render_tile (render_crop outline / jitter / allow_contact) for the 36 tiles of every
                                   window. Instance map (new, for the edit unit gates): every rendered pixel gets the CellID of
                                   the first agent (tile row-major, render order) that covers it AND carries the pixel's final
                                   type (the render's type partition: healthy > cancer > immune); gate lab > 0 == channel 0.
                                   Erosion areas come from a crop of the window +- 96 px; a needed cell with a pixel within
                                   8 px of a crop side that is not the image edge -> STOP.
                                   CRC01 gates: library arrays equal out/nucleus_shapes_lin_crc01.npz, target areas equal
                                   out/cells_nuclei.npz, rendered window tiles equal crc01_exp_channels.tar (kept tiles)
                                   -> S/render/<w>.npz (tiles, ch uint8 (7, 1536, 1536), lab int64 (1536, 1536)), S/render.json
  render_l1 SLIDE S OUT ENV NPROC  hfv_retrain2_20261121 PREREG §7 (L1, user decision 2026-11-21): render_tile(true_shapes=True)
                                   = every Lin cell with its own loose nucleus (nuclei.erode_labels on the tile + 8 px margin,
                                   zero padding outside the slide), separate_instances (1 px line where nuclei touch), type /
                                   state from S/cells_labels.npz; the rule of hfv_retrain2 scripts/cells.py L1 (training tiles).
                                   ENV = a root whose exp_channels/{vasculature,oxygen,glucose}/<tile>.npy are byte-copied
                                   (app 8 kin/roots/orion-crc16e = s3b method C ch7-9).
                                   -> OUT/render/<w>.npz (tiles, ch uint8 (7, 1536, 1536), lab int64 = L1 instance map, CellID
                                   per pixel, tile by tile), OUT/exp_channels/<10 ch>/<tile>.(png|npy) and OUT/labels/<tile>.npy
                                   (int64 (256, 256)) for the 16 central tiles of every window, OUT/tiles.txt, OUT/manifest.json
                                   (sha256 per file), OUT/render_l1.json
                                   Gates: default render_tile (true_shapes=False) of every window tile == S/render/<w>.npz ch
                                   (byte-identical default path); every L1 label is a CellID; lab > 0 == ch 0; type and state
                                   channels are disjoint covers of ch 0 and equal cells_labels ty / st of the pixel's CellID
"""
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
import types
from pathlib import Path

import numpy as np

CC = "/gscratch/cheme/chiu"
C16 = f"{CC}/crc16-20261024"
N01 = f"{CC}/hfv_crc01_20261118"
HFV, SEV = f"{N01}/hfv", f"{N01}/sev/src"
# render_l1 only: S3B_HFV / S3B_SEV point at the hfv / sim-exp-validation copies of the hfv_retrain2 L1 build (separate_instances);
# its default-path gate checks that those copies still reproduce S/render byte for byte
if os.environ.get("S3B_L1") == "1":
    HFV, SEV = os.environ.get("S3B_HFV", HFV), os.environ.get("S3B_SEV", SEV)
SLIDES = {
    "CRC16": dict(mask=f"{C16}/p1t_prior_20261106/c723/lin/P37_S48-CRC16-mask.ome.tif",
                  csv=f"{C16}/pick/P37_S48-CRC16.csv",
                  classmap=f"{CC}/he_celltype_audit_20261118/lin/CRC16_lin_he_class.tiff",
                  celesta=f"{C16}/celltype_pub_20261118/celesta/out", n_celesta=32,
                  gates=f"{C16}/lin_reinit_20261106/lin_gates.json",
                  he="lin/CRC16__19510_C16_US_SCAN_OR_001__152020-registered.ome.tif", ref=None),
    "CRC01": dict(mask=f"{N01}/lin/data__CRC01__P37_S29-CRC01-mask.ome.tif",
                  csv=f"{N01}/lin/data__CRC01__P37_S29-CRC01.csv",
                  classmap=f"{N01}/lin/classmaps/CRC01_lin_he_class.tiff",
                  celesta=f"{N01}/celesta/out", n_celesta=16,
                  gates=f"{N01}/out/lin_gates_crc01.json",
                  he=f"{N01}/lin/data__CRC01__18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif", ref=f"{N01}/out"),
}
CRC16_CELLS = f"{C16}/tcmvp2cel_20261118/cells"
CRC16_WINDOWS = [f"w{i}" for i in range(220, 232)]
CRC01_STAGE0 = [3, 13, 30, 14, 18, 4, 17, 28, 27, 7, 26, 9]
PS, STRIDE, TMIN, DS = 256, 256, 0.1, 64          # tiles.py
T = 256
WIN = 6 * T
M, BAND = 96, 8                                   # erosion crop margin / border band (px)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def wj(p, o):
    Path(p).write_text(json.dumps(o, indent=1) + "\n")


def win_tiles(cy, cx):
    """6 x 6 tile ids "x_y" of the window whose central 4 x 4 block starts at (cy, cx) px."""
    return [[f"{cx - T + T * c}_{cy - T + T * r}" for c in range(6)] for r in range(6)]


# ---- windows ----------------------------------------------------------------------------------------------------
def cmd_windows(slide, S):
    S = Path(S)
    out = []
    if slide == "CRC16":
        for i, w in enumerate(CRC16_WINDOWS):
            with np.load(f"{CRC16_CELLS}/windows/{w}.npz") as z:
                r0, c0 = json.loads(str(z["meta"]))["central_origin_tile"]
                tiles = z["tiles"].astype(str).tolist()
            out.append(dict(name=w, index=i, source=f"tcmvp2cel {w}", central_origin_px_yx=[r0 * T, c0 * T], tiles=tiles))
            assert win_tiles(r0 * T, c0 * T) == tiles, w
    else:
        W = json.loads(Path(f"{CC}/msi_perturb_20261119/code/windows_stage0.json").read_text())["windows"]["CRC01"]
        for k, i in enumerate(CRC01_STAGE0):
            cy, cx = W[i]["central_origin_px_yx"]
            out.append(dict(name=f"c{i:02d}", index=12 + k, source=f"stage-0 CRC01 window {i}", central_origin_px_yx=[cy, cx],
                            tiles=win_tiles(cy, cx)))
    cen = [t for w in out for r in w["tiles"][1:5] for t in r[1:5]]
    assert len(cen) == len(set(cen)) == 16 * len(out), "central tiles overlap"
    wj(S / "windows.json", dict(slide=slide, rule="msi_s3b_render.py windows", windows=out))
    log(slide, [w["name"] for w in out])


# ---- tiles (tiles.py) -------------------------------------------------------------------------------------------
def load_he(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr = s.levels[0].asarray()
        ax = s.axes.upper()
    if ax.startswith("C") or ax.startswith("S"):
        arr = np.moveaxis(arr, 0, -1)
    assert arr.ndim == 3 and arr.shape[2] >= 3, (ax, arr.shape)
    return np.ascontiguousarray(arr[..., :3])


_HE = {}


def _ink(args):
    import cv2
    from sim_exp_validation import style_pool
    t, keep_png, tmp = args
    x0, y0 = map(int, t.split("_"))
    p = f"{tmp}/{t}.png"
    cv2.imwrite(p, cv2.cvtColor(_HE["a"][y0:y0 + PS, x0:x0 + PS], cv2.COLOR_RGB2BGR))
    f = style_pool.pigment_fraction(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    if keep_png:   # TMPDIR is node-local: copy the bytes across file systems, then drop the temporary file
        shutil.copyfile(p, f"{keep_png}/{t}.png")
        os.remove(p)
    else:
        os.remove(p)
    return t, f


def cmd_tiles(slide, S, nproc):
    sys.modules.setdefault("zarr", types.ModuleType("zarr"))   # tiles.py: stub only satisfies utils.ome's import
    sys.path.insert(0, HFV)
    sys.path.insert(0, SEV)
    import tifffile
    from stages.patchify_lib.masking import tissue_mask_hsv
    from stages.patchify_lib.readers import get_tissue_patches
    from utils.normalize import percentile_to_uint8
    from sim_exp_validation import style_pool
    S = Path(S)
    cfg = SLIDES[slide]
    hp = cfg["he"] if cfg["he"].startswith("/") else str(S / cfg["he"])
    he = load_he(hp)
    with tifffile.TiffFile(cfg["mask"]) as tf:
        mshape = tf.series[0].levels[0].shape
    log(slide, "H&E", he.shape, "mask", mshape)
    assert tuple(he.shape[:2]) == tuple(mshape[-2:]), "STOP: H&E and mask sizes differ"
    h, w = he.shape[:2]
    ov = he[0:(h // DS) * DS:DS, 0:(w // DS) * DS:DS, :3]
    if ov.dtype != np.uint8:
        ov = percentile_to_uint8(ov)
    mask = tissue_mask_hsv(np.ascontiguousarray(ov))
    tiles = [f"{x}_{y}" for x, y in get_tissue_patches(mask, w, h, PS, STRIDE, TMIN, DS)]
    W = json.loads((S / "windows.json").read_text())["windows"]
    wt = sorted({t for x in W for r in x["tiles"] for t in r})
    (S / "he").mkdir()
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"s3bt_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    os.makedirs(tmp)
    _HE["a"] = he
    ws, ts = set(wt), set(tiles)
    todo = [(t, str(S / "he") if t in ws else None, tmp) for t in tiles]
    todo += [(t, str(S / "he"), tmp) for t in wt if t not in ts]   # window tiles outside tissue: PNG only
    with mp.get_context("fork").Pool(nproc) as pool:
        res = dict(pool.map(_ink, todo, chunksize=64))
    os.rmdir(tmp)
    ink = {t: res[t] for t in tiles}
    inked = sorted(t for t, v in ink.items() if v > style_pool.MAX_PIGMENT_FRACTION)
    wj(S / "tiles.json", dict(tiles=tiles, ink=ink))
    rep = dict(slide=slide, he=hp, he_shape=list(he.shape), overview_tissue_frac=round(float(mask.mean()), 4), tiles=len(tiles),
               ink_tiles=len(inked), keep=len(tiles) - len(inked), window_tiles=len(wt),
               window_tiles_not_tissue=sorted(set(wt) - set(tiles)), window_tiles_ink=sorted(set(wt) & set(inked)))
    if slide == "CRC01":
        ref = json.loads(Path(f"{N01}/out/tiles_crc01.json").read_text())
        rep["gate_tiles_equal"] = ref["tiles"] == tiles
        rep["gate_ink_equal"] = ref["ink"] == ink
        bad = []
        with tarfile.open(f"{N01}/data/crc01_he_tiles.tar") as tf:
            for t in wt:
                if t in ts:
                    bad += [t] if tf.extractfile(f"he/{t}.png").read() != (S / "he" / f"{t}.png").read_bytes() else []
        rep["gate_png_bytes_differ"] = bad
        ok = rep["gate_tiles_equal"] and rep["gate_ink_equal"] and not bad
    else:   # informative: the stage-1 CRC16 H&E tiles (arm A raw inputs)
        diff, miss = [], []
        for t in wt:
            p = Path(CRC16_CELLS) / "he" / f"{t}.png"
            if not p.exists():
                miss.append(t)
                continue
            import cv2
            if not np.array_equal(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.imread(str(S / "he" / f"{t}.png"), cv2.IMREAD_COLOR)):
                diff.append(t)
        rep["stage1_he_pixels_differ"], rep["stage1_he_missing"] = diff, miss
        ok = True
    rep["ok"] = bool(ok)
    wj(S / "tiles_report.json", rep)
    log(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list) or len(v) < 20}))
    if not ok:
        raise SystemExit("STOP: tiles gate failed")


# ---- labels (labels_crc01.py; envsmc) ---------------------------------------------------------------------------
def cmd_labels(slide, S):
    import glob
    import pandas as pd
    from PIL import Image
    sys.path.insert(0, HFV)
    from stages.lin_typing import labels as LB
    from stages.lin_typing import celltype_celesta_fig3d as F
    S = Path(S)
    cfg = SLIDES[slide]
    df = pd.read_csv(cfg["csv"], usecols=["CellID", "X_centroid", "Y_centroid", "Area", "Ki67"])
    cid = df.CellID.to_numpy(np.int64)
    x, y = df.X_centroid.to_numpy(float), df.Y_centroid.to_numpy(float)
    cmap = np.array(Image.open(cfg["classmap"]))
    k = LB.lin_class_at(x, y, cmap)
    fs = sorted(glob.glob(f"{cfg['celesta']}/tile_*.csv"))
    assert len(fs) == cfg["n_celesta"], "STOP: CELESTA tiles missing"
    cel = pd.concat([pd.read_csv(f, usecols=["CellID", "final"]) for f in fs])
    assert cel.CellID.is_unique, "CELESTA tiles overlap"
    cel = cel.sort_values("CellID")
    final = F.lookup(cel.CellID.to_numpy(np.int64), cel.final.to_numpy(object), cid)
    G = json.load(open(cfg["gates"]))
    gate = float(G["markers"]["Ki67"]["final_thr"])
    dead = np.zeros(len(cid), bool)
    ty, src, st = LB.assign(final, k, df.Ki67.to_numpy(float), gate, dead)
    TT = json.loads((S / "tiles.json").read_text())   # CRC01: equal to out/tiles_crc01.json (tiles gate)
    tiles = set(TT["tiles"])
    ink = {t for t, v in TT["ink"].items() if v > 0}
    keep = tiles - ink
    tid = np.array([f"{a}_{b}" for a, b in zip((x // PS).astype(int) * PS, (y // PS).astype(int) * PS)])
    in_keep = np.isin(tid, sorted(keep))
    in_ink = np.isin(tid, sorted(ink))
    arr = dict(cell_id=cid, x=x, y=y, area_cell_px=df.Area.to_numpy(float), lin_cls=k.astype(np.int8), ty=ty, src=src, st=st,
               dead=dead, in_keep=in_keep, in_ink=in_ink, celesta=np.array([v if isinstance(v, str) else "" for v in final]))
    np.savez_compressed(S / "cells_labels.npz", **arr)
    (S / "tiles_keep.txt").write_text("\n".join(sorted(keep)) + "\n")
    rep = dict(slide=slide, n_cells=int(len(cid)), n_tiles=len(tiles), n_keep=len(keep), ki67_gate=gate,
               ki67_gate_source=G["markers"]["Ki67"]["final_source"], celesta_tiles=len(fs),
               pop_keep={LB.POP_NAMES[i]: int(((ty == i) & in_keep).sum()) for i in range(3)},
               state_keep={LB.STATE_NAMES[i]: int(((st == i) & in_keep).sum()) for i in range(3)})
    ok = True
    if slide == "CRC01":
        with np.load(f"{N01}/out/cells_labels.npz") as z:
            rep["gate_differ"] = [kk for kk in z.files if not np.array_equal(z[kk], arr[kk])] + \
                                 sorted(set(arr) ^ set(z.files))
        rep["gate_keep_equal"] = (S / "tiles_keep.txt").read_text() == Path(f"{N01}/out/tiles_keep.txt").read_text()
        ok = not rep["gate_differ"] and rep["gate_keep_equal"]
    rep["ok"] = bool(ok)
    wj(S / "labels.json", rep)
    log(json.dumps(rep))
    if not ok:
        raise SystemExit("STOP: labels gate failed")


# ---- render (nuclei_render_crc01.py) ----------------------------------------------------------------------------
def open_z(path):
    import tifffile
    import zarr
    z = zarr.open(tifffile.TiffFile(path).series[0].aszarr(), mode="r")
    return z if hasattr(z, "shape") else z["0"]


_R = {}
L1_MARGIN = 8   # hfv_retrain2 cells.py M (px): erosion window = tile + 8 px, zero padded outside the slide


def render_tile(RD, mr, table, x0, y0, library, true_shapes=False, l1=None):
    """One 256 px tile. Default: the hfv Lin-cell render (RD.render_tile; library outlines) -> (ten, inst), unchanged.
    true_shapes=True (hfv_retrain2 L1): l1 = dict(Z, H, W, NU, ty, st, pop, state, n) -> (ten float32 (7, 256, 256), labels
    int64 (256, 256)): each Lin cell's own loose nucleus with separate_instances lines, types / states by CellID."""
    if not true_shapes:
        return RD.render_tile(table, x0, y0, library, mr)
    if "Z" not in l1:      # one zarr handle per process (forked workers must not share a file offset)
        l1["Z"] = open_z(l1["mask"])
    Z, M_ = l1["Z"], L1_MARGIN
    win = np.zeros((T + 2 * M_, T + 2 * M_), np.int64)
    ya, yb = max(y0 - M_, 0), min(y0 + T + M_, l1["H"])
    xa, xb = max(x0 - M_, 0), min(x0 + T + M_, l1["W"])
    win[ya - (y0 - M_):yb - (y0 - M_), xa - (x0 - M_):xb - (x0 - M_)] = np.asarray(Z[ya:yb, xa:xb])
    _, loose = l1["NU"].erode_labels(win)
    sep = mr.separate_instances(loose[M_:-M_, M_:-M_].astype(np.int64))
    ids = np.unique(sep[sep > 0])
    if ids.size and (ids.min() < 1 or ids.max() > l1["n"]):
        raise SystemExit(f"STOP: tile {x0}_{y0} labels outside cells_labels")
    ten = np.zeros((7, T, T), np.float32)
    fg = sep > 0
    ten[0][fg] = 1.0
    k = np.where(fg, sep - 1, 0)
    ty, st = l1["ty"][k], l1["st"][k]
    for j in range(len(l1["pop"])):
        ten[1 + j][fg & (ty == j)] = 1.0
    for j in range(len(l1["state"])):
        ten[4 + j][fg & (st == j)] = 1.0
    return ten, sep


def _render_window(w):
    import cv2  # noqa: F401
    RD, mr, table, library, S = _R["RD"], _R["mr"], _R["table"], _R["library"], _R["S"]
    tiles = w["tiles"]
    ch = np.zeros((7, WIN, WIN), np.uint8)
    n_inst, clipped = 0, 0
    for r in range(6):
        for c in range(6):
            x0, y0 = map(int, tiles[r][c].split("_"))
            ten, inst = render_tile(RD, mr, table, x0, y0, library)
            assert np.isin(ten, [0, 1]).all(), (w["name"], tiles[r][c])
            ch[:, r * T:(r + 1) * T, c * T:(c + 1) * T] = ten > 0.5
            n_inst += len(inst)
            clipped += sum(i["clipped"] for i in inst)
    X0, Y0 = map(int, tiles[0][0].split("_"))
    lab = np.zeros((WIN, WIN), np.int64)
    done = set()
    order = []
    for r in range(6):
        for c in range(6):
            x0, y0 = map(int, tiles[r][c].split("_"))
            agents, positions = RD.tile_agents(table, x0, y0, library)
            for a in agents:
                if a["id"] in done:
                    continue
                done.add(a["id"])
                order.append(a["id"])
                sh = a["outline"]
                h_, w_ = sh.shape
                px, py = positions[a["id"]]["x"] - X0, positions[a["id"]]["y"] - Y0
                top, left = py - h_ // 2, px - w_ // 2           # rasterize._paint_shape geometry
                oy, ox = np.nonzero(sh)
                yy, xx = oy + top, ox + left
                m = (yy >= 0) & (yy < WIN) & (xx >= 0) & (xx < WIN)
                yy, xx = yy[m], xx[m]
                k = 1 + _R["POP"].index(a["type"])
                sel = (lab[yy, xx] == 0) & (ch[k, yy, xx] > 0)
                lab[yy[sel], xx[sel]] = a["id"]
    bad = int(((lab > 0) != (ch[0] > 0)).sum())
    np.savez_compressed(S / "render" / f"{w['name']}.npz", tiles=np.array(tiles), ch=ch, lab=lab)
    return w["name"], dict(instances_tile_sum=n_inst, clipped_tile_sum=int(clipped), agents=len(order), lab_vs_ch0_px=bad,
                           labels_on_central=int(len(np.unique(lab[T:5 * T, T:5 * T])) - 1),
                           frac=[round(float(v), 4) for v in ch.reshape(7, -1).mean(1)])


def target_table(RD, NU, W, c, Z):
    """cmd_render step 1: target area of every cell near a window tile = its loose erosion area on the window crop +- M
    (>= 1 px); a needed cell reaching the crop band -> STOP. -> (CellTable of the needed cells, ids, areas, crops)."""
    H, Wd = Z.shape
    cid = c["cell_id"]
    full = RD.CellTable(cid, c["x"], c["y"], c["ty"], c["st"], np.ones(len(cid)))
    pos = {int(v): i for i, v in enumerate(cid)}
    area_t = {}
    crops = {}
    for w in W:
        X0, Y0 = map(int, w["tiles"][0][0].split("_"))
        need = set()
        for row in w["tiles"]:
            for t in row:
                x0, y0 = map(int, t.split("_"))
                need.update(int(full.id[i]) for i in full.near(x0, y0))
        ya, yb, xa, xb = max(0, Y0 - M), min(H, Y0 + WIN + M), max(0, X0 - M), min(Wd, X0 + WIN + M)
        blk = np.asarray(Z[ya:yb, xa:xb])
        s, l = NU.erode_labels(blk)
        band = np.zeros(blk.shape, bool)
        if ya > 0:
            band[:BAND] = True
        if yb < H:
            band[-BAND:] = True
        if xa > 0:
            band[:, :BAND] = True
        if xb < Wd:
            band[:, -BAND:] = True
        cut = set(np.unique(blk[band]).tolist()) & need
        assert not cut, (w["name"], "STOP: needed cells reach the crop band", sorted(cut)[:10])
        u, n = np.unique(l, return_counts=True)
        lc = dict(zip(u.tolist(), n.tolist()))
        for v in need:
            a = float(max(lc.get(v, 0), 1.0))
            assert area_t.get(v, a) == a, (v, "area differs between crops")
            area_t[v] = a
        crops[w["name"]] = [ya, yb, xa, xb]
    ids = np.array(sorted(area_t), np.int64)
    ix = np.array([pos[int(v)] for v in ids])
    table = RD.CellTable(cid[ix], c["x"][ix], c["y"][ix], c["ty"][ix], c["st"][ix], np.array([area_t[int(v)] for v in ids]))
    return table, ids, area_t, crops


def cmd_render(slide, S, nproc):
    import cv2  # noqa: F401
    sys.path.insert(0, HFV)
    from scipy import ndimage
    from stages.lin_typing import labels as LB
    from stages.lin_typing import nuclei as NU
    from stages.lin_typing import render as RD
    rasterize, mr, channels = RD.renderer(SEV)
    NAMES = list(channels.CHANNEL_NAMES[:7])
    assert NAMES == ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
                     "cell_state_nonprolif", "cell_state_dead"], NAMES
    assert tuple(LB.POP_NAMES) == ("healthy", "cancer", "immune")
    S = Path(S)
    cfg = SLIDES[slide]
    W = json.loads((S / "windows.json").read_text())["windows"]
    c = np.load(S / "cells_labels.npz")
    cid = c["cell_id"]
    keep_all = (S / "tiles_keep.txt").read_text().split()
    Z = open_z(cfg["mask"])
    H, Wd = Z.shape
    R = dict(slide=slide, nucleus_mode="loose", hfv=HFV, sev=SEV, mask=cfg["mask"])

    # 1. target areas of the cells near the window tiles (loose erosion on a window crop)
    table, ids, area_t, crops = target_table(RD, NU, W, c, Z)
    R["crops"] = crops
    R["cells_needed"] = int(len(ids))
    log("areas", len(ids))

    # 2. library (nuclei_render_crc01.py step 2, verbatim rules)
    def region(x0, y0, halo):
        ya, xa = max(0, y0 - halo), max(0, x0 - halo)
        return np.asarray(Z[ya:min(H, y0 + 256 + halo), xa:min(Wd, x0 + 256 + halo)]), ya, xa
    rng = np.random.default_rng(20261118)
    lib_tiles = sorted(rng.choice(sorted(keep_all), size=400, replace=False).tolist())
    masks, types_, areas, srcs, lids = [], [], [], [], []
    lib_n = dict(candidates=0, empty=0, too_big=0)
    for t in lib_tiles:
        x0, y0 = map(int, t.split("_"))
        blk, ya, xa = region(x0, y0, 64)
        s, l = NU.erode_labels(blk)
        s = l
        inside = np.flatnonzero((c["x"] >= x0) & (c["x"] < x0 + 256) & (c["y"] >= y0) & (c["y"] < y0 + 256))
        u, inv = np.unique(s, return_inverse=True)
        comp = inv.reshape(s.shape)
        sl = ndimage.find_objects(comp)
        where = {int(v): j for j, v in enumerate(u)}
        for i in inside:
            lib_n["candidates"] += 1
            j = where.get(int(cid[i]))
            if j is None or j == 0:
                lib_n["empty"] += 1
                continue
            obj = comp[sl[j - 1]] == j
            st, a = NU.stamp(obj)
            if st is None:
                lib_n["too_big"] += 1
                continue
            masks.append(st); types_.append(NU.TYPE_NAMES.index(LB.POP_NAMES[c["ty"][i]])); areas.append(a)
            srcs.append(t); lids.append(int(cid[i]))
    LIB = S / f"nucleus_shapes_lin_{slide.lower()}.npz"
    NU.save_library(str(LIB), masks, types_, areas, srcs, lids)
    library = rasterize.NucleusShapes.load(str(LIB))
    types_ = np.array(types_)
    R["library"] = dict(path=str(LIB), tiles=len(lib_tiles), **lib_n, kept=len(types_),
                        per_type={nm: int((types_ == i).sum()) for i, nm in enumerate(NU.TYPE_NAMES)})
    log("library", json.dumps(R["library"]))
    gates = {}
    if slide == "CRC01":
        with np.load(LIB) as a, np.load(f"{N01}/out/nucleus_shapes_lin_crc01.npz") as b:
            gates["library_differ"] = [k for k in b.files if k not in a.files or not np.array_equal(a[k], b[k])]
        with np.load(f"{N01}/out/cells_nuclei.npz") as z:
            p2 = {int(v): i for i, v in enumerate(z["cell_id"])}
            ref = np.array([z["target_px"][p2[int(v)]] for v in ids], float)
        gates["target_area_differ"] = int((ref != np.array([area_t[int(v)] for v in ids])).sum())

    # 3. render the window tiles + instance maps
    (S / "render").mkdir()
    _R.update(RD=RD, mr=mr, table=table, library=library, S=S, POP=list(LB.POP_NAMES))
    with mp.get_context("fork").Pool(min(nproc, len(W))) as pool:
        res = dict(pool.map(_render_window, W, chunksize=1))
    R["windows"] = res
    gates["lab_vs_ch0_px"] = {k: v["lab_vs_ch0_px"] for k, v in res.items() if v["lab_vs_ch0_px"]}
    if slide == "CRC01":
        import cv2
        keep = set(keep_all)
        bad, n = [], 0
        with tarfile.open(f"{N01}/data/crc01_exp_channels.tar") as tf:
            for w in W:
                with np.load(S / "render" / f"{w['name']}.npz") as z:
                    ch = z["ch"]
                for r in range(6):
                    for cc in range(6):
                        t = w["tiles"][r][cc]
                        if t not in keep:
                            continue
                        n += 1
                        for k, nm in enumerate(NAMES):
                            a = cv2.imdecode(np.frombuffer(tf.extractfile(f"orion-crc01/exp_channels/{nm}/{t}.png").read(),
                                                           np.uint8), cv2.IMREAD_UNCHANGED) > 0
                            if not np.array_equal(a, ch[k, r * T:(r + 1) * T, cc * T:(cc + 1) * T] > 0):
                                bad.append(f"{w['name']}/{t}/{nm}")
        gates["tar_tiles_compared"], gates["tar_differ"] = n, bad
    R["gates"] = gates
    ok = not gates["lab_vs_ch0_px"] and not gates.get("library_differ") and not gates.get("target_area_differ") \
        and not gates.get("tar_differ")
    R["ok"] = bool(ok)
    wj(S / "render.json", R)
    log("gates", json.dumps(gates))
    if not ok:
        raise SystemExit("STOP: render gate failed")


# ---- render_l1 (hfv_retrain2 L1 on the arm-B windows) -----------------------------------------------------------
def _l1_window(w):
    import cv2
    import hashlib
    from scipy import ndimage
    RD, mr, table, library, S, L1, OUT = _R["RD"], _R["mr"], _R["table"], _R["library"], _R["S"], _R["l1"], _R["OUT"]
    tiles = w["tiles"]
    with np.load(S / "render" / f"{w['name']}.npz") as z:
        ref = z["ch"]
    ch = np.zeros((7, WIN, WIN), np.uint8)
    lab = np.zeros((WIN, WIN), np.int64)
    dflt_bad, rows = [], {}
    for r in range(6):
        for c in range(6):
            x0, y0 = map(int, tiles[r][c].split("_"))
            sl = (slice(r * T, (r + 1) * T), slice(c * T, (c + 1) * T))
            d, _ = render_tile(RD, mr, table, x0, y0, library)                       # default path
            if not np.array_equal((d > 0.5).astype(np.uint8), ref[(slice(None),) + sl]):
                dflt_bad.append(tiles[r][c])
            ten, sep = render_tile(RD, mr, table, x0, y0, library, true_shapes=True, l1=L1)
            part = bool(np.array_equal(ten[1] + ten[2] + ten[3], ten[0]) and np.array_equal(ten[4] + ten[5] + ten[6], ten[0]))
            fg = sep > 0
            k = np.where(fg, sep - 1, 0)
            typ_ok = bool(np.array_equal(np.argmax(ten[1:4], 0)[fg], L1["ty"][k][fg])
                          and np.array_equal(np.argmax(ten[4:7], 0)[fg], L1["st"][k][fg]))
            ok = part and typ_ok and bool(np.array_equal(fg, ten[0] > 0)) and set(np.unique(ten).tolist()) <= {0.0, 1.0}
            if not ok:
                raise SystemExit(f"STOP: L1 tile {tiles[r][c]} partition {part} types {typ_ok}")
            ch[(slice(None),) + sl] = ten > 0.5
            lab[sl] = sep
            if 1 <= r <= 4 and 1 <= c <= 4:      # central tile -> eval tree, PNG bytes as hfv_retrain2 cells.py writes
                t = tiles[r][c]
                pngs = [cv2.imencode(".png", (a * 255).astype(np.uint8))[1].tobytes() for a in ten]
                rows[t] = dict(pngs=pngs, lab=sep, ncomp=int(ndimage.label(fg, np.ones((3, 3)))[1]),
                               n_cells=int(len(np.unique(sep[fg]))), frac=ten.reshape(7, -1).mean(1).tolist())
    np.savez_compressed(OUT / "render" / f"{w['name']}.npz", tiles=np.array(tiles), ch=ch, lab=lab)
    man = {}
    for t, v in rows.items():
        for nm, by in zip(_R["NAMES"], v["pngs"]):
            f = OUT / "exp_channels" / nm / f"{t}.png"
            f.write_bytes(by)
            man[str(f.relative_to(OUT))] = hashlib.sha256(by).hexdigest()
        f = OUT / "labels" / f"{t}.npy"
        np.save(f, v["lab"])
        man[str(f.relative_to(OUT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return w["name"], dict(default_tiles_differ=dflt_bad, central={t: dict(ncomp=v["ncomp"], n_cells=v["n_cells"], frac=v["frac"])
                                                                   for t, v in rows.items()}), man


def cmd_render_l1(slide, S, OUT, ENV, nproc):
    import hashlib
    sys.path.insert(0, HFV)
    from stages.lin_typing import labels as LB
    from stages.lin_typing import nuclei as NU
    from stages.lin_typing import render as RD
    rasterize, mr, channels = RD.renderer(SEV)
    if not hasattr(mr, "separate_instances"):
        raise SystemExit(f"STOP: {SEV} morphology_renderer has no separate_instances (sim-exp-validation >= 32494f0e)")
    NAMES = list(channels.CHANNEL_NAMES[:7])
    assert NAMES[1:4] == [f"cell_type_{n}" for n in LB.POP_NAMES] and NAMES[4:7] == [f"cell_state_{n}" for n in LB.STATE_NAMES]
    S, OUT, ENV = Path(S), Path(OUT), Path(ENV)
    if OUT.exists():
        raise SystemExit(f"STOP: {OUT} exists")
    W = json.loads((S / "windows.json").read_text())["windows"]
    c = np.load(S / "cells_labels.npz")
    cid = c["cell_id"]
    if not np.array_equal(cid, np.arange(1, len(cid) + 1)):
        raise SystemExit("STOP: cell_id is not 1..N")
    Z = open_z(SLIDES[slide]["mask"])
    table, ids, _, _ = target_table(RD, NU, W, c, Z)
    library = rasterize.NucleusShapes.load(str(S / f"nucleus_shapes_lin_{slide.lower()}.npz"))
    for d in ["render", "labels"] + [f"exp_channels/{n}" for n in NAMES + ["vasculature", "oxygen", "glucose"]]:
        (OUT / d).mkdir(parents=True)
    _R.update(RD=RD, mr=mr, table=table, library=library, S=S, OUT=OUT, NAMES=NAMES,
              l1=dict(mask=SLIDES[slide]["mask"], H=Z.shape[0], W=Z.shape[1], NU=NU, ty=c["ty"].astype(int), st=c["st"].astype(int),
                      pop=list(LB.POP_NAMES), state=list(LB.STATE_NAMES), n=len(cid)))
    with mp.get_context("fork").Pool(min(nproc, len(W))) as pool:
        res = pool.map(_l1_window, W, chunksize=1)
    man, rep = {}, {}
    for name, r, m in res:
        rep[name] = r
        man.update(m)
    central = sorted(t for w in W for row in w["tiles"][1:5] for t in row[1:5])
    miss = []
    for t in central:      # method C ch7-9: byte copy from ENV
        for nm in ("vasculature", "oxygen", "glucose"):
            src = ENV / "exp_channels" / nm / f"{t}.npy"
            if not src.exists():
                miss.append(str(src))
                continue
            by = src.read_bytes()
            f = OUT / "exp_channels" / nm / f"{t}.npy"
            f.write_bytes(by)
            man[str(f.relative_to(OUT))] = hashlib.sha256(by).hexdigest()
    (OUT / "tiles.txt").write_text("\n".join(central) + "\n")
    wj(OUT / "manifest.json", dict(sorted(man.items())))
    dflt = {k: v["default_tiles_differ"] for k, v in rep.items() if v["default_tiles_differ"]}
    cen = {t: v for r in rep.values() for t, v in r["central"].items()}
    fr = np.array([cen[t]["frac"] for t in central])
    content = hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(man.items())).encode()).hexdigest()
    R = dict(slide=slide, rule="msi_s3b_render.py render_l1 (hfv_retrain2 PREREG §7 L1; cells.py L1 rule)", S=str(S), env=str(ENV),
             hfv=HFV, sev=SEV, mask=SLIDES[slide]["mask"], margin_px=L1_MARGIN, n_windows=len(W), n_central=len(central),
             n_files=len(man), content_sha256=content, gate_default_window_tiles_compared=36 * len(W),
             gate_default_differ=dflt, env_missing=miss[:10], n_env_missing=len(miss),
             central_mean_frac=dict(zip(NAMES, fr.mean(0).round(5).tolist())),
             central_components_median=float(np.median([cen[t]["ncomp"] for t in central])),
             central_components_per_cell=sum(cen[t]["ncomp"] for t in central) / max(sum(cen[t]["n_cells"] for t in central), 1),
             central=cen)
    R["ok"] = not dflt and not miss and len(central) == 16 * len(W)
    wj(OUT / "render_l1.json", R)
    log(json.dumps({k: v for k, v in R.items() if k != "central"}))
    if not R["ok"]:
        raise SystemExit("STOP: render_l1 gate failed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "windows":
        cmd_windows(a[1], a[2])
    elif a[0] == "tiles":
        cmd_tiles(a[1], a[2], int(a[3]))
    elif a[0] == "labels":
        cmd_labels(a[1], a[2])
    elif a[0] == "render":
        cmd_render(a[1], a[2], int(a[3]))
    elif a[0] == "render_l1":
        cmd_render_l1(a[1], a[2], a[3], a[4], int(a[5]))
    else:
        raise SystemExit(f"unknown command {a[0]}")
