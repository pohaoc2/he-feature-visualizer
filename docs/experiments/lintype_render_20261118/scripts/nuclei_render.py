"""lintype_render_20261118 step 2 (cellvit env: numpy, cv2, scipy, tifffile, skimage, h5py; + sim-exp-validation src).

  nuclei_render.py A N HFV SEV MODE NPROC      MODE = test (16 sample tiles, nothing published) | full
  A = he_celltype_audit_20261118 (inputs, read only)  N = this run's dir  HFV = he-feature-visualizer copy
  SEV = sim-exp-validation src (e7c06626)

1. nuclei : every Lin cell eroded by the 5 px dilation, block-wise over the whole mask (stages/lin_typing/nuclei.py);
            per cell strict / loose nucleus area, cell area, touches-another-cell flag.
2. library: Lin nuclei (NUC: loose by default, see below) of the cells centred in 400 random kept tiles (the tile count of sim-exp-validation's
            build_nucleus_shape_library.py) -> N/out/nucleus_shapes_lin_crc33_01.npz (new file; same keys).
3. render : every kept tile (training tiles minus ink tiles) through morphology_renderer.render_crop
            (stages/lin_typing/render.py), written with rasterize.write_channel_dir to node-local disk, then one tar
            N/data/new_exp_channels.tar (orion-crc33/exp_channels/<channel>/<tile>.png + metadata/exp_index.hdf5).
4. compare: on the kept tiles of the audit's 500-tile random sample: rendered vs current (CellViT) channels vs the
            eroded Lin nuclei, and each mask against the H&E hematoxylin (N/out/compare_sample.json + per-tile npz).
"""
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time

import cv2
import numpy as np
import tifffile
from scipy import ndimage

A, N, HFV, SEV, MODE = sys.argv[1:6]
NPROC = int(sys.argv[6]) if len(sys.argv) > 6 else 1
sys.path.insert(0, HFV)
from stages.lin_typing import compare as CP   # noqa: E402
from stages.lin_typing import labels as LB    # noqa: E402
from stages.lin_typing import nuclei as NU    # noqa: E402
from stages.lin_typing import render as RD    # noqa: E402

rasterize, mr, channels = RD.renderer(SEV)
NAMES = list(channels.CHANNEL_NAMES[:7])
HE = "/gscratch/cheme/chiu/crc16-20261024/darkdiag_20261031/s3/he"   # training H&E tiles, read only
# NUC = which eroded nucleus feeds the target areas and the library (stages/lin_typing/nuclei.py): loose (default; does
# not erode across the one-pixel line between touching cells) or strict (erodes every cell by 5 px on all sides).
NUC = os.environ.get("NUC", "loose")
assert NUC in ("loose", "strict")
SUF = "" if NUC == "loose" else "_strict"
OUT = f"{N}/out" if MODE == "full" else f"{N}/out/test"
os.makedirs(OUT, exist_ok=True)
LIB = f"{OUT}/nucleus_shapes_lin_crc33_01{SUF}.npz"
TAR = f"{N}/data/new_exp_channels{SUF}.tar" if MODE == "full" else f"{OUT}/new_exp_channels_test{SUF}.tar"
for p in (LIB, TAR):
    if os.path.exists(p):
        raise SystemExit(f"STOP: {p} exists (new files only)")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


R = dict(mode=MODE, sev_src=SEV, nucleus_mode=NUC)
c = np.load(f"{N}/out/cells_labels.npz")
cid = c["cell_id"]
keep_all = open(f"{N}/out/tiles_keep.txt").read().split()
keep_set = set(keep_all)
sample = [t for t in open(f"{A}/raw/tiles_sample.txt").read().split() if t in keep_set]
render_tiles = keep_all if MODE == "full" else sample[:16]
cmp_tiles = sample if MODE == "full" else sample[:16]
R.update(n_render_tiles=len(render_tiles), n_compare_tiles=len(cmp_tiles))

# ---- 1. nuclei --------------------------------------------------------------------------------------------------
with tifffile.TiffFile(f"{A}/lin/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif") as tf:
    lab = tf.series[0].levels[0].asarray()
log("mask", lab.shape, lab.dtype)
H, W = lab.shape
nmax = int(max(int(cid.max()), int(lab.max()))) + 1
assert nmax < 2 ** 24
strict_a = np.zeros(nmax, np.int64); loose_a = np.zeros(nmax, np.int64); cell_a = np.zeros(nmax, np.int64)
touch = np.zeros(nmax, bool)
B, HALO = 2048, NU.DILATION_PX + 2
for y0 in range(0, H, B):
    for x0 in range(0, W, B):
        ya, yb, xa, xb = max(0, y0 - HALO), min(H, y0 + B + HALO), max(0, x0 - HALO), min(W, x0 + B + HALO)
        blk = lab[ya:yb, xa:xb]
        if not blk.any():
            continue
        s, l = NU.erode_labels(blk)
        core = (slice(y0 - ya, y0 - ya + min(B, H - y0)), slice(x0 - xa, x0 - xa + min(B, W - x0)))
        strict_a += np.bincount(s[core].ravel(), minlength=nmax)[:nmax]
        loose_a += np.bincount(l[core].ravel(), minlength=nmax)[:nmax]
        cell_a += np.bincount(blk[core].ravel(), minlength=nmax)[:nmax]
        touch[NU.touching_labels(blk)] = True
touch[0] = False
log("nuclei done")
strict_c, loose_c, cell_c, touch_c = strict_a[cid], loose_a[cid], cell_a[cid], touch[cid]
area_t = RD.target_area(strict_c, loose_c) if NUC == "strict" else np.maximum(loose_c, 1.0)
if MODE == "full":
    np.savez_compressed(f"{N}/out/cells_nuclei{SUF}.npz", cell_id=cid, strict_px=strict_c, loose_px=loose_c, cell_px=cell_c,
                        touching=touch_c, target_px=area_t)
k = c["in_keep"]
um2 = LB.MPP ** 2
q = lambda v: [round(float(t), 1) for t in np.percentile(v, [10, 25, 50, 75, 90])]
R["nuclei_keep"] = dict(
    n=int(k.sum()), cell_mask_area_matches_csv=float(np.mean(cell_c[k] == c["area_cell_px"][k])),
    touching_frac=round(float(touch_c[k].mean()), 4),
    strict_empty=int((strict_c[k] == 0).sum()), loose_empty=int((loose_c[k] == 0).sum()),
    target_from_loose=int(((strict_c == 0) & (loose_c > 0))[k].sum()), target_floor=int(((strict_c == 0) & (loose_c == 0))[k].sum()),
    q_um2_cell=q(cell_c[k] * um2), q_um2_strict=q(strict_c[k] * um2), q_um2_loose=q(loose_c[k] * um2),
    q_um2_strict_isolated=q(strict_c[k & ~touch_c] * um2), q_um2_strict_touching=q(strict_c[k & touch_c] * um2),
    q_um2_loose_touching=q(loose_c[k & touch_c] * um2))
log("nuclei", json.dumps(R["nuclei_keep"]))


def region(x0, y0, halo):
    ya, xa = max(0, y0 - halo), max(0, x0 - halo)
    return lab[ya:min(H, y0 + 256 + halo), xa:min(W, x0 + 256 + halo)], ya, xa


# ---- 2. library -------------------------------------------------------------------------------------------------
rng = np.random.default_rng(20261118)
lib_tiles = sorted(rng.choice(sorted(keep_all), size=400 if MODE == "full" else 8, replace=False).tolist())
pos = {int(v): i for i, v in enumerate(cid)}
masks, types, areas, srcs, ids = [], [], [], [], []
lib_n = dict(candidates=0, empty=0, too_big=0)
for t in lib_tiles:
    x0, y0 = map(int, t.split("_"))
    blk, ya, xa = region(x0, y0, 64)
    s, l = NU.erode_labels(blk)
    s = l if NUC == "loose" else s
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
        masks.append(st); types.append(NU.TYPE_NAMES.index(LB.POP_NAMES[c["ty"][i]])); areas.append(a)
        srcs.append(t); ids.append(int(cid[i]))
NU.save_library(LIB, masks, types, areas, srcs, ids)
library = rasterize.NucleusShapes.load(LIB)
types, areas = np.array(types), np.array(areas)
R["library"] = dict(path=LIB, tiles=len(lib_tiles), **lib_n, kept=len(types),
                    per_type={n: int((types == i).sum()) for i, n in enumerate(NU.TYPE_NAMES)},
                    q_px_area={n: q(areas[types == i]) if (types == i).any() else None for i, n in enumerate(NU.TYPE_NAMES)})
log("library", json.dumps(R["library"]))

# ---- eroded masks of the compare tiles (before the mask is freed) ------------------------------------------------
ero = {}
for t in cmp_tiles:
    x0, y0 = map(int, t.split("_"))
    blk, ya, xa = region(x0, y0, HALO)
    s, l = NU.erode_labels(blk)
    cy, cx = y0 - ya, x0 - xa
    ero[t] = (s[cy:cy + 256, cx:cx + 256] > 0, l[cy:cy + 256, cx:cx + 256] > 0, blk[cy:cy + 256, cx:cx + 256] > 0)
del lab
log("eroded compare tiles", len(ero))

# ---- 3. render --------------------------------------------------------------------------------------------------
TABLE = RD.CellTable(cid, c["x"], c["y"], c["ty"], c["st"], area_t)
LOCAL = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ltr_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
ROOT = os.path.join(LOCAL, "orion-crc33", "exp_channels")
os.makedirs(ROOT, exist_ok=True)


def work(t):
    x0, y0 = map(int, t.split("_"))
    ten, inst = RD.render_tile(TABLE, x0, y0, library, mr)
    rasterize.write_channel_dir(dict(zip(NAMES, ten)), ROOT, t)
    painted = sum(i["pixels"] for i in inst)
    return t, ten.reshape(7, -1).mean(1), len(inst), int(painted - ten[0].sum()), sum(i["clipped"] for i in inst)


with mp.get_context("fork").Pool(NPROC) as pool:
    res = pool.map(work, render_tiles, chunksize=16)
log("rendered", len(res))
frac_new = np.array([r[1] for r in res])
np.savez_compressed(f"{OUT}/render_tiles{SUF}.npz", tile=np.array([r[0] for r in res]), frac=frac_new, names=np.array(NAMES),
                    n_inst=np.array([r[2] for r in res]), overlap_px=np.array([r[3] for r in res]),
                    clipped=np.array([r[4] for r in res]))
R["render"] = dict(tiles=len(res), mean_frac=dict(zip(NAMES, np.round(frac_new.mean(0), 4).tolist())),
                   instances_per_tile_median=float(np.median([r[2] for r in res])),
                   overlap_px_frac_of_mask=round(float(sum(r[3] for r in res) / max(frac_new[:, 0].sum() * 65536, 1)), 4))

# exp_index for the kept tiles, written with PixCell's own helper format (one dataset of tile ids)
import h5py  # noqa: E402
idx_key = "exp_256"
with tarfile.open(f"{A}/raw/o33_rasters_sample.tar") as tf:
    m = tf.extractfile("orion-crc33/metadata/exp_index.hdf5")
    with h5py.File(io.BytesIO(m.read()), "r") as h:
        old_keys = {k: len(h[k]) for k in h.keys()}
idx_key = next(iter(old_keys)) if len(old_keys) == 1 else idx_key
os.makedirs(os.path.join(LOCAL, "orion-crc33", "metadata"), exist_ok=True)
with h5py.File(os.path.join(LOCAL, "orion-crc33", "metadata", "exp_index.hdf5"), "w") as h:
    h.create_dataset(idx_key, data=np.array([t.encode() for t in render_tiles]))
R["exp_index"] = dict(old_keys=old_keys, new_key=idx_key, new_n=len(render_tiles))
with tarfile.open(TAR + ".part", "w") as tf:
    tf.add(os.path.join(LOCAL, "orion-crc33"), arcname="orion-crc33")
os.replace(TAR + ".part", TAR)
log("tar", TAR, os.path.getsize(TAR))

# ---- 4. compare (sample tiles) ----------------------------------------------------------------------------------
old_tar = tarfile.open(f"{A}/raw/o33_rasters_sample.tar")


def png(fobj):
    return cv2.imdecode(np.frombuffer(fobj.read(), np.uint8), cv2.IMREAD_UNCHANGED)


rows = []
old_png_max = set()
for t in cmp_tiles:
    old = {n: png(old_tar.extractfile(f"orion-crc33/exp_channels/{n}/{t}.png")) for n in NAMES}
    old_png_max |= {int(v.max()) for v in old.values()}
    old = {n: v > 0 for n, v in old.items()}
    new = {n: cv2.imread(os.path.join(ROOT, n, f"{t}.png"), cv2.IMREAD_UNCHANGED) > 0 for n in NAMES}
    s, l, cellm = ero[t]
    rgb = cv2.cvtColor(cv2.imread(f"{HE}/{t}.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h = CP.hematoxylin(rgb)
    o, n_ = old["cell_masks"], new["cell_masks"]
    row = dict(tile=t, cov_old=o.mean(), cov_new=n_.mean(), cov_strict=s.mean(), cov_loose=l.mean(), cov_linmask=cellm.mean(),
               dice_old_new=CP.dice(o, n_), dice_old_strict=CP.dice(o, s), dice_strict_new=CP.dice(s, n_),
               dice_old_loose=CP.dice(o, l), dice_old_linmask=CP.dice(o, cellm),
               auc_old=CP.nucleus_auc(h, o), auc_new=CP.nucleus_auc(h, n_), auc_strict=CP.nucleus_auc(h, s),
               auc_loose=CP.nucleus_auc(h, l), auc_linmask=CP.nucleus_auc(h, cellm))
    for nm in NAMES[1:]:
        row[f"frac_old_{nm}"] = old[nm].mean(); row[f"frac_new_{nm}"] = new[nm].mean()
        row[f"dice_{nm}"] = CP.dice(old[nm], new[nm])
    rows.append(row)
keys = [k for k in rows[0] if k != "tile"]
tab = {k: np.array([r[k] for r in rows], float) for k in keys}
np.savez_compressed(f"{OUT}/compare_tiles{SUF}.npz", tile=np.array([r["tile"] for r in rows]),
                    ero_strict=np.packbits(np.stack([ero[r["tile"]][0] for r in rows])),
                    ero_loose=np.packbits(np.stack([ero[r["tile"]][1] for r in rows])), **tab)
brng = np.random.default_rng(20261118)


def med_ci(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    bs = np.median(v[brng.integers(len(v), size=(2000, len(v)))], axis=1)
    return [round(float(np.median(v)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(len(v))]


R["compare_sample"] = {k: med_ci(v) for k, v in tab.items()}
R["compare_sample_note"] = "per key: [median over tiles, bootstrap 95% CI low, high, n tiles]"
R["old_png_values_max"] = sorted(old_png_max)
json.dump(R, open(f"{OUT}/nuclei_render{SUF}.json", "w"), indent=1)
shutil.rmtree(LOCAL)
log("done")
