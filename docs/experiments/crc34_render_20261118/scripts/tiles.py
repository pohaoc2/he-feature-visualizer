"""crc34_render_20261118: tissue tiles, H&E tile PNGs and ink tiles for CRC34 (cellvit env + imagecodecs).

  tiles.py N HFV SEV NPROC
Rules, each first checked on CRC33_01 against what the generator was trained on (STOP if not reproduced):
  tiles : hfv stages/patchify.py selection: H&E overview = level 0 subsampled every 64 px, tissue_mask_hsv,
          get_tissue_patches(patch 256, stride 256, tissue_min 0.1) (hfv patchify_lib functions, imported as is; a
          zarr stub only satisfies utils.ome's import, the overview is read from the numpy array as read_overview_chw does).
          CRC33_01 check: the tile set equals processed_crc33/index.json (10,379 tiles).
  he    : the 256 px crop of Lin's registered H&E, PNG (CRC33_01 check: 50 random tiles pixel-equal to the training he/).
  ink   : sim-exp-validation style_pool.pigment_fraction(cv2.imread(png, GRAYSCALE)) > MAX_PIGMENT_FRACTION (0)
          (darkdiag stage_dark.ink; CRC33_01 check: every tile's value equals darkdiag ink.json).
Writes N/data/crc34_he_tiles.tar (he/<tile>.png), N/out/tiles_crc34.json (tile list, ink fractions), N/out/tiles.json.
"""
import json, multiprocessing as mp, os, sys, tarfile, time, types
import cv2
import numpy as np
import tifffile

N, HFV, SEV = sys.argv[1:4]
NPROC = int(sys.argv[4]) if len(sys.argv) > 4 else 1
sys.modules.setdefault("zarr", types.ModuleType("zarr"))
sys.path.insert(0, HFV); sys.path.insert(0, SEV)
from stages.patchify_lib.masking import tissue_mask_hsv          # noqa: E402
from stages.patchify_lib.readers import get_tissue_patches       # noqa: E402
from utils.normalize import percentile_to_uint8                  # noqa: E402
from sim_exp_validation import style_pool                        # noqa: E402

A = "/gscratch/cheme/chiu/he_celltype_audit_20261118"
D = "/gscratch/cheme/chiu/crc16-20261024/darkdiag_20261031"
PS, STRIDE, TMIN, DS = 256, 256, 0.1, 64
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def load_he(path):
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr = s.levels[0].asarray()
        ax = s.axes.upper()
    if ax.startswith("C") or ax.startswith("S"):
        arr = np.moveaxis(arr, 0, -1)
    assert arr.ndim == 3 and arr.shape[2] >= 3, (ax, arr.shape)
    return np.ascontiguousarray(arr[..., :3])


def select(he):
    h, w = he.shape[:2]
    ov = he[0:(h // DS) * DS:DS, 0:(w // DS) * DS:DS, :3]
    if ov.dtype != np.uint8:
        ov = percentile_to_uint8(ov)
    mask = tissue_mask_hsv(np.ascontiguousarray(ov))
    return get_tissue_patches(mask, w, h, PS, STRIDE, TMIN, DS), float(mask.mean())


def tid(x0, y0):
    return f"{x0}_{y0}"


R = dict(rule=dict(patch=PS, stride=STRIDE, tissue_min=TMIN, overview_ds=DS,
                   pigment_level=style_pool.PIGMENT_LEVEL, max_pigment_fraction=style_pool.MAX_PIGMENT_FRACTION))
# ---- CRC33_01 checks --------------------------------------------------------------------------------------------
he33 = load_he(f"{A}/lin/CRC33_01__19510_C33_US_SCAN_OR_001__160715-2-registered.ome.tif")
c33, _ = select(he33)
idx = json.load(open(f"{A}/raw/processed_crc33/index.json"))
train33 = {tid(int(p["x0"]), int(p["y0"])) for p in idx["patches"]}
mine33 = {tid(x, y) for x, y in c33}
R["check_crc33_tiles"] = dict(train=len(train33), mine=len(mine33), equal=mine33 == train33,
                              only_train=len(train33 - mine33), only_mine=len(mine33 - train33))
log("CRC33 tiles", R["check_crc33_tiles"])
assert mine33 == train33, "STOP: tile rule does not reproduce the CRC33_01 training tiles"
ink33 = json.load(open(f"{D}/ink.json"))
tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"c34t_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
os.makedirs(f"{tmp}/c33", exist_ok=True)
rng = np.random.default_rng(20261118)
samp = set(rng.choice(sorted(train33), 50, replace=False).tolist())
bad_px, bad_ink = 0, []
for t in sorted(train33):
    x0, y0 = map(int, t.split("_"))
    p = f"{tmp}/c33/{t}.png"
    cv2.imwrite(p, cv2.cvtColor(he33[y0:y0 + PS, x0:x0 + PS], cv2.COLOR_RGB2BGR))
    f = style_pool.pigment_fraction(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    if f != ink33[t]:
        bad_ink.append(t)
    if t in samp:
        ref = cv2.imread(f"{D}/s3/he/{t}.png", cv2.IMREAD_COLOR)
        bad_px += int(not np.array_equal(ref, cv2.imread(p, cv2.IMREAD_COLOR)))
    os.remove(p)
R["check_crc33_he_png"] = dict(sample=len(samp), not_equal=bad_px)
R["check_crc33_ink"] = dict(tiles=len(train33), mismatched=len(bad_ink), example=bad_ink[:5])
log("CRC33 png/ink", R["check_crc33_he_png"], R["check_crc33_ink"])
assert bad_px == 0 and not bad_ink, "STOP: H&E PNG / ink rule does not reproduce CRC33_01"
del he33

# ---- CRC34 ------------------------------------------------------------------------------------------------------
HE = load_he(f"{N}/lin/data__CRC34__19510_C34_US_SCAN_OR_001__160949-registered.ome.tif")
with tifffile.TiffFile(f"{N}/lin/data__CRC34__P37_S77-CRC34-mask.ome.tif") as tf:
    mshape = tf.series[0].levels[0].shape
log("CRC34 H&E", HE.shape, "mask", mshape)
assert tuple(HE.shape[:2]) == tuple(mshape[-2:]), "STOP: H&E and mask sizes differ"
c34, tissue_frac = select(HE)
tiles = [tid(x, y) for x, y in c34]
ROOT = f"{tmp}/c34"
os.makedirs(f"{ROOT}/he", exist_ok=True)


def work(t):
    x0, y0 = map(int, t.split("_"))
    p = f"{ROOT}/he/{t}.png"
    cv2.imwrite(p, cv2.cvtColor(HE[y0:y0 + PS, x0:x0 + PS], cv2.COLOR_RGB2BGR))
    return t, style_pool.pigment_fraction(cv2.imread(p, cv2.IMREAD_GRAYSCALE))


with mp.get_context("fork").Pool(NPROC) as pool:
    ink = dict(pool.map(work, tiles, chunksize=64))
inked = sorted(t for t, v in ink.items() if v > style_pool.MAX_PIGMENT_FRACTION)
TAR = f"{N}/data/crc34_he_tiles.tar"
assert not os.path.exists(TAR), f"STOP: {TAR} exists"
with tarfile.open(TAR + ".part", "w") as tf:
    tf.add(f"{ROOT}/he", arcname="he")
os.replace(TAR + ".part", TAR)
json.dump(dict(tiles=tiles, ink=ink), open(f"{N}/out/tiles_crc34.json", "w"))
R["crc34"] = dict(he_shape=list(HE.shape), overview_tissue_frac=round(tissue_frac, 4), tiles=len(tiles),
                  ink_tiles=len(inked), ink_gt_0p005=sum(v > 0.005 for v in ink.values()),
                  ink_max=round(max(ink.values()), 4), keep=len(tiles) - len(inked), tar_bytes=os.path.getsize(TAR))
json.dump(R, open(f"{N}/out/tiles.json", "w"), indent=1)
log("CRC34", R["crc34"])
import shutil  # noqa: E402
shutil.rmtree(tmp)
