"""hfv Stage-4 vasculature / oxygen / glucose for a list of 256-px tiles, from the raw multiplex OME-TIFF (steps 1b, 2).

  HFV=<hfv code> mx_channels.py --mx-tiff P --markers P --tiles P --out-tar P --prefix orion-crc36/exp_channels
        --summary P --ds DS --o2-L-um L --glc-L-um L [--sma-refine] [--workers N] [--expect-mpp 0.325]

--tiles: a hfv index.json (key 'patches') or a tiles json with keys 'tiles' + 'ink' (kept = ink fraction <=
--max-ink, default 0.0 = sim-exp-validation style_pool.MAX_PIGMENT_FRACTION).
Output tar members: <prefix>/{vasculature,oxygen,glucose}/<x0>_<y0>.npy, byte format = np.save of
  vasculature: bool (256, 256)   (= hfv vasculature_mask/<tile>.npy = orion-crc33 exp_channels/vasculature)
  oxygen/glucose: float32 (256, 256) in [0, 1]   (= hfv oxygen_npy / glucose_npy = orion-crc33 exp_channels/...)
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
import io

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mx_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mx-tiff", required=True)
ap.add_argument("--markers", required=True)
ap.add_argument("--tiles", required=True)
ap.add_argument("--max-ink", type=float, default=0.0)
ap.add_argument("--out-tar", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--summary", required=True)
ap.add_argument("--ds", type=int, required=True)
ap.add_argument("--o2-L-um", type=float, required=True)
ap.add_argument("--glc-L-um", type=float, required=True)
ap.add_argument("--sma-refine", action="store_true")
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--expect-mpp", type=float, default=None)
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)


tj = json.load(open(a.tiles))
if "patches" in tj:
    tiles = sorted(f"{p['x0']}_{p['y0']}" for p in tj["patches"])
    n_all, n_ink = len(tiles), 0
else:
    n_all = len(tj["tiles"])
    tiles = sorted(t for t in tj["tiles"] if tj["ink"][t] <= a.max_ink)
    n_ink = n_all - len(tiles)
log("tiles", n_all, "kept", len(tiles), "ink-excluded", n_ink)

ci, names = L.channel_index(a.markers)
info = L.ome_info(a.mx_tiff)
mpp = float(info["mpp"][0])
log("ome", info, "channels", ci)
if a.expect_mpp is not None:
    assert abs(mpp - a.expect_mpp) < 1e-9, f"STOP: OME mpp {mpp} != expected {a.expect_mpp}"
H, W = info["shape"][-2:]
outside = [t for t in tiles if int(t.split("_")[0]) + L.PATCH > W or int(t.split("_")[1]) + L.PATCH > H]

t1 = time.time()
F = L.wsi_fields(a.mx_tiff, ci, a.ds, {"oxygen": a.o2_L_um, "glucose": a.glc_L_um}, mpp, a.sma_refine)
t_wsi = time.time() - t1
log("wsi fields", F["_shape"], f"{t_wsi:.0f}s")

t1 = time.time()
CD31 = L.read_plane(a.mx_tiff, ci["CD31"])
SMA = L.read_plane(a.mx_tiff, ci["SMA"]) if a.sma_refine else None
t_planes = time.time() - t1
log("level-0 planes", CD31.shape, f"{t_planes:.0f}s")


def work(t):
    x0, y0 = map(int, t.split("_"))
    v = L.vessel_mask(L.crop(CD31, x0, y0), L.crop(SMA, x0, y0) if SMA is not None else None)
    o = L.coarse_tile(F["oxygen"], x0, y0, a.ds, mpp)
    g = L.coarse_tile(F["glucose"], x0, y0, a.ds, mpp)
    return t, L.npy_bytes(v), L.npy_bytes(o), L.npy_bytes(g), float(v.mean()), float(o.mean()), float(g.mean()), \
        float(o.min()), float(o.max()), float(g.min()), float(g.max())


t1 = time.time()
stats = {"vasculature": [], "oxygen": [], "glucose": []}
mins = {"oxygen": 1.0, "glucose": 1.0}
maxs = {"oxygen": 0.0, "glucose": 0.0}
sha = hashlib.sha256()
os.makedirs(os.path.dirname(os.path.abspath(a.out_tar)), exist_ok=True)
part = a.out_tar + ".part"
with tarfile.open(part, "w") as tar, mp.get_context("fork").Pool(a.workers) as pool:
    for i, r in enumerate(pool.imap(work, tiles, chunksize=32)):
        t = r[0]
        for c, b in zip(("vasculature", "oxygen", "glucose"), r[1:4]):
            ti = tarfile.TarInfo(f"{a.prefix}/{c}/{t}.npy")
            ti.size = len(b)
            ti.mtime = 0
            tar.addfile(ti, io.BytesIO(b))
            sha.update(ti.name.encode() + hashlib.sha256(b).digest())
        stats["vasculature"].append(r[4]); stats["oxygen"].append(r[5]); stats["glucose"].append(r[6])
        mins["oxygen"] = min(mins["oxygen"], r[7]); maxs["oxygen"] = max(maxs["oxygen"], r[8])
        mins["glucose"] = min(mins["glucose"], r[9]); maxs["glucose"] = max(maxs["glucose"], r[10])
        if (i + 1) % 2000 == 0:
            log("tiles done", i + 1)
os.replace(part, a.out_tar)
t_tiles = time.time() - t1


def q(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), p5=float(np.percentile(x, 5)), p50=float(np.median(x)),
                p95=float(np.percentile(x, 95)))


S = dict(
    mx_tiff=a.mx_tiff, mx_tiff_bytes=os.path.getsize(a.mx_tiff), ome=info, markers=names, channel_index=ci,
    tiles_json=a.tiles, n_tiles_all=n_all, n_ink_excluded=n_ink, n_tiles=len(tiles), tiles_outside_image=len(outside),
    params=dict(ds=a.ds, oxygen_L_um=a.o2_L_um, glucose_L_um=a.glc_L_um, sma_refine=a.sma_refine, mpp=mpp,
                k_base=L.K_BASE, demand_weight=L.DEMAND_W, cd68_weight=L.CD68_W, sma_adjacency_px=L.SMA_ADJ,
                vasc_noisy_max_fraction=L.NOISY_MAX, coarse_shape=F["_shape"]),
    hfv_equivalent_command=(
        "python -m stages.multiplex_layers --multiplex-dir <patchify multiplex/> --index <index.json> "
        "--metadata-csv <markers> --out <out> --multiplex-tiff <mx tiff> --oxygen-model wsi-pde "
        f"--glucose-model wsi-pde --wsi-pde-ds {a.ds} --oxygen-pde-krogh-um {a.o2_L_um:g} "
        f"--glucose-pde-krogh-um {a.glc_L_um:g}" + (" --vasc-sma-refine" if a.sma_refine else "")),
    command=" ".join(sys.argv),
    format=dict(vasculature="npy bool (256,256)", oxygen="npy float32 (256,256) [0,1]",
                glucose="npy float32 (256,256) [0,1]", member=f"{a.prefix}/<channel>/<x0>_<y0>.npy"),
    per_tile_mean=dict(vasculature_fraction=q(stats["vasculature"]), oxygen=q(stats["oxygen"]),
                       glucose=q(stats["glucose"])),
    value_range=dict(oxygen=[mins["oxygen"], maxs["oxygen"]], glucose=[mins["glucose"], maxs["glucose"]]),
    tiles_with_any_vessel=int(np.sum(np.asarray(stats["vasculature"]) > 0)),
    out_tar=a.out_tar, out_tar_bytes=os.path.getsize(a.out_tar), content_sha256=sha.hexdigest(),
    timing_s=dict(wsi_fields=t_wsi, level0_planes=t_planes, tiles=t_tiles, total=time.time() - T0),
    workers=a.workers, host=os.uname().nodename,
)
json.dump(S, open(a.summary, "w"), indent=1)
log("done", json.dumps(S["per_tile_mean"]), S["out_tar_bytes"])
