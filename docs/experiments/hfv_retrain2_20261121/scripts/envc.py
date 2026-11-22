"""hfv_retrain2_20261121 stage 2 (PREREG.md §4): ch7-9 = sim2real method C for every stage-1 tissue tile of one slide,
from the local multiplex OME-TIFF, Hyak CPU.

  HFV=<hfv code> SEV=<sim-exp-validation checkout> python envc.py TAG MX_TIFF MARKERS TISSUE TRAIN STORED_TAR \
      PREFIX ID_PREFIX OUT WORKERS [--oct160 NPZ]

Recipe = scripts/analysis/heldout/real_env_c.py (contract + solve), the only difference being local file reads:
  contract  per tissue tile: contract_vessels.contract (= draw_centerline(real_vessel_skeleton(CD31, AF1, frozen CRC33
            fit FIT_VALUES), 8 um)) on the level-0 256-px crop (zero padded outside the image, as patchify);
            ds-4 vessel pixel = block fraction >= 0.5 -> contract_coarse
  solve     W = vessel ? 1 : exp(-dist / L_local), L_local from k(x) exactly as solve_wsi_pde_map; the ds-4 channel stack
            is hfv read_wsi_channel_stack(CD31, Ki67, CD68) (the stack that reproduced orion-crc33 ch 8-9 byte for byte,
            crc36_render_20261118/mx); patches by extract_patch_from_coarse
Output tar members PREFIX/{vasculature,oxygen,glucose}/<ID_PREFIX><tile>.npy: vasculature = the level-0 contract line
(bool), oxygen / glucose = method C float32 in [0, 1] (the training format of orion-crc33 / c36train).
Gates (exit 1, nothing renamed into place):
  G1 legacy W (Otsu CD31 ds-4 mask) == solve_wsi_pde_map, both fields
  G2 legacy patch npy bytes == STORED_TAR oxygen / glucose (the channels the current Concat was trained on), every tile
  G3 every TRAIN tile is a tissue tile; every written array finite, oxygen / glucose in [0, 1]
  G4 (--oct160) the October CRC33 contract lines (cpm-two-sample-20261012 realA ch 7, 160 tiles) are reproduced exactly
Descriptive (not gated, PREREG §4 amendment): the October fit rule (vessel_contract.fit_real_vessels on the TRAIN tiles)
refitted on this slide; refit constants and Dice of refit vs frozen lines on the TRAIN tiles.
"""
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time

import numpy as np

SEV = os.environ["SEV"]
sys.path.insert(0, os.path.join(SEV, "src"))
sys.path.insert(0, os.path.join(SEV, "scripts/analysis/day0_control"))
sys.path.insert(0, os.environ["MXLIB"])          # crc36_render_20261118/mx (mx_lib.py; needs HFV)
import contract_vessels as CV  # noqa: E402
import mx_lib as L  # noqa: E402
from sim_exp_validation import vessel_contract as vc  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from stages.multiplex_layers_lib import wsi_pde  # noqa: E402
from stages.multiplex_layers_lib.masks import binarize_otsu  # noqa: E402
from stages.multiplex_layers_lib.pde import build_consumption_map, compute_metabolic_demand_map  # noqa: E402
from utils.normalize import percentile_norm  # noqa: E402

TAG, MX, MARK, TISSUE, TRAIN, STORED, PREFIX, IDP, OUT = sys.argv[1:10]
WORKERS = int(sys.argv[10])
OCT = sys.argv[sys.argv.index("--oct160") + 1] if "--oct160" in sys.argv else None
DS, MPP, TILE = 4, 0.325, 256
KROGH = {"oxygen": 200.0, "glucose": 120.0}      # multiplex_layers defaults = real_env_c ST.O2/GLC_PDE_KROGH
T0 = time.time()
G = {}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def stop(m):
    print("STOP", m, flush=True)
    sys.exit(1)


def tiles_of(path):
    j = json.load(open(path))
    if "patches" in j:
        return sorted(f"{p['x0']}_{p['y0']}" for p in j["patches"])
    return sorted(j["tiles"])


def planes19(x0, y0):
    s = np.zeros((19, TILE, TILE), np.uint16)
    s[1] = L.crop(G["af1"], x0, y0)
    s[2] = L.crop(G["cd31"], x0, y0)
    return s


def line_of(t):
    x0, y0 = map(int, t.split("_"))
    return CV.contract(planes19(x0, y0))


def w_contract(t):
    ln = line_of(t)
    b = TILE // DS
    return t, np.packbits(ln), ln.reshape(b, DS, b, DS).mean((1, 3)) >= 0.5, float(ln.mean())


def w_refit(t):
    x0, y0 = map(int, t.split("_"))
    s = planes19(x0, y0)
    return t, np.packbits(vc.draw_centerline(vc.real_vessel_skeleton(s[2], s[1], G["fit"]), width_um=CV.WIDTH_UM))


def npy(a):
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


def main():
    os.makedirs(OUT, exist_ok=False)
    rep = dict(tag=TAG, mx_tiff=MX, fit_frozen=list(CV.FIT_VALUES), width_um=CV.WIDTH_UM, ds=DS, krogh_um=KROGH)
    ci, names = L.channel_index(MARK)
    if names[1] != "AF1" or ci["CD31"] != 2:
        stop(f"marker order {names[:3]} (contract() reads AF1 = plane 1, CD31 = plane 2)")
    info = L.ome_info(MX)
    if abs(float(info["mpp"][0]) - MPP) > 1e-9:
        stop(f"mpp {info['mpp']}")
    tissue, train = tiles_of(TISSUE), open(TRAIN).read().split()
    miss = sorted(set(train) - set(tissue))
    rep.update(n_tissue=len(tissue), n_train=len(train), G3_train_not_tissue=miss[:20], ome=info)
    if miss:
        stop(f"{len(miss)} training tiles are not tissue tiles")
    log("tiles", len(tissue), "train", len(train))
    # ---- ds-4 stack (hfv) and level-0 planes
    stack = wsi_pde.read_wsi_channel_stack(MX, DS, channel_indices=[ci["CD31"], ci["Ki67"], ci["CD68"]])
    log("stack", stack.shape)
    G["af1"], G["cd31"] = L.read_plane(MX, 1), L.read_plane(MX, ci["CD31"])
    log("level-0 planes", G["cd31"].shape)
    # ---- contract lines of every tissue tile
    lines, coarse = {}, np.zeros(stack.shape[1:], bool)
    b = TILE // DS
    with mp.get_context("fork").Pool(WORKERS) as p:
        for i, (t, pk, cb, fr) in enumerate(p.imap_unordered(w_contract, tissue, chunksize=16)):
            lines[t] = pk
            x0, y0 = map(int, t.split("_"))
            ys, xs = y0 // DS, x0 // DS
            hh, ww = min(b, coarse.shape[0] - ys), min(b, coarse.shape[1] - xs)
            coarse[ys:ys + hh, xs:xs + ww] = cb[:hh, :ww]
            if (i + 1) % 2000 == 0:
                log("contract", i + 1)
    rep["t_contract_s"] = round(time.time() - T0)
    log("contract done, ds-4 vessel fraction", float(coarse.mean()))
    np.save(f"{OUT}/contract_coarse.npy", coarse)
    # ---- G4 October lines
    if OCT:
        z = np.load(OCT)
        bad = [s for s, pk in zip(z["ids"].astype(str), z["line"])
               if not np.array_equal(line_of(s), np.unpackbits(pk, axis=-1).astype(bool))]
        rep["G4_oct160"] = dict(n=int(len(z["ids"])), mismatch=bad)
        log("G4", rep["G4_oct160"])
    # ---- solve
    vessel_l = binarize_otsu(percentile_norm(stack[0]))
    kmap = build_consumption_map(compute_metabolic_demand_map(stack[1], None), base_rate=L.K_BASE,
                                 demand_weight=L.DEMAND_W, immune_map=percentile_norm(stack[2]), immune_weight=L.CD68_W)
    d_c = distance_transform_edt(~coarse).astype(np.float32)
    d_l = distance_transform_edt(~vessel_l).astype(np.float32)
    rep.update(vessel_frac_legacy=float(vessel_l.mean()), vessel_frac_contract=float(coarse.mean()))
    W = {}
    for ch, krogh in KROGH.items():
        ref = wsi_pde.solve_wsi_pde_map(stack, cd31_idx=0, ki67_idx=1, mpp=MPP, ds=DS, krogh_um=krogh, k_base=L.K_BASE,
                                        demand_weight=L.DEMAND_W, immune_weight=L.CD68_W, sma_idx=None,
                                        sma_adjacency_px=L.SMA_ADJ, cd68_idx=2, pcna_idx=None)
        Ll = (krogh / (MPP * DS) / np.sqrt(kmap / L.K_BASE)).astype(np.float32)
        Wl = np.where(vessel_l, 1.0, np.exp(-d_l / Ll)).astype(np.float32)
        rep[f"G1_{ch}_legacy_equals_solver"] = bool(np.array_equal(Wl, ref))
        W[ch] = (Wl, np.where(coarse, 1.0, np.exp(-d_c / Ll)).astype(np.float32))
        del ref, Ll
    log("solve", {k: v for k, v in rep.items() if k.startswith("G1")})
    if not all(rep[f"G1_{c}_legacy_equals_solver"] for c in KROGH):
        stop("G1")
    patch = lambda f, t: wsi_pde.extract_patch_from_coarse(f, *map(int, t.split("_")), TILE, MPP, MPP, DS)
    # ---- G2 legacy patches vs the stored training channels
    n2, bad2 = 0, []
    with tarfile.open(STORED) as tf:
        for m in tf:
            p = m.name.split("/")
            if not m.isfile() or p[-2] not in KROGH:
                continue
            t = p[-1][:-4]
            if npy(patch(W[p[-2]][0], t)) != tf.extractfile(m).read():
                bad2.append(f"{p[-2]}/{t}")
            n2 += 1
    rep["G2_legacy_vs_stored"] = dict(n=n2, mismatch=len(bad2), examples=bad2[:10])
    log("G2", rep["G2_legacy_vs_stored"])
    if bad2 or n2 == 0:
        stop("G2")
    # ---- write the method C channels of every tissue tile
    tr = set(train)
    st = {k: [] for k in ("line", "oxygen", "glucose", "oxygen_legacy", "glucose_legacy")}
    sha = hashlib.sha256()
    part = f"{OUT}/envc_{TAG}.tar.part"
    with tarfile.open(part, "w") as tar:
        for t in tissue:
            ln = np.unpackbits(lines[t]).reshape(TILE, TILE).astype(bool)
            arr = dict(vasculature=ln, oxygen=patch(W["oxygen"][1], t).astype(np.float32),
                       glucose=patch(W["glucose"][1], t).astype(np.float32))
            for c in ("oxygen", "glucose"):
                a = arr[c]
                if not (np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1):
                    stop(f"G3 range {c} {t}")
            for c, a in arr.items():
                by = npy(a)
                ti = tarfile.TarInfo(f"{PREFIX}/{c}/{IDP}{t}.npy")
                ti.size, ti.mtime = len(by), 0
                tar.addfile(ti, io.BytesIO(by))
                sha.update(ti.name.encode() + hashlib.sha256(by).digest())
            if t in tr:
                st["line"].append(ln.mean()); st["oxygen"].append(arr["oxygen"].mean()); st["glucose"].append(arr["glucose"].mean())
                st["oxygen_legacy"].append(patch(W["oxygen"][0], t).mean()); st["glucose_legacy"].append(patch(W["glucose"][0], t).mean())
    os.replace(part, f"{OUT}/envc_{TAG}.tar")
    q = lambda v: dict(mean=float(np.mean(v)), p5=float(np.percentile(v, 5)), p50=float(np.median(v)), p95=float(np.percentile(v, 95)))
    rep["train_per_tile_mean"] = {k: q(v) for k, v in st.items()}
    rep["train_tiles_with_line"] = int(np.sum(np.array(st["line"]) > 0))
    rep.update(tar=f"{OUT}/envc_{TAG}.tar", tar_bytes=os.path.getsize(f"{OUT}/envc_{TAG}.tar"), content_sha256=sha.hexdigest(),
               n_written=len(tissue), t_written_s=round(time.time() - T0))
    np.savez_compressed(f"{OUT}/lines_train.npz", ids=np.array(train), line=np.stack([lines[t] for t in train]))
    log("written", rep["tar_bytes"])
    # ---- descriptive refit (October rule on the training tiles of this slide)
    cd = np.stack([L.crop(G["cd31"], *map(int, t.split("_"))) for t in train])
    af = np.stack([L.crop(G["af1"], *map(int, t.split("_"))) for t in train])
    G["fit"] = vc.fit_real_vessels(cd, af)
    del cd, af
    inter = tot = 0
    dices = []
    with mp.get_context("fork").Pool(WORKERS) as p:
        for t, pk in p.imap_unordered(w_refit, train, chunksize=16):
            a, bb = np.unpackbits(pk).astype(bool), np.unpackbits(lines[t]).astype(bool)
            i_, s_ = int((a & bb).sum()), int(a.sum() + bb.sum())
            inter += i_; tot += s_
            if s_:
                dices.append(2 * i_ / s_)
    rep["refit_descriptive"] = dict(fit=[G["fit"].slope, G["fit"].intercept, G["fit"].threshold],
                                    dice_pooled=2 * inter / max(tot, 1), dice_tile_median=float(np.median(dices)) if dices else None,
                                    n_tiles_with_any_line=len(dices))
    log("refit", rep["refit_descriptive"])
    rep["ok"] = not (OCT and rep["G4_oct160"]["mismatch"])
    rep["t_total_s"] = round(time.time() - T0)
    json.dump(rep, open(f"{OUT}/envc_{TAG}.json", "w"), indent=1)
    if not rep["ok"]:
        stop("G4")
    log("DONE")


if __name__ == "__main__":
    main()
