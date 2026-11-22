"""hfv_retrain2_20261121 stage 3 (PREREG §9), Hyak CPU: inputs for the model-N generation on Colab L4.

  python -I s3prep.py packs R EVALPACKS IDS_DIR OUT     -> OUT/evalpackN_<set>.tar (orion-<set>/...), OUT/packs.json
  python -I s3prep.py ndkin R APP8 OUT PIX              -> OUT/kin_r2ND/{roots/orion-crc16L1, ref.npz, uni.npz,
                                                           design.json}, OUT/kin_r2ND.tar, OUT/ndkin.json

packs (PREREG §9.1): for crc36ho (crc36ho_p300_ids.txt, 300) and crc33sec (crc33sec_fig2_ids.txt, 100) a data root with
  only those tiles = evalpack_<set>.tar (features/<t>_uni.npy, he/<t>.png, vae_features/<t>_sd3_vae.npy unchanged) with
  three things replaced: the 7 cell channels (R/cells2/eval_<set>_L1.tar), ch7-9 (R/envc/<slide>/envc_<slide>.tar, method
  C; CRC36 members carry the c36_ prefix) and the mask latent (R/masklat/masklat_eval_<set>_L1.tar).
  Gates (STOP): every id has all 10 channels, UNI, H&E, both latents; cell PNGs are 256x256 uint8 {0,255}; type and state
  channels each partition cell_masks; vasculature bool 256x256; oxygen / glucose float32 in [0, 1]; mask latent shape and
  dtype = the R0 latent it replaces. Descriptive: per set mean |L1 - R0| mask pixels, R0 / L1 coverage.
ndkin (PREREG §9.3, arm ND): model N, CRC16 L1 channels + method C ch7-9 (R/eval/crc16_L1/exp_channels, byte copies),
  style D_<tile> (app 8 kin uni.npz = app 7 bytes, app 7 derangement), seeds 42 and 43, 384 rows for r2_gen.py.
  ref.npz <tile>__n = PixCell cb62fa7 (PIX) load_exp_channels(tile, config.data.active_channels, 256, <kin root>/
  exp_channels) as float32 (10, 256, 256) (the app9_bank.py / app10_vec.py rule; the loader mirror-pads the non-binary
  channels, so the tensor is not rebuilt by hand). Gates: 192 tiles = the app 8 tiles; D_ bytes = app 7 embeddings.npz;
  window / style_source of every tile = the app 8 KD rows.
"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

CELL = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
        "cell_state_nonprolif", "cell_state_dead"]
ENV = ["vasculature", "oxygen", "glucose"]
SETS = {"crc36ho": ("crc36ho_p300_ids.txt", "crc36", "c36/exp_channels", "c36_"),
        "crc33sec": ("crc33sec_fig2_ids.txt", "crc33", "orion-crc33/exp_channels", "")}


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def members(tar, want):
    """{name: bytes} for the wanted member names (one pass)"""
    out = {}
    with tarfile.open(tar) as tf:
        for m in tf:
            if m.name in want:
                out[m.name] = tf.extractfile(m).read()
    return out


def png(b):
    return np.asarray(Image.open(io.BytesIO(b)))


def npy(b):
    return np.load(io.BytesIO(b), allow_pickle=False)


def add(tf, name, b):
    ti = tarfile.TarInfo(name)
    ti.size = len(b)
    ti.mode = 0o644
    tf.addfile(ti, io.BytesIO(b))


def packs(R, EP, IDS, OUT):
    R, EP, IDS, OUT = map(Path, (R, EP, IDS, OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for s, (idf, slide, epre, idp) in SETS.items():
        ids = [t.strip() for t in open(IDS / idf) if t.strip()]
        root = f"orion-{s}"
        keep = {f"{root}/features/{t}_uni.npy" for t in ids} | {f"{root}/he/{t}.png" for t in ids} | \
               {f"{root}/vae_features/{t}_sd3_vae.npy" for t in ids} | \
               {f"{root}/vae_features/{t}_mask_sd3_vae.npy" for t in ids} | \
               {f"{root}/exp_channels/cell_masks/{t}.png" for t in ids}
        old = members(EP / f"evalpack_{s}.tar", keep)
        cells = members(R / f"cells2/eval_{s}_L1.tar", {f"exp_channels/{c}/{t}.png" for c in CELL for t in ids})
        env = members(R / f"envc/{slide}/envc_{slide}.tar", {f"{epre}/{c}/{idp}{t}.npy" for c in ENV for t in ids})
        ml = members(R / f"masklat/masklat_eval_{s}_L1.tar", {f"ml_eval_{s}/{t}_mask_sd3_vae.npy" for t in ids})
        n_exp = dict(old=len(ids) * 5, cells=len(ids) * 7, env=len(ids) * 3, ml=len(ids))
        got = dict(old=len(old), cells=len(cells), env=len(env), ml=len(ml))
        if got != n_exp:
            stop(f"{s}: members {got} != {n_exp}")
        dmask, cov0, cov1 = [], [], []
        tarp = OUT / f"evalpackN_{s}.tar"
        with tarfile.open(str(tarp) + ".part", "w") as tf:
            for t in ids:
                ch = {c: png(cells[f"exp_channels/{c}/{t}.png"]) for c in CELL}
                for c, a in ch.items():
                    if a.shape != (256, 256) or a.dtype != np.uint8 or not set(np.unique(a)) <= {0, 255}:
                        stop(f"{s} {t} {c}: {a.shape} {a.dtype} {np.unique(a)[:5]}")
                m = ch["cell_masks"] > 0
                for grp in (CELL[1:4], CELL[4:7]):
                    st = np.stack([ch[c] > 0 for c in grp])
                    if (st.sum(0) > 1).any() or not np.array_equal(st.any(0), m):
                        stop(f"{s} {t}: {grp} do not partition cell_masks")
                r0 = png(old[f"{root}/exp_channels/cell_masks/{t}.png"]) > 0
                dmask.append(float((r0 != m).mean()))
                cov0.append(float(r0.mean()))
                cov1.append(float(m.mean()))
                for c in CELL:
                    add(tf, f"{root}/exp_channels/{c}/{t}.png", cells[f"exp_channels/{c}/{t}.png"])
                for c in ENV:
                    b = env[f"{epre}/{c}/{idp}{t}.npy"]
                    a = npy(b)
                    if c == "vasculature":
                        if a.dtype != bool or a.shape != (256, 256):
                            stop(f"{s} {t} vasculature {a.dtype} {a.shape}")
                    elif a.dtype != np.float32 or a.shape != (256, 256) or not np.isfinite(a).all() \
                            or a.min() < 0 or a.max() > 1:
                        stop(f"{s} {t} {c} {a.dtype} {a.shape} [{a.min()}, {a.max()}]")
                    add(tf, f"{root}/exp_channels/{c}/{t}.npy", b)
                a_new = npy(ml[f"ml_eval_{s}/{t}_mask_sd3_vae.npy"])
                a_old = npy(old[f"{root}/vae_features/{t}_mask_sd3_vae.npy"])
                if a_new.shape != a_old.shape or a_new.dtype != a_old.dtype or not np.isfinite(a_new).all():
                    stop(f"{s} {t} mask latent {a_new.shape} {a_new.dtype} vs {a_old.shape} {a_old.dtype}")
                add(tf, f"{root}/vae_features/{t}_mask_sd3_vae.npy", ml[f"ml_eval_{s}/{t}_mask_sd3_vae.npy"])
                for n in (f"features/{t}_uni.npy", f"he/{t}.png", f"vae_features/{t}_sd3_vae.npy"):
                    add(tf, f"{root}/{n}", old[f"{root}/{n}"])
        Path(str(tarp) + ".part").rename(tarp)
        rep[s] = dict(tiles=len(ids), ids_file=idf, tar=str(tarp), tar_sha256=sha(tarp), bytes=tarp.stat().st_size,
                      members=len(ids) * 14, mask_pixels_changed_L1_vs_R0=float(np.mean(dmask)),
                      coverage_R0=float(np.mean(cov0)), coverage_L1=float(np.mean(cov1)))
        print(s, json.dumps(rep[s]), flush=True)
    (OUT / "packs.json").write_text(json.dumps(rep, indent=1) + "\n")


def ndkin(R, A8, OUT, PIX):
    R, A8, OUT, PIX = (Path(p).resolve() for p in (R, A8, OUT, PIX))
    K = OUT / "kin_r2ND"
    if K.exists():
        stop(f"{K} exists")
    src = R / "eval/crc16_L1/exp_channels"
    tl = [t.strip() for t in open(R / "eval/crc16_L1/tiles.txt") if t.strip()]
    d8 = json.loads((A8 / "kin/design_k.json").read_text())["rows"]
    kd = {r["tile"]: r for r in d8 if r["cond"] == "KD" and r["seed"] == 42}
    if sorted(tl) != sorted(kd) or len(tl) != 192:
        stop(f"tiles {len(tl)} != app 8 KD tiles {len(kd)}")
    u8 = np.load(A8 / "kin/uni.npz")
    e7 = np.load(Path("/gscratch/cheme/chiu/sim2real_apps_20261120/app7/held/embeddings.npz"))
    uni, ref, rows = {}, {}, []
    dst = K / "roots/orion-crc16L1/exp_channels"
    for c in CELL + ENV:
        (dst / c).mkdir(parents=True)
    for t in tl:
        k = kd[t]["reference"]
        if not k.startswith("D_") or u8[k].tobytes() != e7[k].tobytes() or u8[k].dtype != np.float32:
            stop(f"style {t} {k}")
        uni[k] = u8[k]
        for c in CELL:
            b = (src / c / f"{t}.png").read_bytes()
            (dst / c / f"{t}.png").write_bytes(b)
            a = png(b)
            if a.shape != (256, 256) or not set(np.unique(a)) <= {0, 255}:
                stop(f"{t} {c}")
        for c in ENV:
            (dst / c / f"{t}.npy").write_bytes((src / c / f"{t}.npy").read_bytes())
    import os
    os.chdir(PIX)
    sys.path.insert(0, str(PIX))
    from diffusion.utils.misc import read_config
    from tools.stage3.tile_pipeline import load_exp_channels
    cfg = read_config("configs/config_controlnet_exp_a1_concat.py")
    if getattr(cfg, "tme_input_mode", None) != "all_channels" or list(cfg.data.active_channels) != CELL + ENV:
        stop(f"config {list(cfg.data.active_channels)}")
    for t in tl:
        c = load_exp_channels(t, cfg.data.active_channels, 256, dst)
        a = np.asarray(c, np.float32)
        if a.shape != (10, 256, 256) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            stop(f"ref {t} {a.shape} [{a.min()}, {a.max()}]")
        ref[f"{t}__n"] = a
    for seed in (42, 43):
        for t in tl:
            rows.append(dict(id=f"N_ND.{t}_{seed}", model="N", cond="ND", root="orion-crc16L1", tile=t,
                             window=kd[t]["window"], seed=seed, reference=kd[t]["reference"],
                             style_source=kd[t]["style_source"], ref_key=f"{t}__n", guidance=2.5, steps=20,
                             conditioning_scale=1.0))
    np.savez_compressed(K / "ref.npz", **ref)
    np.savez(K / "uni.npz", **uni)
    (K / "design.json").write_text(json.dumps(dict(n_rows=len(rows), rows=rows,
                                                   rule="hfv_retrain2 PREREG §9.3 arm ND (s3prep.py ndkin)"), indent=1) + "\n")
    with tarfile.open(OUT / "kin_r2ND.tar", "w") as tf:
        tf.add(K, arcname=".")
    rep = dict(rows=len(rows), tiles=len(tl), kin_tar=str(OUT / "kin_r2ND.tar"), kin_sha256=sha(OUT / "kin_r2ND.tar"),
               manifest_sha256=sha(R / "eval/crc16_L1/manifest.json"))
    (OUT / "ndkin.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    {"packs": packs, "ndkin": ndkin}[sys.argv[1]](*sys.argv[2:])
