"""hfv_retrain3_cellvit_20261122 (PREREG §5.1): full evaluation packs with the official-CellViT inputs, Hyak CPU.

  python -I evalpack3.py R EVALPACKS OUT
For crc36ho (all 1,092 held-out tiles) and crc33sec (824 channel tiles + 209 style-only tiles): every member of the
hfv_retrain_20261118 evalpack_<set>.tar is copied byte for byte, except three things (as hfv_retrain2 s3prep.py packs):
  the 7 cell channels   <- R/cells/eval_<set>_OC.tar exp_channels/<ch>/<t>.png
  ch7-9 (method C)      <- hfv_retrain2 envc/<slide>/envc_<slide>.tar (CRC36 members carry c36_), the training-build bytes
  mask latent           <- R/masklat/masklat_eval_<set>_OC.tar ml_eval_<set>/<t>_mask_sd3_vae.npy
Gates (STOP): the output has exactly the member names of the old pack; every replaced member found once; cell PNGs
256x256 uint8 {0,255}; type channels disjoint, state channels disjoint, both unions equal and inside cell_masks;
vasculature bool, oxygen / glucose float32 in [0, 1]; mask latent shape / dtype = the one it replaces.
-> OUT/evalpack3_<set>.tar (orion-<set>/...), OUT/packs3.json
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
E2 = Path("/gscratch/cheme/chiu/hfv_retrain2_20261121")
SETS = {"crc36ho": ("crc36", "c36/exp_channels", "c36_"), "crc33sec": ("crc33", "orion-crc33/exp_channels", "")}


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


def members(tar, keep):
    out = {}
    with tarfile.open(tar) as tf:
        for m in tf:
            if m.isfile() and keep(m.name):
                if m.name in out:
                    stop(f"{tar}: {m.name} twice")
                out[m.name] = tf.extractfile(m).read()
    return out


def png(b):
    return np.asarray(Image.open(io.BytesIO(b)))


def npy(b):
    return np.load(io.BytesIO(b), allow_pickle=False)


def main(R, EP, OUT):
    R, EP, OUT = map(Path, (R, EP, OUT))
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for s, (slide, epre, idp) in SETS.items():
        root = f"orion-{s}"
        old = members(EP / f"evalpack_{s}.tar", lambda n: True)
        ch_tiles = sorted({n.split("/")[-1].rsplit(".", 1)[0] for n in old if n.startswith(f"{root}/exp_channels/cell_masks/")})
        want_env = {f"{epre}/{c}/{idp}{t}.npy" for c in ENV for t in ch_tiles}
        cells = members(R / f"cells/eval_{s}_OC.tar", lambda n: True)
        env = members(E2 / f"envc/{slide}/envc_{slide}.tar", lambda n: n in want_env)
        ml = members(R / f"masklat/masklat_eval_{s}_OC.tar", lambda n: n.endswith("_mask_sd3_vae.npy"))
        if len(cells) != 7 * len(ch_tiles) or len(env) != 3 * len(ch_tiles) or len(ml) != len(ch_tiles):
            stop(f"{s}: cells {len(cells)} env {len(env)} ml {len(ml)} for {len(ch_tiles)} channel tiles")
        new, dmask = {}, []
        for t in ch_tiles:
            ch = {c: png(cells[f"exp_channels/{c}/{t}.png"]) for c in CELL}
            for c, a in ch.items():
                if a.shape != (256, 256) or a.dtype != np.uint8 or not set(np.unique(a).tolist()) <= {0, 255}:
                    stop(f"{s} {t} {c}: {a.shape} {a.dtype}")
            m = ch["cell_masks"] > 0
            ty = np.stack([ch[c] > 0 for c in CELL[1:4]])
            st = np.stack([ch[c] > 0 for c in CELL[4:7]])
            if (ty.sum(0) > 1).any() or (st.sum(0) > 1).any() or not np.array_equal(ty.any(0), st.any(0)) \
                    or (ty.any(0) & ~m).any():
                stop(f"{s} {t}: type / state channels do not partition the matched nuclei")
            dmask.append(float((png(old[f"{root}/exp_channels/cell_masks/{t}.png"]) > 0).mean() - m.mean()))
            for c in CELL:
                new[f"{root}/exp_channels/{c}/{t}.png"] = cells[f"exp_channels/{c}/{t}.png"]
            for c in ENV:
                b = env[f"{epre}/{c}/{idp}{t}.npy"]
                a = npy(b)
                if c == "vasculature":
                    if a.dtype != bool or a.shape != (256, 256):
                        stop(f"{s} {t} vasculature {a.dtype} {a.shape}")
                elif a.dtype != np.float32 or a.shape != (256, 256) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                    stop(f"{s} {t} {c} {a.dtype} {a.shape}")
                new[f"{root}/exp_channels/{c}/{t}.npy"] = b
            b = ml[f"ml_eval_{s}/{t}_mask_sd3_vae.npy"]
            a_new, a_old = npy(b), npy(old[f"{root}/vae_features/{t}_mask_sd3_vae.npy"])
            if a_new.shape != a_old.shape or a_new.dtype != a_old.dtype or not np.isfinite(a_new).all():
                stop(f"{s} {t} mask latent {a_new.shape} {a_new.dtype} vs {a_old.shape} {a_old.dtype}")
            new[f"{root}/vae_features/{t}_mask_sd3_vae.npy"] = b
        miss = [n for n in new if n not in old]
        if miss:
            stop(f"{s}: replaced members not in the old pack, e.g. {miss[:3]}")
        tarp = OUT / f"evalpack3_{s}.tar"
        n_rep = 0
        with tarfile.open(str(tarp) + ".part", "w") as tf:
            for n in sorted(old):
                b = new.get(n, old[n])
                n_rep += n in new
                ti = tarfile.TarInfo(n)
                ti.size, ti.mode, ti.mtime = len(b), 0o644, 0
                tf.addfile(ti, io.BytesIO(b))
        got = sorted(m.name for m in tarfile.open(str(tarp) + ".part") if m.isfile())
        if got != sorted(old):
            stop(f"{s}: member names differ from the old pack")
        Path(str(tarp) + ".part").rename(tarp)
        rep[s] = dict(tar=str(tarp), bytes=tarp.stat().st_size, members=len(got), replaced=n_rep,
                      channel_tiles=len(ch_tiles), sha256=hashlib.sha256(tarp.read_bytes()).hexdigest(),
                      coverage_old_minus_new_mean=float(np.mean(dmask)))
        print(s, json.dumps(rep[s]), flush=True)
    (OUT / "packs3.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
