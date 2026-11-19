"""hfv_retrain_20261118 evaluation: one held-out slide's evaluation tiles as one tar in the PixCell ORION layout that the
stage-3 scripts read (exp_channels/<ch>/<id>.<ext>, features/<id>_uni.npy, vae_features/<id>_sd3_vae.npy and
<id>_mask_sd3_vae.npy, he/<id>.png, metadata/exp_index.hdf5 with key exp_256), tile ids unchanged (one root per slide).

  python evalpack.py ROOT TILES OUT_TAR CH_TAR MX_TAR UNI_TAR VAE_TAR MASK_TAR HE_TAR [STYLE_TILES]
ROOT   top folder name in the tar (orion-crc36ho, orion-crc34eval)
TILES  text file, one tile id per line (the evaluation tiles of that slide)
STYLE_TILES  optional text file: tiles used only as unpaired style sources (UNI + H&E required, the rest copied when
       present; not listed in the index). CRC33 paper mapping: 5 style tiles are ink tiles without new channels.
CH_TAR the final cell-channel render (7 channels; dead labels from CellViT); MX_TAR vasculature / oxygen / glucose;
UNI_TAR UNI-2h features with the uni-2h config.json transform (feat_uni.py hub); VAE_TAR SD3.5 VAE H&E latents (its
vae_features/ only); MASK_TAR mask latents of the final cell_masks; HE_TAR the H&E tile PNGs (he/<id>.png).
Members are streamed tar to tar. STOP when a tile lacks any of the 10 channels, its UNI, VAE, mask latent or H&E PNG,
or when a member is duplicated. Writes OUT_TAR and OUT_TAR.json (counts, source md5 not recomputed here).
"""
import io
import json
import os
import re
import sys
import tarfile

import h5py
import numpy as np

ROOT, TILES, OUT, CH, MX, UNI, VAE, MASK, HE = sys.argv[1:10]
CELL = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
        "cell_state_nonprolif", "cell_state_dead"]
MXCH = ["vasculature", "oxygen", "glucose"]
ids = [t.strip() for t in open(TILES) if t.strip()]
assert len(set(ids)) == len(ids), "duplicate tile ids"
STY = {t.strip() for t in open(sys.argv[10]) if t.strip()} - set(ids) if len(sys.argv) > 10 else set()
T = set(ids) | STY
have, R = {}, dict(root=ROOT, tiles=len(ids), style_only_tiles=len(STY), sources={})
out = tarfile.open(OUT + ".part", "w")


def stop(msg):
    print(f"STOP {msg}", flush=True)
    sys.exit(1)


def copy(src, pick):
    n = 0
    with tarfile.open(src) as tf:
        for m in tf:
            if not m.isfile():
                continue
            r = pick(m.name)
            if r is None or r[1] not in T:
                continue
            kind, tile, name = r
            if (kind, tile) in have:
                stop(f"duplicate {kind} {tile} in {src}")
            m2 = tarfile.TarInfo(f"{ROOT}/{name}")
            m2.size, m2.mtime, m2.mode = m.size, m.mtime, 0o644
            out.addfile(m2, tf.extractfile(m))
            have[(kind, tile)] = name
            n += 1
    R["sources"][os.path.basename(src)] = n


def ch_pick(chs):
    def f(name):
        p = name.split("/")
        if len(p) < 2 or p[-2] not in chs:
            return None
        stem, ext = os.path.splitext(p[-1])
        return p[-2], stem, f"exp_channels/{p[-2]}/{stem}{ext}"
    return f


def feat_pick(suffix, kind, subdir, dest):
    rx = re.compile(rf"(?:^|/)(\d+_\d+)_{suffix}\.npy$")

    def f(name):
        m = rx.search(name)
        if m is None or (subdir and f"/{subdir}/" not in f"/{name}"):
            return None
        return kind, m.group(1), f"{dest}/{m.group(1)}_{suffix}.npy"
    return f


def he_pick(name):
    m = re.search(r"(?:^|/)he/(\d+_\d+)\.png$", name)
    return None if m is None else ("he", m.group(1), f"he/{m.group(1)}.png")


copy(CH, ch_pick(CELL))
copy(MX, ch_pick(MXCH))
copy(UNI, feat_pick("uni", "uni", "features", "features"))
copy(VAE, feat_pick("sd3_vae", "vae", "vae_features", "vae_features"))
copy(MASK, feat_pick("mask_sd3_vae", "vae_mask", "", "vae_features"))
copy(HE, he_pick)
kinds = CELL + MXCH + ["uni", "vae", "vae_mask", "he"]
miss = {k: sum((k, t) not in have for t in ids) for k in kinds}
miss.update({f"style_{k}": sum((k, t) not in have for t in STY) for k in ("uni", "he")})
if any(miss.values()):
    stop(f"tiles without a member: {miss}")
b = io.BytesIO()
with h5py.File(b, "w") as h:
    h.create_dataset("exp_256", data=np.array(sorted(ids), dtype=object), dtype=h5py.string_dtype())
for name, data in ((f"{ROOT}/metadata/exp_index.hdf5", b.getvalue()), (f"{ROOT}/tile_ids.txt", "".join(f"{t}\n" for t in sorted(ids)).encode())):
    m = tarfile.TarInfo(name)
    m.size, m.mode = len(data), 0o644
    out.addfile(m, io.BytesIO(data))
out.close()
os.replace(OUT + ".part", OUT)
R["members"] = len(have) + 2
R["bytes"] = os.path.getsize(OUT)
json.dump(R, open(OUT + ".json", "w"), indent=1)
print(json.dumps(R), flush=True)
