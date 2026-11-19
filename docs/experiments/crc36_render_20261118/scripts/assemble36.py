"""crc36_render_20261118: the CRC36 part of the CRC33_01 + CRC36 training root, as one tar in the PixCell paired-exp
layout (diffusion/data/datasets/paired_exp_controlnet_dataset.py), every CRC36 tile id prefixed "c36_" so it cannot
collide with a CRC33 id (both slides name tiles <x0>_<y0>).

  python assemble36.py N MX_TAR SRC_DIR OUT_TAR
N        hfv_crc36_20261118 (data/crc36_exp_channels.tar = the final render with CellViT dead labels,
         data/pass1_nodead/crc36_exp_channels.tar = first pass, out/combine.json = training tile ids)
MX_TAR   vasculature / oxygen / glucose .npy channels of CRC36 (mx/crc36.slurm)
SRC_DIR  the Colab tars from S3: uni_crc36_hub.tar (UNI-2h, uni-2h config.json preprocessing = sim-exp-validation
         forward.py:load_uni_extractor, ea5d5bd9), feat_crc36.tar (only its SD3.5 VAE H&E latents are used; its UNI used
         timm defaults), masklat_crc36.tar (SD3.5 VAE latents of the first-pass cell_masks)
Members are streamed tar to tar (no per-file extraction on gscratch). Checks, each a STOP on failure:
  - the cell_masks members of the final render are byte-equal to the first pass (so masklat_crc36 is the mask latent of
    the final cell_masks; the dead labels change only the state channels)
  - every training tile has all 10 channels, its UNI, its H&E VAE latent and its mask latent, nothing else is written
Writes OUT_TAR (members c36/exp_channels/<ch>/c36_<t>.<ext>, c36/features/c36_<t>_uni.npy,
c36/vae_features/c36_<t>_sd3_vae.npy and c36_<t>_mask_sd3_vae.npy, c36/c36_train_ids.txt) and OUT_TAR.json.
"""
import hashlib
import json
import os
import re
import sys
import tarfile

N, MX, SRC, OUT = sys.argv[1:5]
CELL = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
        "cell_state_nonprolif", "cell_state_dead"]
MXCH = ["vasculature", "oxygen", "glucose"]
train = json.load(open(f"{N}/out/combine.json"))["holdout"]["train_tile_ids"]
T = set(train)
assert len(T) == len(train)


def stop(msg):
    print(f"STOP {msg}", flush=True)
    sys.exit(1)


def md5_masks(path):
    h = {}
    with tarfile.open(path) as tf:
        for m in tf:
            p = m.name.split("/")
            if m.isfile() and len(p) >= 2 and p[-2] == "cell_masks":
                h[p[-1]] = hashlib.md5(tf.extractfile(m).read()).hexdigest()
    return h


new_tar = f"{N}/data/crc36_exp_channels.tar"
h1, h2 = md5_masks(f"{N}/data/pass1_nodead/crc36_exp_channels.tar"), md5_masks(new_tar)
if h1.keys() != h2.keys():
    stop(f"cell_masks tile sets differ: pass1 {len(h1)}, final {len(h2)}")
diff = [k for k in h1 if h1[k] != h2[k]]
if diff:
    stop(f"{len(diff)} cell_masks tiles changed after the dead re-render, e.g. {diff[:3]}")

have = {}   # (kind, tile) -> member name written
R = dict(train_tiles=len(T), cell_masks_equal_pass1=len(h1), sources={})
out = tarfile.open(OUT, "w")


def copy(src_tar, pick):
    """pick(member name) -> (kind, tile, out name) or None"""
    n = 0
    with tarfile.open(src_tar) as tf:
        for m in tf:
            if not m.isfile():
                continue
            r = pick(m.name)
            if r is None:
                continue
            kind, tile, name = r
            if tile not in T:
                continue
            if (kind, tile) in have:
                stop(f"duplicate {kind} {tile} in {src_tar}")
            m2 = tarfile.TarInfo(name)
            m2.size, m2.mtime, m2.mode = m.size, m.mtime, 0o644
            out.addfile(m2, tf.extractfile(m))
            have[(kind, tile)] = name
            n += 1
    R["sources"][os.path.basename(src_tar)] = n


def ch_pick(chs):
    def f(name):
        p = name.split("/")
        if len(p) < 2 or p[-2] not in chs:
            return None
        stem, ext = os.path.splitext(p[-1])
        return p[-2], stem, f"c36/exp_channels/{p[-2]}/c36_{stem}{ext}"
    return f


def feat_pick(suffix, kind, subdir):
    rx = re.compile(rf"(?:^|/)(\d+_\d+)_{suffix}\.npy$")

    def f(name):
        m = rx.search(name)
        if m is None or (subdir and f"/{subdir}/" not in f"/{name}"):
            return None
        return kind, m.group(1), f"c36/{'features' if kind == 'uni' else 'vae_features'}/c36_{m.group(1)}_{suffix}.npy"
    return f


copy(new_tar, ch_pick(CELL))
copy(MX, ch_pick(MXCH))
copy(f"{SRC}/uni_crc36_hub.tar", feat_pick("uni", "uni", "features"))
copy(f"{SRC}/feat_crc36.tar", feat_pick("sd3_vae", "vae", "vae_features"))
copy(f"{SRC}/masklat_crc36.tar", feat_pick("mask_sd3_vae", "vae_mask", ""))
kinds = CELL + MXCH + ["uni", "vae", "vae_mask"]
miss = {k: sum((k, t) not in have for t in T) for k in kinds}
if any(miss.values()):
    stop(f"training tiles without a member: {miss}")
ids = "".join(f"c36_{t}\n" for t in sorted(T)).encode()
import io  # noqa: E402
m = tarfile.TarInfo("c36/c36_train_ids.txt")
m.size = len(ids)
out.addfile(m, io.BytesIO(ids))
out.close()
R.update(members=len(have) + 1, out=OUT, out_bytes=os.path.getsize(OUT),
         out_md5=hashlib.md5(open(OUT, "rb").read()).hexdigest())
json.dump(R, open(OUT + ".json", "w"), indent=1)
print(json.dumps(R), flush=True)
