"""hfv_retrain2_20261121 stage 2: pack the Tillicum training inputs (setup2.sh layout), Hyak CPU.

  python pack.py R MODE
R = run folder; MODE = cells.py mode that was built. Writes R/pack/{newch33_v2,masklat33_v2,c36train_v2}.tar + pack.json.
  newch33_v2   orion-crc33/exp_channels/<7 cell ch>/<t>.png (cells_crc33_MODE.tar) + <vasculature|oxygen|glucose>/<t>.npy
               (envc_crc33.tar, training tiles only) + orion-crc33/metadata/exp_index.hdf5 (byte copy of the index of the
               current newch33 = hfv_lintype_20261118/data/new_exp_channels.tar, after checking it lists the same tiles)
  masklat33_v2 masklat/<t>_mask_sd3_vae.npy (masklat_crc33_MODE.tar)
  c36train_v2  c36/exp_channels/<10 ch>/c36_<t>.(png|npy) (cells_crc36_MODE.tar, envc_crc36.tar), c36/vae_features/
               c36_<t>_mask_sd3_vae.npy (masklat_crc36_MODE.tar); c36/features/*, c36/vae_features/*_sd3_vae.npy (H&E) and
               c36/c36_train_ids.txt byte-copied from c36train.tar (the current CRC36 training tar)
Gates (exit 1, nothing renamed into place): every training id has every channel and its mask latent exactly once;
no member twice; after writing, every member of each output re-read and its sha256 equal to the source member's.
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import time

import h5py

R, MODE = sys.argv[1:3]
O = f"{R}/pack"
CELL = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
        "cell_state_nonprolif", "cell_state_dead"]
ENV = ["vasculature", "oxygen", "glucose"]
C36OLD = "/gscratch/cheme/chiu/hfv_crc36_20261118/data/c36train.tar"
NEWCH33 = "/gscratch/cheme/chiu/hfv_lintype_20261118/data/new_exp_channels.tar"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def stop(m):
    print("STOP", m, flush=True)
    sys.exit(1)


class Out:
    def __init__(self, name):
        self.path = f"{O}/{name}.tar"
        self.tar = tarfile.open(self.path + ".part", "w")
        self.sha = {}

    def add(self, name, data):
        if name in self.sha:
            stop(f"{self.path}: {name} twice")
        ti = tarfile.TarInfo(name)
        ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
        self.tar.addfile(ti, io.BytesIO(data))
        self.sha[name] = hashlib.sha256(data).hexdigest()

    def close(self):
        self.tar.close()
        n = 0
        with tarfile.open(self.path + ".part") as tf:
            for m in tf:
                if m.isfile():
                    if hashlib.sha256(tf.extractfile(m).read()).hexdigest() != self.sha.get(m.name):
                        stop(f"{self.path}: re-read sha differs for {m.name}")
                    n += 1
        if n != len(self.sha):
            stop(f"{self.path}: {n} members read back, {len(self.sha)} written")
        md5 = hashlib.md5()
        with open(self.path + ".part", "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                md5.update(b)
        os.replace(self.path + ".part", self.path)
        return dict(path=self.path, members=n, bytes=os.path.getsize(self.path), md5=md5.hexdigest())


def copy(src, out, keep, rename=lambda n: n):
    """Copy members of tar src for which keep(name) is true; -> names copied."""
    got = []
    with tarfile.open(src) as tf:
        for m in tf:
            if m.isfile() and keep(m.name):
                out.add(rename(m.name), tf.extractfile(m).read())
                got.append(rename(m.name))
    return got


def need(out, names):
    miss = [n for n in names if n not in out.sha]
    if miss:
        stop(f"{out.path}: {len(miss)} expected members missing, e.g. {miss[:3]}")


def main():
    os.makedirs(O, exist_ok=False)
    rep = dict(mode=MODE)
    t33 = open("/gscratch/cheme/chiu/hfv_lintype_20261118/out/tiles_keep.txt").read().split()
    t36 = ["c36_" + t for t in open(f"{R}/ref/c36_train_ids.txt").read().split()]   # member ids carry c36_
    s33, s36 = set(t33), set(t36)
    # ---- newch33_v2
    o = Out("newch33_v2")
    copy(f"{R}/cells2/cells_crc33_{MODE}.tar", o, lambda n: n.split("/")[-1][:-4] in s33)
    copy(f"{R}/envc/crc33/envc_crc33.tar", o, lambda n: n.split("/")[-1][:-4] in s33)
    with tarfile.open(NEWCH33) as tf:
        idx = tf.extractfile("orion-crc33/metadata/exp_index.hdf5").read()
    with h5py.File(io.BytesIO(idx), "r") as h:
        ids = [v.decode() if isinstance(v, bytes) else v for k in h for v in h[k][()]]
        keys = list(h.keys())
    if sorted(ids) != sorted(t33) or len(ids) != len(t33):
        stop("current CRC33 index does not list the training tiles")
    o.add("orion-crc33/metadata/exp_index.hdf5", idx)
    need(o, [f"orion-crc33/exp_channels/{c}/{t}.png" for c in CELL for t in t33]
         + [f"orion-crc33/exp_channels/{c}/{t}.npy" for c in ENV for t in t33])
    if len(o.sha) != 10 * len(t33) + 1:
        stop("newch33_v2 member count")
    rep["newch33_v2"] = dict(o.close(), index_keys=keys, index_n=len(ids))
    log("newch33_v2", rep["newch33_v2"])
    # ---- masklat33_v2
    o = Out("masklat33_v2")
    copy(f"{R}/masklat/masklat_crc33_{MODE}.tar", o, lambda n: n.endswith("_mask_sd3_vae.npy"),
         lambda n: "masklat/" + n.split("/")[-1])
    need(o, [f"masklat/{t}_mask_sd3_vae.npy" for t in t33])
    if len(o.sha) != len(t33):
        stop("masklat33_v2 member count")
    rep["masklat33_v2"] = o.close()
    log("masklat33_v2", rep["masklat33_v2"])
    # ---- c36train_v2
    o = Out("c36train_v2")
    pid = lambda n: n.split("/")[-1].split(".")[0]
    copy(f"{R}/cells2/cells_crc36_{MODE}.tar", o, lambda n: pid(n) in s36)
    copy(f"{R}/envc/crc36/envc_crc36.tar", o, lambda n: pid(n) in s36)
    copy(f"{R}/masklat/masklat_crc36_{MODE}.tar", o, lambda n: n.endswith("_mask_sd3_vae.npy"),
         lambda n: "c36/vae_features/" + n.split("/")[-1])
    old = copy(C36OLD, o, lambda n: n.startswith("c36/features/") or n == "c36/c36_train_ids.txt"
               or (n.startswith("c36/vae_features/") and not n.endswith("_mask_sd3_vae.npy")))
    with tarfile.open(C36OLD) as tf:
        ids_old = tf.extractfile("c36/c36_train_ids.txt").read().decode().split()
    if sorted(ids_old) != sorted(t36):
        stop("c36_train_ids.txt differs from combine.json train ids")
    need(o, [f"c36/exp_channels/{c}/{t}.png" for c in CELL for t in t36]
         + [f"c36/exp_channels/{c}/{t}.npy" for c in ENV for t in t36]
         + [f"c36/features/{t}_uni.npy" for t in t36] + [f"c36/vae_features/{t}_sd3_vae.npy" for t in t36]
         + [f"c36/vae_features/{t}_mask_sd3_vae.npy" for t in t36] + ["c36/c36_train_ids.txt"])
    if len(o.sha) != 13 * len(t36) + 1:
        stop(f"c36train_v2 member count {len(o.sha)}")
    rep["c36train_v2"] = dict(o.close(), byte_copied_from_c36train=len(old))
    log("c36train_v2", rep["c36train_v2"])
    rep["t_total_s"] = round(time.time() - T0)
    json.dump(rep, open(f"{O}/pack.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
