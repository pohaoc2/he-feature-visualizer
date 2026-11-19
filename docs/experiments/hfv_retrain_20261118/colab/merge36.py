"""hfv_retrain_20261118 Colab side: add the CRC36 training tiles (crc36_render_20261118 assemble36.py tar, extracted to
C36 = .../c36) to the CRC33_01 data root D and write the combined index.

  python merge36.py D C36
Files of C36/exp_channels/<ch>/, C36/features/, C36/vae_features/ move into the same folders of D (names are prefixed
c36_, so nothing is overwritten; an existing name is a STOP). D/metadata/exp_index.hdf5 (the CRC33_01 index, 9,418 ids: the A100 setup log prints exp_256 = 9418)
is kept as exp_index_crc33.hdf5 and replaced by one key exp_256 = CRC33_01 ids + C36/c36_train_ids.txt, written as
PixCell's _write_h5_index does (vlen str). Rerunning after a finished merge only checks the index.
"""
import os
import shutil
import sys

import h5py

D, C = sys.argv[1:3]
H, H33 = f"{D}/metadata/exp_index.hdf5", f"{D}/metadata/exp_index_crc33.hdf5"
N33 = 9418
# the 10 active_channels of configs/config_controlnet_exp.py (exp_channels/ also holds two legacy folders, cell_types and
# cell_states, with one entry each, which the loader does not read)
CH = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
      "cell_state_nonprolif", "cell_state_dead", "vasculature", "oxygen", "glucose"]


def ids_of(path):
    with h5py.File(path, "r") as h:
        return [v.decode() if isinstance(v, bytes) else v for k in h for v in h[k][()]]


if not os.path.exists(H33):
    shutil.copy2(H, H33)
i33 = ids_of(H33)
assert len(i33) == N33 and not any(t.startswith("c36_") for t in i33), (len(i33), "unexpected CRC33 index")
i36 = [t.strip() for t in open(f"{C}/c36_train_ids.txt") if t.strip()]
assert all(t.startswith("c36_") for t in i36) and len(set(i36)) == len(i36)
moved = 0
for sub in ("exp_channels", "features", "vae_features"):
    for root, _, files in os.walk(f"{C}/{sub}"):
        dst = os.path.join(D, os.path.relpath(root, C))
        os.makedirs(dst, exist_ok=True)
        for f in files:
            if os.path.exists(f"{dst}/{f}"):
                sys.exit(f"STOP {dst}/{f} exists")
            os.rename(f"{root}/{f}", f"{dst}/{f}")
            moved += 1
for ch in CH:
    have = {os.path.splitext(f)[0] for f in os.listdir(f"{D}/exp_channels/{ch}")}
    miss = [t for t in i33 + i36 if t not in have]
    assert not miss, f"STOP channel {ch}: {len(miss)} ids without a file, e.g. {miss[:3]}"
for t in i36:
    for p in (f"features/{t}_uni.npy", f"vae_features/{t}_sd3_vae.npy", f"vae_features/{t}_mask_sd3_vae.npy"):
        assert os.path.exists(f"{D}/{p}"), f"STOP missing {p}"
ids = i33 + i36
with h5py.File(H, "w") as h:
    ds = h.create_dataset("exp_256", shape=(len(ids),), dtype=h5py.special_dtype(vlen=str))
    for i, t in enumerate(ids):
        ds[i] = t
print(f"merge36: moved {moved} files; index exp_256 = {len(i33)} CRC33_01 + {len(i36)} CRC36 = {len(ids)}", flush=True)
