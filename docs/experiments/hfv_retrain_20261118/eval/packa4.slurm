#!/bin/bash
#SBATCH --job-name=ev-packa4
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_a4_20261120/logs/%x-%j.out
# hfv_retrain_20261118 follow-up 1 (coordinator 2026-11-20): evaluation pack for the Fig 3 a4 UNI-probe sweep rerun with
# the retrained Concat. Tiles = the paper's fixed 30-tile manifest (inference_output_20260527.zip a1_concat/a4_uni_probe/
# shared_tiles.json) minus 20992_46080 and 26368_50688, which are not among the 9,418 eligible CRC33_01 tiles (same ink /
# tissue rule that dropped 10496_17664 from crc33sec; approved deviation) -> 28 tiles (a4_ids28.txt).
# Same sources and the same eval/evalpack.py as pack33.slurm: new cell channels (src33/newch33.tar = S3
# inputs/new_exp_channels_crc33_01.tar), new mask latents (src33/masklat33.tar), and vasculature / oxygen / glucose,
# stored UNI, VAE H&E latents and H&E from orion-crc33.zip (range-read members of these 28 tiles = S3
# inputs/eval/z_a4.tar, same member filter as pack33.slurm). src33 is only read.
# Check (coordinator: STOP if it fails): the UNI vectors the sweep edits (a4 features.npz "uni", the paper's file) must
# equal the pack's stored UNI (features/<id>_uni.npy) exactly for all 28 tiles.
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) packa4.slurm
set -euo pipefail
A=/gscratch/cheme/chiu/hfv_a4_20261120; E=/gscratch/cheme/chiu/hfv_eval_20261118; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
module load coenv/aws/2.17.32
echo "== start $(date -Is) $(hostname) job $SLURM_JOB_ID"
mkdir -p $A/src $A/packs; cd $A/src
aws s3 cp --only-show-errors $S/inputs/eval/z_a4.tar z_a4.tar
aws s3 cp --only-show-errors $S/inputs/eval/a4_inputs.tar - | tar -x
md5sum z_a4.tar $E/src33/newch33.tar $E/src33/masklat33.tar a4_inputs/features.npz a4_inputs/a4_ids28.txt
cd $A/packs
$PY $A/evalpack.py orion-crc33a4 $A/src/a4_inputs/a4_ids28.txt evalpack_crc33a4.tar $E/src33/newch33.tar \
  $A/src/z_a4.tar $A/src/z_a4.tar $A/src/z_a4.tar $E/src33/masklat33.tar $A/src/z_a4.tar
rm -rf chk && mkdir chk && tar -xf evalpack_crc33a4.tar -C chk
$PY - $A/src/a4_inputs chk/orion-crc33a4 <<'PYC'
import json, sys
from pathlib import Path
import numpy as np
src, root = Path(sys.argv[1]), Path(sys.argv[2])
ids = [t.strip() for t in open(src / "a4_ids28.txt") if t.strip()]
f = np.load(src / "features.npz", allow_pickle=True)
index = {str(t): i for i, t in enumerate(f["tile_ids"].tolist())}
uni = np.asarray(f["uni"])
res, bad = {}, []
for t in ids:
    a = uni[index[t]].reshape(-1)
    b = np.load(root / "features" / f"{t}_uni.npy").reshape(-1)
    eq = a.shape == b.shape and np.array_equal(a.astype(np.float32), b.astype(np.float32))
    res[t] = {"equal": bool(eq), "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape else None,
              "dtype_features_npz": str(a.dtype), "dtype_pack": str(b.dtype)}
    if not eq:
        bad.append(t)
chans = sorted(p.name for p in (root / "exp_channels").iterdir())
n_ch = {c: len(list((root / "exp_channels" / c).glob("*"))) for c in chans}
out = {"tiles": len(ids), "uni_equal": len(ids) - len(bad), "uni_not_equal": bad, "channels": n_ch, "per_tile": res}
json.dump(out, open("unicheck_a4.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_tile"}))
if bad:
    print("STOP UNI check failed:", bad); sys.exit(1)
print("UNI check ok: features.npz uni == pack features/<id>_uni.npy for all", len(ids))
PYC
rm -rf chk
md5sum evalpack_crc33a4.tar | tee evalpack_crc33a4.md5
for f in evalpack_crc33a4.tar evalpack_crc33a4.tar.json unicheck_a4.json; do aws s3 cp --only-show-errors $f $S/inputs/eval/$f; done
echo "== done $(date -Is)"
