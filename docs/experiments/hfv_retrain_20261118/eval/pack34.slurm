#!/bin/bash
#SBATCH --job-name=ev-pack34
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_eval_20261118/logs/%x-%j.out
# hfv_retrain_20261118 evaluation: the CRC34 (patient 2, held out) pack = every kept CRC34 tile (out/tiles_keep.txt, the
# training filters) as cell_masks, because run_evaluation.py draws its 1,000 tiles with random.sample from all tiles of
# the root (README rules); full members only for the drawn 1,000 and the 300 picks; + the 300 primary picks and their derangement (eval/picks.py). Sources: the final render with CellViT
# dead labels (data/crc34_exp_channels.tar after MODE=cvdead / labels / full), mx channels, gpu34.slurm outputs
# (uni_crc34_hub, feat_crc34, masklat_crc34 = mask latents of the first-pass cell_masks) and the kept H&E tiles.
# STOP unless the final cell_masks are byte-equal to the first pass (so the mask latents belong to the final masks).
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/pack34.slurm
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_crc34_20261118; E=/gscratch/cheme/chiu/hfv_eval_20261118
PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
module load coenv/aws/2.17.32
echo "== start $(date -Is) $(hostname)"
mkdir -p $E/packs; cd $E/packs
for f in $N/data/crc34_exp_channels.tar $N/data/pass1_nodead/crc34_exp_channels.tar $N/gpu/uni_crc34_hub.tar \
         $N/gpu/feat_crc34.tar $N/gpu/masklat_crc34.tar; do test -s $f || { echo "STOP missing $f"; exit 1; }; done
$PY - <<PYM
import hashlib, sys, tarfile
def h(p):
    d = {}
    with tarfile.open(p) as tf:
        for m in tf:
            q = m.name.split("/")
            if m.isfile() and len(q) >= 2 and q[-2] == "cell_masks":
                d[q[-1]] = hashlib.md5(tf.extractfile(m).read()).hexdigest()
    return d
a, b = h("$N/data/pass1_nodead/crc34_exp_channels.tar"), h("$N/data/crc34_exp_channels.tar")
diff = [k for k in a if b.get(k) != a[k]]
print("cell_masks pass1", len(a), "final", len(b), "differ", len(diff))
sys.exit(1 if (a.keys() != b.keys() or diff) else 0)
PYM
sort -u $N/out/tiles_keep.txt > crc34eval_pool.txt; wc -l crc34eval_pool.txt
$PY $E/scripts/picks.py crc34eval_pool.txt 300 crc34eval_p300
# the 1,000 tiles run_evaluation.py will draw (its own steps: random.seed(42), sorted cell_masks ids, random.sample);
# full members for these + the 300 picks, cell_masks only for the rest of the slide
$PY - <<'PYS'
import random
pool = sorted(t.strip() for t in open("crc34eval_pool.txt") if t.strip())
random.seed(42)
s = random.sample(pool, min(1000, len(pool)))
open("crc34eval_n1000_order.txt", "w").write("".join(t + "\n" for t in s))
full = sorted(set(s) | {t.strip() for t in open("crc34eval_p300_ids.txt") if t.strip()})
open("crc34eval_full_ids.txt", "w").write("".join(t + "\n" for t in full))
open("crc34eval_maskonly_ids.txt", "w").write("".join(t + "\n" for t in pool if t not in set(full)))
print("pool", len(pool), "drawn", len(s), "full", len(full), "first drawn", s[:3])
PYS
$PY $E/scripts/evalpack.py orion-crc34eval crc34eval_full_ids.txt evalpack_crc34eval.tar $N/data/crc34_exp_channels.tar \
  $N/data/crc34_mx_channels.tar $N/gpu/uni_crc34_hub.tar $N/gpu/feat_crc34.tar $N/gpu/masklat_crc34.tar \
  $N/data/crc34_he_tiles_keep.tar - crc34eval_maskonly_ids.txt
md5sum evalpack_crc34eval.tar crc34eval_p300_ids.txt crc34eval_p300_unpaired.json crc34eval_n1000_order.txt | tee evalpack_crc34eval.md5
for f in evalpack_crc34eval.tar evalpack_crc34eval.tar.json crc34eval_p300_ids.txt crc34eval_p300_unpaired.json crc34eval_n1000_order.txt; do
  aws s3 cp --only-show-errors $f $S/inputs/eval/$f; done
aws s3 ls $S/inputs/eval/
echo "== done $(date -Is)"
