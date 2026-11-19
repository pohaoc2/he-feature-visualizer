#!/bin/bash
#SBATCH --job-name=ev-pack36
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_eval_20261118/logs/%x-%j.out
# hfv_retrain_20261118 evaluation: the CRC36 hold-out evaluation pack (all 1,092 hold-out tiles; guard band and training
# tiles left out) + the 300 primary picks and their unpaired derangement (eval/picks.py, README rules), uploaded to
# s3://.../hfv_retrain_20261118/inputs/. Sources = the files the CRC36 training tar was assembled from (crc36_render_20261118
# c36train.slurm: final render with CellViT dead labels, mx channels, uni_crc36_hub / feat_crc36 / masklat_crc36) + the
# H&E tiles. One job, no array; set -e, a failure stops the job, no retry.
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/pack36.slurm
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_crc36_20261118; E=/gscratch/cheme/chiu/hfv_eval_20261118
PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
module load coenv/aws/2.17.32
echo "== start $(date -Is) $(hostname)"
mkdir -p $E/packs; cd $E/packs
$PY -c "import json; h=json.load(open('$N/out/combine.json'))['holdout']; open('crc36ho_pool.txt','w').write(''.join(t+'\n' for t in sorted(h['holdout_tile_ids']))); print('holdout', len(h['holdout_tile_ids']))"
$PY $E/scripts/picks.py crc36ho_pool.txt 300 crc36ho_p300
$PY $E/scripts/evalpack.py orion-crc36ho crc36ho_pool.txt evalpack_crc36ho.tar $N/data/crc36_exp_channels.tar \
  $N/data/crc36_mx_channels.tar $N/src/uni_crc36_hub.tar $N/src/feat_crc36.tar $N/src/masklat_crc36.tar $N/data/crc36_he_tiles.tar
md5sum evalpack_crc36ho.tar crc36ho_p300_ids.txt crc36ho_p300_unpaired.json | tee evalpack_crc36ho.md5
for f in evalpack_crc36ho.tar evalpack_crc36ho.tar.json crc36ho_p300_ids.txt crc36ho_p300_unpaired.json; do
  aws s3 cp --only-show-errors $f $S/inputs/eval/$f; done
aws s3 ls $S/inputs/eval/
echo "== done $(date -Is)"
