#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=c36-train
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc36_20261118/logs/%x-%j.out
# crc36_render_20261118 -> hfv_retrain_20261118: CRC36 training tar for the Colab runs (one job, no array).
# 1. combine.py (hold-out, guard band, training ids) on the final render; 2. the Colab outputs uni_crc36_hub /
# feat_crc36 / masklat_crc36 from S3 (presigned GET, links file outside git); 3. assemble36.py; 4. upload
# inputs/c36train.tar (presigned multipart). Every step stops the job on failure (set -e), no retry.
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_crc36_20261118
L=/gscratch/cheme/chiu/hfv_lintype_20261118
PYG=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python
LINKS=$N/links_c36train.json
S=$N/scripts
echo "== start $(date -Is) $(hostname)"
$PYG $S/combine.py $N $L
mkdir -p $N/src
for k in uni_crc36_hub feat_crc36 masklat_crc36; do $PYG $S/s3io.py get $LINKS $k $N/src/$k.tar; done
md5sum $N/data/crc36_mx_channels.tar $N/data/crc36_exp_channels.tar $N/src/*.tar
$PYG $S/assemble36.py $N $N/data/crc36_mx_channels.tar $N/src $N/data/c36train.tar
cat $N/data/c36train.tar.json
$PYG $S/s3io.py put $LINKS c36train $N/data/c36train.tar
echo "== done $(date -Is)"
