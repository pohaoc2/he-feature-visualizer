#!/bin/bash
#SBATCH --job-name=r2-s3kr
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/stage3/logs/%x-%j.out
# hfv_retrain2 stage 3 (PREREG §9.2): inputs that do not need model N. Extract the R0 evaluation packs
# (hfv_eval_20261118/packs/evalpack_<set>.tar) to G/packs/R0 and the N packs (G/inputs/evalpackN_<set>.tar) to G/packs/N;
# K's 2026-11-19 L4 Fig 2 Concat images (S3 hfv_retrain_20261118/outputs/score_fig2_<set>.tar, tiles/a1_concat/<tile>.png
# only) to G/kfig2; then r2score.py zip G KR. Any failure STOP.
set -uo pipefail
G=/gscratch/cheme/chiu/hfv_retrain2_20261121/stage3; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118/outputs
module load coenv/aws/2.17.32
stop() { echo "$(date -Is) STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
for d in packs kfig2; do test -e $G/$d && stop "$G/$d exists"; done
mkdir -p $G/packs/R0 $G/packs/N $G/kfig2 $G/tmp
for s in crc36ho crc33sec; do
  tar -xf /gscratch/cheme/chiu/hfv_eval_20261118/packs/evalpack_$s.tar -C $G/packs/R0 || stop "R0 $s"
  tar -xf $G/inputs/evalpackN_$s.tar -C $G/packs/N || stop "N $s"
  aws s3 cp --only-show-errors $S/score_fig2_$s.tar $G/tmp/k_$s.tar || stop "download K $s"
  sha256sum $G/tmp/k_$s.tar
  tar -tf $G/tmp/k_$s.tar | grep -E "^fig2_$s/tiles/a1_concat/[0-9]+_[0-9]+\.png$" > $G/tmp/k_$s.list
  echo "$s K a1_concat members $(wc -l < $G/tmp/k_$s.list)"
  tar -xf $G/tmp/k_$s.tar -C $G/kfig2 -T $G/tmp/k_$s.list || stop "extract K $s"
  rm $G/tmp/k_$s.tar
done
$PY -I $G/scripts/r2score.py zip $G KR || stop zip
ls -la $G/zips
echo "### $(date -Is) DONE"
