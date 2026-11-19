#!/bin/bash
#SBATCH --job-name=ev-slimck
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_eval_20261118/logs/%x-%j.out
# hfv_retrain_20261118 evaluation: inference copies of the epoch-20 checkpoints (eval/slimck.py, weights unchanged and
# checked tensor by tensor) -> S3 outputs/infer_<run>_e20.tar (<run>/step_*/controlnet_epoch_20_step_*.pth +
# tme_module.pth), the format eval/evalgen.py --ck reads. Runs given as arguments (default: all four).
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/slimck.slurm [RUN ...]
set -euo pipefail
E=/gscratch/cheme/chiu/hfv_eval_20261118; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118/outputs
module load coenv/aws/2.17.32
RUNS=${*:-tr_concat tr_grouped tr_bypass tr_perch}; EP=${EP:-20}   # EP: epoch (env), 20 = final; other epochs only for kit tests
echo "== start $(date -Is) $(hostname) runs $RUNS"
mkdir -p $E/ck/full $E/ck/infer_e$EP
for R in $RUNS; do
  F=$E/ck/full/$R; mkdir -p $F
  for x in meta tme cn; do aws s3 cp --only-show-errors $S/ckpt_${R}_e${EP}_$x $F/$x; done
  cat $F/meta; echo
  $PY $E/scripts/slimck.py $F/meta $F/cn $F/tme $E/ck/infer_e$EP $R
  tar -cf $E/ck/infer_${R}_e${EP}.tar -C $E/ck/infer_e$EP $R
  md5sum $E/ck/infer_${R}_e${EP}.tar
  aws s3 cp --only-show-errors $E/ck/infer_${R}_e${EP}.tar $S/infer_${R}_e${EP}.tar
  rm -f $F/cn; echo "== $R done $(date -Is)"
done
aws s3 ls $S/ | grep infer_
echo "== done $(date -Is)"
