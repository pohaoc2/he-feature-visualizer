#!/bin/bash
#SBATCH --job-name=r3-slimck
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122: copy of hfv_retrain_20261118 eval/slimck.slurm (slimck.py unchanged, weights checked
# tensor by tensor). Reads this run's S3 outputs/ckpt_tr3_<v>_e20_*, writes the inference copy under the run name
# evalgen.py expects (tr3_concat -> tr_concat, ...) and uploads outputs/infer_<tr3 run>_e20.tar.
#   sbatch slimck3.slurm tr3_concat [tr3_grouped ...]
set -euo pipefail
R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122/outputs
module load coenv/aws/2.17.32
echo "== start $(date -Is) $(hostname) runs $*"
mkdir -p $R/ck/full $R/ck/infer_e20
for R3 in "$@"; do
  EV=${R3/tr3_/tr_}; F=$R/ck/full/$R3; mkdir -p $F
  for x in meta tme cn; do aws s3 cp --only-show-errors $S/ckpt_${R3}_e20_$x $F/$x; done
  cat $F/meta; echo
  $PY /gscratch/cheme/chiu/hfv_eval_20261118/scripts/slimck.py $F/meta $F/cn $F/tme $R/ck/infer_e20 $EV
  tar -cf $R/ck/infer_${R3}_e20.tar -C $R/ck/infer_e20 $EV
  md5sum $R/ck/infer_${R3}_e20.tar
  aws s3 cp --only-show-errors $R/ck/infer_${R3}_e20.tar $S/infer_${R3}_e20.tar
  rm -f $F/cn; echo "== $R3 done $(date -Is)"
done
aws s3 ls $S/ | grep infer_
echo "== done $(date -Is)"
