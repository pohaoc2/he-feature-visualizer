#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122 (copy of hfv_retrain2 pack.slurm, run folder / names changed): pack3.py + evalpack3.py (gated tars) then upload to the run's S3 inputs/ and check sizes.
#   sbatch --job-name=r3-pack pack3.slurm MODE         Any failure is a STOP: no retry, no requeue.
set -uo pipefail
MODE=$1; R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; S=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122/inputs
stop() { echo "STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) mode $MODE"
/gscratch/cheme/chiu/hfv_mx_20261118/venv/bin/python -c "import h5py" 2>/dev/null && PY=/gscratch/cheme/chiu/hfv_mx_20261118/venv/bin/python || PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
$PY -u $R/scripts/pack3.py $R $MODE || stop pack
$PY -u $R/scripts/evalpack3.py $R /gscratch/cheme/chiu/hfv_eval_20261118/packs $R/evalpack || stop evalpack
module load coenv/aws/2.17.32
for n in newch33_v3 masklat33_v3 c36train_v3; do
  aws s3 ls $S/$n.tar >/dev/null 2>&1 && stop "$S/$n.tar exists (new keys only)"
  aws s3 cp --only-show-errors $R/pack/$n.tar $S/$n.tar || stop "upload $n"
  a=$(stat -c %s $R/pack/$n.tar); b=$(aws s3api head-object --bucket bagherilab-working --key pohao/hfv_retrain3_cellvit_20261122/inputs/$n.tar --query ContentLength --output text)
  [ "$a" = "$b" ] || stop "size $n local $a S3 $b"
  echo "uploaded $n $a bytes"
done
aws s3 cp --only-show-errors $R/pack/pack.json $S/pack.json
for s in crc36ho crc33sec; do
  aws s3 ls $S/eval/evalpack3_$s.tar >/dev/null 2>&1 && stop "evalpack3_$s exists"
  aws s3 cp --only-show-errors $R/evalpack/evalpack3_$s.tar $S/eval/evalpack3_$s.tar || stop "upload evalpack3_$s"
done
aws s3 cp --only-show-errors $R/evalpack/packs3.json $S/eval/packs3.json
echo "### $(date -Is) DONE"
