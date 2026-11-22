#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 stage 2: pack.py (gated tars) then upload to the run's S3 inputs/ and check sizes.
#   sbatch --job-name=r2-pack pack.slurm MODE         Any failure is a STOP: no retry, no requeue.
set -uo pipefail
MODE=$1; R=/gscratch/cheme/chiu/hfv_retrain2_20261121; S=s3://bagherilab-working/pohao/hfv_retrain2_20261121/inputs
stop() { echo "STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) mode $MODE"
/gscratch/cheme/chiu/hfv_mx_20261118/venv/bin/python -c "import h5py" 2>/dev/null && PY=/gscratch/cheme/chiu/hfv_mx_20261118/venv/bin/python || PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
$PY -u $R/scripts/pack.py $R $MODE || stop pack
module load coenv/aws/2.17.32
for n in newch33_v2 masklat33_v2 c36train_v2; do
  aws s3 ls $S/$n.tar >/dev/null 2>&1 && stop "$S/$n.tar exists (new keys only)"
  aws s3 cp --only-show-errors $R/pack/$n.tar $S/$n.tar || stop "upload $n"
  a=$(stat -c %s $R/pack/$n.tar); b=$(aws s3api head-object --bucket bagherilab-working --key pohao/hfv_retrain2_20261121/inputs/$n.tar --query ContentLength --output text)
  [ "$a" = "$b" ] || stop "size $n local $a S3 $b"
  echo "uploaded $n $a bytes"
done
aws s3 cp --only-show-errors $R/pack/pack.json $S/pack.json
echo "### $(date -Is) DONE"
