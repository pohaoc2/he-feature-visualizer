#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122: the evaluation-pack half of pack3.slurm alone (the training tars are already built and
# uploaded; rerunning pack3.slurm would rebuild them). evalpack3.py (gated) then upload under new keys only, size check.
#   sbatch --job-name=r3-evalpack evalpack3.slurm
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; S=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122/inputs
stop() { echo "STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
$PY -I -u $R/scripts/evalpack3.py $R /gscratch/cheme/chiu/hfv_eval_20261118/packs $R/evalpack || stop evalpack
module load coenv/aws/2.17.32
for s in crc36ho crc33sec; do
  aws s3 ls $S/eval/evalpack3_$s.tar >/dev/null 2>&1 && stop "evalpack3_$s exists (new keys only)"
  aws s3 cp --only-show-errors $R/evalpack/evalpack3_$s.tar $S/eval/evalpack3_$s.tar || stop "upload evalpack3_$s"
  a=$(stat -c %s $R/evalpack/evalpack3_$s.tar); b=$(aws s3api head-object --bucket bagherilab-working --key pohao/hfv_retrain3_cellvit_20261122/inputs/eval/evalpack3_$s.tar --query ContentLength --output text)
  [ "$a" = "$b" ] || stop "size evalpack3_$s $a != $b"; echo "uploaded evalpack3_$s $a"
done
aws s3 cp --only-show-errors $R/evalpack/packs3.json $S/eval/packs3.json
cat $R/evalpack/packs3.json
echo "### $(date -Is) DONE"
