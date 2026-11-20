#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_robust_20261120/logs/%x-%j.out
# hfv_retrain_20261118 PQ robustness readouts (eval/robust.py), step 1: stage the scored groups, real H&E and real-H&E
# CellViT JSONs of the given sets from S3 into R (new folders only), then write the StarDist image list per set.
#   sbatch --job-name=rob-stage eval/robust_stage.slurm "crc36ho crc33sec"
# Any failure is a STOP (no retry, no requeue).
set -uo pipefail
SETS=$1
R=/gscratch/cheme/chiu/hfv_robust_20261120; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
Wd=/tmp/rob_stage_$SLURM_JOB_ID
source $R/scripts/robust_envsetup.sh
module load coenv/aws/2.17.32
mkdir -p $Wd $R/lists; trap 'rm -rf $Wd' EXIT
stop() { echo "STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) sets $SETS"
if [ ! -d $R/PixCell ]; then
  mkdir $R/PixCell && aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $R/PixCell || stop PixCell
fi
for s in $SETS; do
  for d in $R/real/$s $R/realcv/realcv_$s; do [ -e $d ] && stop "$d exists (new folders only)"; done
  for k in fig2 abl loo decomp; do
    key=$S/outputs/score_${k}_$s.tar
    aws s3 ls $key >/dev/null 2>&1 || { echo "skip $k $s (no $key)"; continue; }
    [ -e $R/score/${k}_$s ] && stop "$R/score/${k}_$s exists"
    mkdir -p $R/score/${k}_$s
    aws s3 cp --only-show-errors $key $Wd/t.tar || stop "download $key"
    tar -xf $Wd/t.tar -C $R/score/${k}_$s --exclude='*/features/*' --exclude='*/sensitivity_tiles/*' \
      --exclude='*/generated/*' --exclude='*.npy' --exclude='*.pt' || stop "untar $key"
    rm -f $Wd/t.tar; echo "### $(date -Is) $k $s $(find $R/score/${k}_$s -name '*.png' | wc -l) png"
  done
  mkdir -p $R/real/$s
  aws s3 cp --only-show-errors $S/inputs/eval/realhe_$s.zip $Wd/r.zip && $PY -m zipfile -e $Wd/r.zip $R/real/$s || stop "realhe $s"
  mkdir -p $R/realcv
  aws s3 cp --only-show-errors $S/outputs/realcv_$s.tar - | tar -x -C $R/realcv || stop "realcv $s"
  echo "### $(date -Is) real $s $(ls $R/real/$s | wc -l) png, realcv $(ls $R/realcv/realcv_$s | grep -c json) json"
  (cd $R/PixCell && PYTHONPATH=$R/PixCell $PY -u $R/scripts/robust.py sdlist $s $R/score $R/real $R/lists/$s.tsv) \
    || stop "sdlist $s"
done
echo "### $(date -Is) DONE"
