#!/bin/bash
#SBATCH --job-name=r2-s3prep
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/stage3/logs/%x-%j.out
# hfv_retrain2 stage 3 (PREREG §9): N evaluation packs (Fig 2 tiles) + the ND kin (s3prep.py), then S3 stage3/inputs/.
# Any failure STOP, no retry.
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; G=$R/stage3; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S1=s3://bagherilab-working/pohao/hfv_retrain_20261118/inputs/eval; S=s3://bagherilab-working/pohao/hfv_retrain2_20261121/stage3/inputs
module load coenv/aws/2.17.32
stop() { echo "$(date -Is) STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
mkdir -p $G/inputs/lists
for f in crc36ho_p300_ids.txt crc33sec_fig2_ids.txt; do aws s3 cp --only-show-errors $S1/$f $G/inputs/lists/$f || stop "list $f"; done
$PY -I $G/scripts/s3prep.py packs $R /gscratch/cheme/chiu/hfv_eval_20261118/packs $G/inputs/lists $G/inputs || stop packs
$PY -I $G/scripts/s3prep.py ndkin $R /gscratch/cheme/chiu/sim2real_apps_20261120/app8 $G/inputs || stop ndkin
for f in evalpackN_crc36ho.tar evalpackN_crc33sec.tar kin_r2ND.tar lists/crc36ho_p300_ids.txt lists/crc33sec_fig2_ids.txt; do
  aws s3 cp --only-show-errors $G/inputs/$f $S/$(basename $f) || stop "upload $f"
done
aws s3 ls $S/
echo "### $(date -Is) DONE"
