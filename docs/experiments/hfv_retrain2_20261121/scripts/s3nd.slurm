#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/stage3/logs/%x-%j.out
# hfv_retrain2 stage 3 (PREREG §9.3): the ND kin with ref.npz from PixCell cb62fa7 load_exp_channels (s3prep.py ndkin,
# execution-log deviation: the first build in s3prep.slurm rebuilt the tensor by hand, without the loader's mirror padding
# of oxygen / glucose; that kin is not used). New folder inputs_nd/; S3 stage3/inputs/kin_r2ND.tar replaced.
# 2026-11-22: optional first argument NP (default ND) builds arm NP the same way into inputs_np/ (kin_r2NP.tar):
#   sbatch -J r2-s3np s3nd.slurm NP     (the app-10 KP arm is not generated after the ch6 STOP)
set -uo pipefail
ARM=${1:-ND}; a=$(echo $ARM | tr A-Z a-z); [[ $ARM =~ ^(ND|NP)$ ]] || { echo "STOP arm $ARM"; exit 1; }
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; G=$R/stage3; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
PIX=/gscratch/cheme/chiu/sim2real_apps_20261120/app10/pixcell
S=s3://bagherilab-working/pohao/hfv_retrain2_20261121/stage3/inputs
module load coenv/aws/2.17.32
stop() { echo "$(date -Is) STOP $*"; exit 1; }
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) PixCell tile_pipeline md5 $(md5sum $PIX/tools/stage3/tile_pipeline.py | cut -c1-32)"
test -e $G/inputs_$a && stop "inputs_$a exists"
mkdir -p $G/inputs_$a
$PY -I $G/scripts/s3prep.py ${a}kin $R /gscratch/cheme/chiu/sim2real_apps_20261120/app8 $G/inputs_$a $PIX || stop ${a}kin
aws s3 cp --only-show-errors $G/inputs_$a/kin_r2$ARM.tar $S/kin_r2$ARM.tar || stop upload
aws s3 ls $S/
echo "### $(date -Is) DONE"
