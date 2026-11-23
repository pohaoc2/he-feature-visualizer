#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --no-requeue
#SBATCH --job-name=pp_paperimg
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%j.out
# paperpq_official_20261122 PREREG §9 descriptive: regenerated runner-consistent images vs the paper's Fig 2 images.
# Reads S3 share_space/04_inference/si_a1_a2/tiles/<variant> (100 ids) into node /tmp; scoring with NVIDIA_TF32_OVERRIDE=0, CPU.
set -euo pipefail
CC=/gscratch/cheme/chiu; D=$CC/paperpq_official_20261122; S=s3://bagherilab-working/pohao/share_space/04_inference/si_a1_a2/tiles
PY=$CC/crc16-20261024/cellvit/env/bin/python; Wd=/tmp/pp_paperimg_$SLURM_JOB_ID; trap 'rm -rf $Wd' EXIT
module load coenv/aws/2.17.32
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
for v in a1_concat a2_bypass_full_tme a1_per_channel a2_off_shelf production; do mkdir -p $Wd/$v
  aws s3 cp --only-show-errors --recursive --exclude "*" --include "*.png" $S/$v/ $Wd/$v/; echo "$v $(ls $Wd/$v | wc -l) png"; done
export TORCH_HOME=$CC/hfv_score_20261119/torch_home NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$CC/hfv_score_20261119/pylib:$CC/sim2real_apps_20261120/app10/pixcell
$PY -u $D/scripts/pp_paperimg.py $D $Wd || { echo "$(date -Is) STOP pp_paperimg"; exit 1; }
echo "### $(date -Is) DONE"
