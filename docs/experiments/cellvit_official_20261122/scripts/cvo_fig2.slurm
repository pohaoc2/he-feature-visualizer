#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --job-name=cvo_f2
# cellvit_official_20261122 (PREREG.md §5): cvo_fig2.py with PixCell cb62fa7 (the app-10 copy used by s3metrics.slurm).
#   sbatch --output=$D/logs/%x-%j.out cvo_fig2.slurm
set -uo pipefail
CC=/gscratch/cheme/chiu; D=$CC/cellvit_official_20261122; PIX=$CC/sim2real_apps_20261120/app10/pixcell
PY=$CC/crc16-20261024/cellvit/env/bin/python
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) pixcell compute_ablation_metrics md5 $(md5sum $PIX/tools/compute_ablation_metrics.py | cut -c1-32)"
test -e $D/full/fig2 && { echo "STOP $D/full/fig2 exists"; exit 1; }
cd $PIX
OMP_NUM_THREADS=1 PYTHONPATH=$PIX PYTHONDONTWRITEBYTECODE=1 $PY -u $D/scripts/cvo_fig2.py $D $D/full/fig2 || { echo "STOP fig2"; exit 1; }
echo "### $(date -Is) DONE"
