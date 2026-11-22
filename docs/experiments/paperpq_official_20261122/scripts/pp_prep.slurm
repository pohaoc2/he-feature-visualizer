#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pp_prep
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%j.out
# paperpq_official_20261122 (PREREG.md §3-4): pp_prep.py with PixCell cb62fa7 (the app-10 copy cvo_fig2 used).
set -uo pipefail
CC=/gscratch/cheme/chiu; D=$CC/paperpq_official_20261122; PIX=$CC/sim2real_apps_20261120/app10/pixcell
PY=$CC/crc16-20261024/cellvit/env/bin/python
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) pixcell compute_ablation_metrics md5 $(md5sum $PIX/tools/compute_ablation_metrics.py | cut -c1-32)"
cd $PIX
OMP_NUM_THREADS=2 PYTHONPATH=$PIX PYTHONDONTWRITEBYTECODE=1 $PY -u $D/scripts/pp_prep.py $D || { echo "STOP prep"; exit 1; }
tar -cf $D/roots.tar -C $D roots && md5sum $D/roots.tar
echo "### $(date -Is) DONE"
