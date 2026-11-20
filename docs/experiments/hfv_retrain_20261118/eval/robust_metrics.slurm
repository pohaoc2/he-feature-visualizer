#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_robust_20261120/logs/%x-%j.out
# hfv_retrain_20261118 PQ robustness readouts, step 3: per-image StarDist PQ / DICE and CellViT detection F1 vs Lin
# (eval/robust.py metrics), then the bootstrap summary (robust.py stats). One set per call; OUT is a new folder.
#   sbatch --job-name=rob-met-SET eval/robust_metrics.slurm SET LIN_DIR SD OUT
set -uo pipefail
SET=$1; LIN=$2; SD=$3; OUT=$4
R=/gscratch/cheme/chiu/hfv_robust_20261120; PY=$R/sdenv/bin/python
export OMP_NUM_THREADS=1 MPLBACKEND=Agg PYTHONPATH=$R/PixCell
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $SET"
cd $R/PixCell || exit 1
$PY -u $R/scripts/robust.py metrics $SET $R/score $R/real $R/realcv/realcv_$SET $SD $LIN $OUT $SLURM_CPUS_PER_TASK || { echo "STOP metrics"; exit 1; }
$PY -u $R/scripts/robust.py stats $OUT || { echo "STOP stats"; exit 1; }
echo "### $(date -Is) DONE"
