#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_robust_20261120/logs/%x-%A_%a.out
# hfv_retrain_20261118 PQ robustness readout 1, step 2: StarDist 2D_versatile_he on CPU over one slice of the image list
# (task k of N takes rows k, k+N, ...). Output SD/sd_<k>.npz (new files only). No retry, no requeue: a failed task is a STOP.
#   sbatch --job-name=rob-sd --array=0-39 eval/robust_sd.slurm LIST SD 40
set -uo pipefail
LIST=$1; SD=$2; N=$3
R=/gscratch/cheme/chiu/hfv_robust_20261120
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg KERAS_HOME=$R/keras_home
echo "### $(date -Is) job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID $(hostname)"
cd $R/scripts && $R/sdenv/bin/python -u robust.py stardist $LIST $SD $SLURM_ARRAY_TASK_ID $N || { echo "STOP stardist rc $?"; exit 1; }
echo "### $(date -Is) DONE"
