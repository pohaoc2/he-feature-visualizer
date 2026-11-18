#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_lintype_20261118/logs/%x-%j.out
# lintype_render_20261118. MODE=labels (envsmc) | test | full (cellvit env + sim-exp-validation src) | summary
#   sbatch --job-name=ltr-labels --cpus-per-task=2 --mem=16G  --export=ALL,MODE=labels  run.slurm
#   sbatch --job-name=ltr-test   --cpus-per-task=4 --mem=40G  --export=ALL,MODE=test    run.slurm
#   sbatch --job-name=ltr-full   --cpus-per-task=16 --mem=64G --export=ALL,MODE=full    run.slurm
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_lintype_20261118
A=/gscratch/cheme/chiu/he_celltype_audit_20261118
SMC=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python
CV=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
export PYTHONPATH=$A/pylib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== $(date -Is) $(hostname) MODE=$MODE"
case $MODE in
  labels)  $SMC -u $N/scripts/labels_job.py $A $N $N/hfv ;;
  test|full) $CV -u $N/scripts/nuclei_render.py $A $N $N/hfv $N/sev/src $MODE ${SLURM_CPUS_PER_TASK:-1} ;;
  summary) $CV -u $N/scripts/summary.py $A $N $N/hfv ;;
  *) echo "STOP unknown MODE"; exit 1 ;;
esac
echo "== done $(date -Is)"
