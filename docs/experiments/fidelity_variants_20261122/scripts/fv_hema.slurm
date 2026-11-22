#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --job-name=fv_hema
#SBATCH --output=/gscratch/cheme/chiu/fidelity_variants_20261122/logs/%x-%j.out
# fidelity_variants_20261122 (hfv docs/experiments/fidelity_variants_20261122/PREREG.md §4), Hyak CPU, read-only inputs.
#   sbatch fv_hema.slurm RUN        RUN = new subdirectory name
# Any failure is a STOP: no retry, no requeue. Writes only under /gscratch/cheme/chiu/fidelity_variants_20261122.
set -uo pipefail
RUN=${1:?RUN}
CC=/gscratch/cheme/chiu; D=$CC/fidelity_variants_20261122; O=$D/$RUN
test -e $O && { echo "STOP $O exists (new folders only)"; exit 1; }
HFV=$CC/hfv_lintype_20261118/hfv
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) run $RUN code $(cat $D/scripts/COMMIT 2>/dev/null)"
echo "932ebd728e549a21eb668c59e98581f1  $HFV/stages/lin_typing/compare.py" | md5sum -c || { echo "STOP compare.py md5"; exit 1; }
sha256sum $D/scripts/fv_hema.py
PY=$CC/crc16-20261024/cellvit/env/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$HFV \
  $PY -u $D/scripts/fv_hema.py $O $SLURM_CPUS_PER_TASK || { echo "$(date -Is) STOP fv_hema"; exit 1; }
echo "### $(date -Is) DONE"
