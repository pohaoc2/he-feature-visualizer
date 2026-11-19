#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc01_20261118/logs/%x-%j.out
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_crc01_20261118
echo "== $(date -Is) $(hostname)"
/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python -u $N/scripts/step1_compose.py $N/lin $N/hfv ${SLIDES:-CRC01}
echo "== done $(date -Is)"
