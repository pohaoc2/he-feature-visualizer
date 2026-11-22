#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --job-name=r3prep
set -euo pipefail
R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122
/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python -u $R/scripts/prep_he.py $R 800
ls $R/he/*.zip > $R/he/list_real.txt; wc -l $R/he/list_real.txt
