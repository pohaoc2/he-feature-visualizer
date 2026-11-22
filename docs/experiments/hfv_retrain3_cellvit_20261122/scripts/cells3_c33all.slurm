#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# D2: cells_oc.py evaluation-only build of all 1,028 CRC33 eval-pack channel tiles (824 + 204 style-only) -> cells/eval_crc33secall_OC.tar
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; LIN=/gscratch/cheme/chiu/hfv_lintype_20261118/out
export HFV=$R/hfv SEV_SRC=$R/sev/src PYTHONPATH=/gscratch/cheme/chiu/hfv_retrain_20261118/PixCell OMP_NUM_THREADS=1
/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python -u $R/scripts/cells_oc.py crc33 /gscratch/cheme/chiu/he_celltype_audit_20261118/lin/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif $LIN "$R/cv/c33_*" - orion-crc33/exp_channels '' $R/cells 32 --eval crc33secall $R/cells/crc33secall_ids.txt || { echo STOP; exit 1; }
