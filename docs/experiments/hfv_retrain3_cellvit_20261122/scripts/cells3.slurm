#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122 (PREREG §3.1-3.2): 7 cell channels from official CellViT nuclei (cells_oc.py), one slide.
#   sbatch --job-name=r3cells-SLIDE cells3.slurm crc33|crc36 [TILES_OVERRIDE]      Any failure is a STOP: no retry.
set -uo pipefail
SL=$1; R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; E=/gscratch/cheme/chiu/hfv_eval_20261118/packs
OUT=${3:-$R/cells}
case $SL in
  crc33) LIN=/gscratch/cheme/chiu/hfv_lintype_20261118/out
         A=(crc33 /gscratch/cheme/chiu/he_celltype_audit_20261118/lin/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif $LIN
            "$R/cv/c33_*" ${2:-$LIN/tiles_keep.txt} orion-crc33/exp_channels "" $OUT 32 --eval crc33sec $E/crc33sec_pack_ids.txt) ;;
  crc36) LIN=/gscratch/cheme/chiu/hfv_crc36_20261118/out
         A=(crc36 /gscratch/cheme/chiu/hfv_crc36_20261118/lin/data__CRC36__P37_S79-CRC36-mask.ome.tif $LIN
            "$R/cv/c36*" ${2:-/gscratch/cheme/chiu/hfv_retrain2_20261121/ref/c36_train_ids.txt} c36/exp_channels c36_ $OUT 32
            --eval crc36ho $E/crc36ho_pool.txt) ;;
  *) echo "STOP slide"; exit 1 ;;
esac
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $SL sev $(cat $R/sev/COMMIT) hfv $(cat $R/hfv/COMMIT)"
export HFV=$R/hfv SEV_SRC=$R/sev/src PYTHONPATH=/gscratch/cheme/chiu/hfv_retrain_20261118/PixCell OMP_NUM_THREADS=1
/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python -u $R/scripts/cells_oc.py "${A[@]}" || { echo "STOP cells"; exit 1; }
echo "### $(date -Is) DONE"
