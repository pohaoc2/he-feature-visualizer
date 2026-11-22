#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 stage 2: new 7 cell channels (cells.py) for one slide.
#   sbatch --job-name=cells-SLIDE-MODE cells.slurm crc33|crc36 L1|L1b       Any failure is a STOP: no retry, no requeue.
set -uo pipefail
SL=$1; MODE=$2
R=/gscratch/cheme/chiu/hfv_retrain2_20261121
case $MODE in L1) CK=res; COL=n_L1; F=tiles.csv ;; L1b) CK=l1b; COL=n_L1b; F=l1b.csv ;; *) echo "STOP mode"; exit 1 ;; esac
case $SL in
  crc33) LIN=/gscratch/cheme/chiu/hfv_lintype_20261118/out
         A=(crc33 /gscratch/cheme/chiu/he_celltype_audit_20261118/lin/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif $LIN
            $LIN/nucleus_shapes_lin_crc33_01.npz $LIN/tiles_keep.txt orion-crc33/exp_channels "" $R/cells 32
            --eval crc33sec $R/res/crc33sec/tiles.csv $R/$CK/crc33sec/$F $COL) ;;
  crc36) LIN=/gscratch/cheme/chiu/hfv_crc36_20261118/out
         A=(crc36 /gscratch/cheme/chiu/hfv_crc36_20261118/lin/data__CRC36__P37_S79-CRC36-mask.ome.tif $LIN
            $LIN/nucleus_shapes_lin_crc36.npz $R/ref/c36_train_ids.txt c36/exp_channels c36_ $R/cells 32
            --eval crc36ho $R/res/crc36ho/tiles.csv $R/$CK/crc36ho/$F $COL) ;;
  *) echo "STOP slide"; exit 1 ;;
esac
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $SL $MODE sev $(cat $R/sev2/COMMIT)"
export HFV=/gscratch/cheme/chiu/hfv_lintype_20261118/hfv SEV_SRC=$R/sev2/src OMP_NUM_THREADS=1
/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python -u $R/scripts/cells.py $MODE "${A[@]}" || { echo "STOP cells"; exit 1; }
echo "### $(date -Is) DONE"
