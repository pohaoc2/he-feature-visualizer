#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=0:20:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
R=/gscratch/cheme/chiu/hfv_retrain2_20261121
export HFV=/gscratch/cheme/chiu/hfv_lintype_20261118/hfv SEV_SRC=$R/sev3/src
/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python -u $R/scripts/dbg_l1.py \
  /gscratch/cheme/chiu/he_celltype_audit_20261118/lin/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif $R/res/crc33sec/tiles.csv \
  10240_11520,10240_14336,10240_14848,10240_17920,0_10752
