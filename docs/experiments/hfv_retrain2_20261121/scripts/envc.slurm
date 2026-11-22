#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=170G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 stage 2: method C ch7-9 for one slide (envc.py). Any failure is a STOP: no retry, no requeue.
#   sbatch --job-name=envc-SLIDE envc.slurm crc33|crc36
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; MXW=/gscratch/cheme/chiu/hfv_mx_20261118; N=/gscratch/cheme/chiu/hfv_crc36_20261118
export HFV=$MXW/code SEV=$R/sev MXLIB=$MXW/scripts MPLCONFIGDIR=$R/tmp_mpl OMP_NUM_THREADS=1 PYTHONPATH=$MXW/code
case $1 in
  crc33) A=(crc33 $MXW/lin/CRC33_01__P37_S76_01_A24_C59qX_E15_20220302_235136_561323-zlib.ome.tiff $MXW/lin/CRC33_01__markers.csv
            $MXW/raw/processed_crc33_index.json /gscratch/cheme/chiu/hfv_lintype_20261118/out/tiles_keep.txt
            $MXW/raw/o33_mx_channels.tar orion-crc33/exp_channels "" $R/envc/crc33 32 --oct160 $R/ref/oct160_ch7.npz) ;;
  crc36) A=(crc36 $N/lin/data__CRC36__P37_S79_Full_A24_C59qX_E15_20220302_235254_496641-zlib.ome.tiff $N/lin/data__CRC36__markers.csv
            $N/out/tiles_crc36.json $R/ref/c36_train_ids.txt $N/data/crc36_mx_channels.tar c36/exp_channels c36_ $R/envc/crc36 32) ;;
  *) echo "STOP slide"; exit 1 ;;
esac
mkdir -p $R/envc $R/tmp_mpl
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $1 sev $(cat $SEV/COMMIT)"
$MXW/venv/bin/python -u $R/scripts/envc.py "${A[@]}" || { echo "STOP envc"; exit 1; }
echo "### $(date -Is) DONE"
