#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_mx_20261118/logs/%x-%j.out
# step 2: CRC36 kept tiles (tiles_crc36.json minus ink), ONLY after repro33 reproduces CRC33_01.
# submit: DS=4 O2L=200 GLCL=120 sbatch --export=ALL --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) \
#           --job-name=mx-crc36 crc36.slurm
set -euo pipefail
W=/gscratch/cheme/chiu/hfv_mx_20261118
N=/gscratch/cheme/chiu/hfv_crc36_20261118
echo "== $(date -Is) $(hostname)"
export HFV=$W/code MPLCONFIGDIR=$W/tmp_mpl
$W/venv/bin/python -u $W/scripts/mx_channels.py \
  --mx-tiff $N/lin/data__CRC36__P37_S79_Full_A24_C59qX_E15_20220302_235254_496641-zlib.ome.tiff \
  --markers $N/lin/data__CRC36__markers.csv --tiles $N/out/tiles_crc36.json --max-ink 0.0 \
  --out-tar $N/data/crc36_mx_channels.tar --prefix orion-crc36/exp_channels --summary $N/out/mx_channels.json \
  --ds ${DS} --o2-L-um ${O2L} --glc-L-um ${GLCL} ${SMA:-} --workers 16 --expect-mpp 0.325
echo "== $(date -Is) done"
