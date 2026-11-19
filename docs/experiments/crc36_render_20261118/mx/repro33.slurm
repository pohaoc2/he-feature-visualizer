#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_mx_20261118/logs/%x-%j.out
# step 1b: reproduce orion-crc33 vasculature/oxygen/glucose for ALL 10,379 tiles with the settings found in step 1a
# (sweep_crc33.json), then compare with the training channels.
# submit: DS=4 O2L=200 GLCL=120 sbatch --export=ALL --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) \
#           --job-name=mx-repro33 repro33.slurm          (SMA=--sma-refine only if step 1a selects it; it did not)
set -euo pipefail
W=/gscratch/cheme/chiu/hfv_mx_20261118
echo "== $(date -Is) $(hostname)"
export HFV=$W/code MPLCONFIGDIR=$W/tmp_mpl
$W/venv/bin/python -u $W/scripts/mx_channels.py \
  --mx-tiff $W/lin/CRC33_01__P37_S76_01_A24_C59qX_E15_20220302_235136_561323-zlib.ome.tiff \
  --markers $W/lin/CRC33_01__markers.csv --tiles $W/raw/processed_crc33_index.json \
  --out-tar $W/out/crc33_repro_mx_channels.tar --prefix orion-crc33/exp_channels --summary $W/out/crc33_repro.json \
  --ds ${DS} --o2-L-um ${O2L} --glc-L-um ${GLCL} ${SMA:-} --workers 16 --expect-mpp 0.325
$W/venv/bin/python -u $W/scripts/compare_crc33.py $W/out/crc33_repro_mx_channels.tar $W/raw/o33_mx_channels.tar \
  $W/out/compare_crc33.json
echo "== $(date -Is) done"
