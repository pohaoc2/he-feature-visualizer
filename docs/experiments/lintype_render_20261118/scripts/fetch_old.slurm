#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --job-name=ltr-fetch
#SBATCH --output=/gscratch/cheme/chiu/hfv_lintype_20261118/logs/%x-%j.out
# lintype_render_20261118 step 0: the current training channels (7 cell channels, all tiles) + exp_index, one tar,
# ranged reads of s3://bagherilab-working/pohao/share_space/02_datasets/orion-crc33.zip (own S3, read only).
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_lintype_20261118
PY=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python
set +u; module load coenv/aws/2.17.32; set -u
echo "== $(date -Is) $(hostname)"
test -s $N/data/old_channels.tar && { echo "STOP: $N/data/old_channels.tar exists (new dirs only)"; exit 1; }
TAR=$N/data/old_channels.tar.part $PY -u $N/scripts/zip_extract.py s3://bagherilab-working/pohao/share_space/02_datasets/orion-crc33.zip $N/data \
  orion-crc33/metadata/ orion-crc33/exp_channels/cell_masks/ orion-crc33/exp_channels/cell_type_healthy/ \
  orion-crc33/exp_channels/cell_type_cancer/ orion-crc33/exp_channels/cell_type_immune/ \
  orion-crc33/exp_channels/cell_state_prolif/ orion-crc33/exp_channels/cell_state_nonprolif/ \
  orion-crc33/exp_channels/cell_state_dead/
mv $N/data/old_channels.tar.part $N/data/old_channels.tar
ls -la $N/data/old_channels.tar
echo "== done $(date -Is)"
