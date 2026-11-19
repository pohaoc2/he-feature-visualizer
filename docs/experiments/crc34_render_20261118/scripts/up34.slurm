#!/bin/bash
#SBATCH --job-name=c34-up
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc34_20261118/logs/%x-%j.out
# crc34_render_20261118: CRC34 inputs for the Colab CRC34 cell (cvgate34 / cv34auto / uni34hub / feat34 / masklat34)
# and for the later frozen-inference evaluation, copied to s3://bagherilab-working/pohao/hfv_retrain_20261118/inputs/
# with the Hyak AWS module (no credentials leave Hyak; Colab reads them through one-day presigned links).
#   crc34_he_tiles_keep.tar -> crc34_he_tiles.tar (kept tiles only), crc34_exp_channels.tar (pass 1: cell_masks are
#   final; dead state is re-rendered after the CellViT gate), cvgate34_inputs.tar, crc34_mx_channels.tar
set -euo pipefail
N=/gscratch/cheme/chiu/hfv_crc34_20261118; S=s3://bagherilab-working/pohao/hfv_retrain_20261118/inputs
module load coenv/aws/2.17.32
echo "== $(date -Is) $(hostname)"
for p in "crc34_he_tiles_keep.tar crc34_he_tiles.tar" "cvgate34_inputs.tar cvgate34_inputs.tar" \
         "crc34_exp_channels.tar crc34_exp_channels.tar" "crc34_mx_channels.tar crc34_mx_channels.tar"; do
  set -- $p
  md5=$(md5sum $N/data/$1 | cut -d' ' -f1)
  aws s3 cp --only-show-errors $N/data/$1 $S/$2 --metadata md5=$md5
  r=$(aws s3api head-object --bucket bagherilab-working --key pohao/hfv_retrain_20261118/inputs/$2 --query ContentLength --output text)
  l=$(stat -c %s $N/data/$1)
  [ "$r" = "$l" ] || { echo "STOP size mismatch $2: S3 $r local $l"; exit 1; }
  echo "uploaded $1 -> $2 bytes $l md5 $md5"
done
echo "== $(date -Is) done"
