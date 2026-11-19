#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain_20261118/logs/%x-%j.out
# hfv_retrain_20261118 CPU smoke test of the Colab kit before any GPU session (no GPU here):
#  1. data root as boot.sh builds it: orion-crc33.zip (own S3), the seven old cell channels + exp_index moved aside,
#     the new CRC33_01 cell channels (lintype_render_20261118 new_exp_channels.tar, 9,419 tiles) unpacked in place;
#  2. train_wrap.py with each of the four variant configs, micro-batch 2 (accumulation 40), 1 optimizer step, on CPU:
#     catches config / dataset / model-construction errors. A failure inside the attention kernel (xformers needs CUDA)
#     after the first forward started counts as "reached the training step".
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_retrain_20261118; PY=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/env/bin/python
set +u; module load coenv/aws/2.17.32; set -u
echo "== $(date -Is) $(hostname)"
D=$R/data/orion-crc33
if [ ! -f $R/data/ASSEMBLED_33 ]; then
  aws s3 cp --quiet s3://bagherilab-working/pohao/share_space/02_datasets/orion-crc33.zip $R/data/orion-crc33.zip
  ls -l $R/data/orion-crc33.zip
  (cd $R/data && unzip -q orion-crc33.zip) && rm $R/data/orion-crc33.zip
  mkdir -p $R/data/old33
  for c in cell_masks cell_type_healthy cell_type_cancer cell_type_immune cell_state_prolif cell_state_nonprolif cell_state_dead; do
    mv $D/exp_channels/$c $R/data/old33/$c; done
  mv $D/metadata/exp_index.hdf5 $R/data/old33/exp_index.hdf5
  tar -xf /gscratch/cheme/chiu/hfv_lintype_20261118/data/new_exp_channels.tar -C $R/data
  touch $R/data/ASSEMBLED_33
fi
for c in $D/exp_channels/*; do echo "$(ls $c | wc -l) $c"; done
$PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print({k: len(h[k]) for k in h})"
cd $R/PixCell
for cfg in config_controlnet_exp_a1_concat config_controlnet_exp config_controlnet_exp_a2_bypass config_controlnet_exp_a1_per_channel; do
  echo "=== $cfg"
  CUDA_VISIBLE_DEVICES= timeout 40m $PY -u $R/kit/train_wrap.py --pixcell $R/PixCell --config configs/$cfg.py --data-root $D \
    --work-dir $R/smoke/$cfg --micro-bs 2 --max-steps 1 --num-workers 2 > $R/smoke/$cfg.log 2>&1
  echo "rc $?"; grep -E "train_wrap|Starting ControlNet|total_steps|Error|error" $R/smoke/$cfg.log | head -12; tail -n 4 $R/smoke/$cfg.log
done
echo "== done $(date -Is)"
