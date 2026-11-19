#!/bin/bash
#SBATCH --job-name=c34-gpu
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc34_20261118/logs/%x-%j.out
# crc34_render_20261118: the CRC34 GPU steps of the hfv_retrain_20261118 Colab cell (colab/boot.sh cvgate34 / cv34auto /
# uni34hub / feat34 / masklat34), moved to a Hyak checkpoint A100 because the Colab account runs one premium session at a
# time (coordinator 2026-11-18). Same commands as boot.sh:
#   cvgate34 : CellViT-256 (he-feature-visualizer stages/run_cellvit_local.py e88f450, unmodified) with TF32 off on the 50
#              seed-42 gate tiles, compared with the old JSONs by colab/cvgate34.py (README criterion, fixed before the run)
#   cv34     : all kept CRC34 tiles, only if cvgate34 does not pass
#   uni34hub : UNI-2h with the uni-2h config.json transform (colab/feat_uni.py hub = sim-exp-validation ea5d5bd9)
#   feat34   : SD3.5 VAE latents of the H&E tiles (PixCell cb62fa7 stage1_extract_features.py --skip-uni, unmodified)
#   masklat34: SD3.5 VAE latents of the new cell_masks (--vae-prefix mask_sd3_vae --skip-uni)
# Environment: the Hyak CellViT env of the CRC16 G-ckpt gate (crc16-20261024/cellvit/env; boot.sh: its freeze = the Colab
# req.txt + torch / torchvision / xformers + scikit-image / numba), read only. Pretrained weights: the Hyak files the Colab
# tars were made from (sev-conditioning-recovery-20261006/assets), md5 printed. Checkpoint partition: every step writes
# a marker, a requeued job skips finished steps. Outputs stay in N/gpu and go to the S3 run prefix as the Colab cell would.
set -uo pipefail
N=/gscratch/cheme/chiu/hfv_crc34_20261118; G=$N/gpu; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; PY=$C/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg HF_HOME=$G/cache/hf XDG_CACHE_HOME=$G/cache
module load coenv/aws/2.17.32
st() { echo "### $(date -Is) $*"; }
mkdir -p $G/out $G/cache; cd $G
st "start job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch, skimage, numba, numpy, timm, diffusers; print('env', torch.__version__, torch.version.cuda, skimage.__version__, numba.__version__, numpy.__version__, timm.__version__, diffusers.__version__)"
if [ ! -f $G/ok_prep ]; then
  rm -rf kit PixCell g34 he34 c34; mkdir -p kit PixCell/pretrained_models g34 he34 c34
  aws s3 cp --only-show-errors $S/inputs/kit.tar.gz - | tar -xz -C kit
  aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C PixCell
  for d in uni-2h sd-3.5-vae; do ln -sfn $A/$d PixCell/pretrained_models/$d; (cd -P PixCell/pretrained_models/$d && find . -type f | sort | xargs md5sum); done
  tar -xf $N/data/cvgate34_inputs.tar -C g34
  tar -xf $N/data/crc34_he_tiles_keep.tar -C he34
  (cd he34 && python3 -c "import zipfile,os; z=zipfile.ZipFile('he34.zip','w',zipfile.ZIP_STORED); [z.write(os.path.join('he',f)) for f in sorted(os.listdir('he'))]; z.close()")
  tar -xf $N/data/crc34_exp_channels.tar -C c34
  echo "he tiles $(ls he34/he | wc -l); gate $(wc -l < g34/ids.txt)"
  touch ok_prep; st prep
fi
P=$G/PixCell
CV="env NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$C/CellViT $PY -u $C/hfv/stages/run_cellvit_local.py --checkpoint $C/ckpt/CellViT-256.pth --cellvit-repo $C/CellViT --no-skip-existing"
if [ ! -f ok_cvgate34 ]; then
  $CV --zip g34/he_tiles.zip --out out/cvgate34; rc=$?
  $PY kit/cvgate34.py g34/ref out/cvgate34 out/cvgate34.json; cat out/cvgate34.json | head -c 3000; echo
  [ $rc = 0 ] && touch ok_cvgate34; st "cvgate34 rc $rc"
fi
if $PY -c "import json,sys; sys.exit(0 if json.load(open('out/cvgate34.json'))['PASS'] else 1)"; then
  st "cvgate34 PASS: cv34 not needed"
elif [ ! -f ok_cv34 ]; then
  st "cvgate34 not passed: running cv34"
  $CV --zip he34/he34.zip --out cv34 --batch-size 32; rc=$?; st "cv34 rc $rc"
  echo "cellvit json $(find cv34 -name '*.json' | wc -l) of $(ls he34/he | wc -l) tiles"
  [ $rc = 0 ] && tar -cf cellvit_crc34.tar cv34 && aws s3 cp --only-show-errors cellvit_crc34.tar $S/outputs/cellvit_crc34 && touch ok_cv34
fi
if [ ! -f ok_uni34hub ]; then
  rm -rf u34; $PY -u kit/feat_uni.py $P hub --image-dir $G/he34/he --output-dir $G/u34/features --skip-vae; rc=$?; st "uni34hub rc $rc"
  echo "uni $(ls u34/features | wc -l) of $(ls he34/he | wc -l)"
  [ $rc = 0 ] && tar -cf uni_crc34_hub.tar u34 && aws s3 cp --only-show-errors uni_crc34_hub.tar $S/outputs/uni_crc34_hub && touch ok_uni34hub
fi
if [ ! -f ok_feat34 ]; then
  rm -rf f34; (cd $P && $PY -u stage1_extract_features.py --image-dir $G/he34/he --output-dir $G/f34/vae_features --skip-uni); rc=$?
  st "feat34 rc $rc"; echo "vae $(ls f34/vae_features | wc -l) of $(ls he34/he | wc -l)"
  [ $rc = 0 ] && tar -cf feat_crc34.tar f34 && aws s3 cp --only-show-errors feat_crc34.tar $S/outputs/feat_crc34 && touch ok_feat34
fi
if [ ! -f ok_masklat34 ]; then
  M=$(find $G/c34 -type d -path '*exp_channels/cell_masks' | head -1)
  rm -rf masklat34; (cd $P && $PY -u stage1_extract_features.py --image-dir $M --output-dir $G/masklat34 --vae-prefix mask_sd3_vae --skip-uni); rc=$?
  st "masklat34 rc $rc"; echo "mask latents $(ls masklat34 | wc -l) of $(ls $M | wc -l)"
  [ $rc = 0 ] && tar -cf masklat_crc34.tar masklat34 && aws s3 cp --only-show-errors masklat_crc34.tar $S/outputs/masklat_crc34 && touch ok_masklat34
fi
ls ok_*; st done
