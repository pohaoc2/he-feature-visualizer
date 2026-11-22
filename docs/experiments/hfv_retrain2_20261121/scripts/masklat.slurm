#!/bin/bash
#SBATCH --job-name=r2-masklat
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 stage 2: SD3.5 VAE latents of the new cell_masks (PixCell cb62fa7 stage1_extract_features.py
# --vae-prefix mask_sd3_vae --skip-uni, unmodified), same command / env / weights as crc34_render_20261118 gpu34.slurm
# masklat34 (Hyak checkpoint A100). Steps write markers in K; a requeued job skips finished steps.
#   sbatch masklat.slurm MODE        (MODE = the cells.py mode that was built)
# Device check first (descriptive): latents of 64 CRC36 R0 masks from c36train.tar vs the stored ones (made on Colab).
set -uo pipefail
MODE=$1
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; K=$R/masklat; G=/tmp/r2ml_$SLURM_JOB_ID; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; PY=$C/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg HF_HOME=$G/cache/hf XDG_CACHE_HOME=$G/cache
module load coenv/aws/2.17.32
st() { echo "### $(date -Is) $*"; }
mkdir -p $K $G/cache; cd $G; trap 'rm -rf $G' EXIT
st "start job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) mode $MODE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p PixCell/pretrained_models
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C PixCell
ln -sfn $A/sd-3.5-vae PixCell/pretrained_models/sd-3.5-vae; (cd -P PixCell/pretrained_models/sd-3.5-vae && find . -type f | sort | xargs md5sum)
P=$G/PixCell
ML() { (cd $P && $PY -u stage1_extract_features.py --image-dir $1 --output-dir $2 --vae-prefix mask_sd3_vae --skip-uni); }
if [ ! -f $K/ok_devcheck ]; then
  mkdir -p dc/m dc/ref
  tar -tf /gscratch/cheme/chiu/hfv_crc36_20261118/data/c36train.tar | grep '^c36/exp_channels/cell_masks/' | sort | head -64 > dc/list
  sed 's#c36/exp_channels/cell_masks/\(.*\)\.png#c36/vae_features/\1_mask_sd3_vae.npy#' dc/list > dc/list2
  tar -xf /gscratch/cheme/chiu/hfv_crc36_20261118/data/c36train.tar -T dc/list -T dc/list2
  mv c36/exp_channels/cell_masks/* dc/m/; mv c36/vae_features/* dc/ref/
  ML dc/m dc/out
  $PY - dc/out dc/ref <<'PYEOF' | tee $K/devcheck.json
import json, sys, glob, os, numpy as np
o, r = sys.argv[1:3]
d, sc = [], []
for f in sorted(glob.glob(f"{r}/*.npy")):
    a, b = np.load(f"{o}/{os.path.basename(f)}").astype(np.float64), np.load(f).astype(np.float64)
    d.append(np.abs(a - b).max()); sc.append(np.abs(b).max())
print(json.dumps(dict(n=len(d), max_abs_diff=float(max(d)), median_max_abs_diff=float(np.median(d)), median_abs_max_value=float(np.median(sc)))))
PYEOF
  touch $K/ok_devcheck
fi
for SL in crc33 crc36 eval_crc33sec eval_crc36ho; do
  [ -f $K/ok_$SL ] && continue
  case $SL in crc33|crc36) T=$R/cells2/cells_${SL}_$MODE.tar ;; *) T=$R/cells2/${SL}_$MODE.tar ;; esac
  rm -rf in_$SL ml_$SL; mkdir in_$SL
  tar -xf $T -C in_$SL --wildcards '*/cell_masks/*' || { st "STOP extract $SL"; exit 1; }
  M=$(find in_$SL -type d -name cell_masks | head -1); n=$(ls $M | wc -l)
  ML $M ml_$SL; rc=$?; m=$(ls ml_$SL | grep -c _mask_sd3_vae.npy)
  st "$SL rc $rc latents $m of $n"
  [ $rc = 0 ] && [ $m = $n ] || { st "STOP masklat $SL"; exit 1; }
  tar -cf $K/masklat_${SL}_$MODE.tar.part ml_$SL && mv $K/masklat_${SL}_$MODE.tar.part $K/masklat_${SL}_$MODE.tar && touch $K/ok_$SL
done
st DONE
