#!/bin/bash
#SBATCH --account=bagherilab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/pohaoc2/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122 (PREREG §3.5): one paper variant on official-CellViT masks, one Tillicum H200. Copy of
# hfv_retrain_20261118 tr_variant.slurm with only the run folder changed (data from setup3.sh). Same PixCell cb62fa7,
# train_wrap.py, micro 80 = effective batch 80, 20 epochs, seed 42, no TF32 override, checkpoints every 2nd epoch + the
# last to this run's S3 outputs/ (presign3.py slots). 4:00:00 at most ($3.60).
#   sbatch --job-name=RUN tools/tr3.slurm RUN CONFIG [fetch]
#   tr3_concat config_controlnet_exp_a1_concat | tr3_grouped config_controlnet_exp | tr3_bypass config_controlnet_exp_a2_bypass
#   | tr3_perch config_controlnet_exp_a1_per_channel. First submission starts fresh; "fetch" only after a time limit.
set -u
RUN=$1; CFG=$2; FETCH=${3:-}
W=/gpfs/scrubbed/pohaoc2/hfv_retrain3_cellvit_20261122; P=$W/PixCell; D=$W/data/orion-crc33; LK=$W/links_$RUN.json
PY=/gpfs/projects/bagherilab/pohaoc2/envs/pixcell/bin/python; LOG=$W/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out
export MPLBACKEND=Agg HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache TORCH_HOME=$W/cache/torch TRITON_CACHE_DIR=$W/cache/triton
mkdir -p $W/cache
[ -f $W/SETUP_OK ] && [ -f $LK ] && [ -f $P/configs/$CFG.py ] || { echo "STOP setup / links / config missing"; exit 1; }
echo "== $(date -Is) $(hostname) job $SLURM_JOB_ID run $RUN config $CFG fetch ${FETCH:-no}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "TF32 override: ${NVIDIA_TF32_OVERRIDE:-none}"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'matmul.allow_tf32', torch.backends.cuda.matmul.allow_tf32, 'cudnn.allow_tf32', torch.backends.cudnn.allow_tf32)"
python3 -c "import json,urllib.request; u=json.load(open('$LK'))['get']['kit']; print('S3 reach', urllib.request.urlopen(urllib.request.Request(u, headers={'Range': 'bytes=0-0'}), timeout=30).status)" || echo "WARNING S3 not reachable from the node"
( while sleep 120; do echo "gpu $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    python3 $W/kit/s3io.py put $LK log_${RUN}_h200 $LOG >/dev/null 2>&1; done ) & LP=$!
$PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print('index', {k: len(h[k]) for k in h})"
EXTRA=""; [ "$FETCH" = fetch ] && EXTRA="--fetch-resume"
$PY -u $W/kit/train_wrap.py --pixcell $P --config configs/$CFG.py --data-root $D \
  --work-dir $W/runs/$RUN --micro-bs 80 --save-every-epoch --upload-every 2 --links $LK --run-name $RUN \
  --resume auto $EXTRA
rc=$?; echo "== $(date -Is) train $RUN rc $rc"
kill $LP 2>/dev/null; python3 $W/kit/s3io.py put $LK log_${RUN}_h200 $LOG >/dev/null 2>&1
exit $rc
