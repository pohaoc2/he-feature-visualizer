#!/bin/bash
#SBATCH --account=bagherilab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/pohaoc2/hfv_retrain_20261118/logs/%x-%j.out
# hfv_retrain_20261118: one paper variant on one full Tillicum H200 (user decision 2026-11-18: Grouped, Grouped + bypass
# and Per-channel move from Colab to Tillicum like Concat; nothing from the Colab runs is reused).
#   sbatch --job-name=RUN tools/tr_variant.slurm RUN CONFIG [fetch]
# RUN / CONFIG = the Colab train task's (tr_grouped config_controlnet_exp, tr_bypass config_controlnet_exp_a2_bypass,
# tr_perch config_controlnet_exp_a1_per_channel); micro 80 (Colab auto on a >= 70 GB card), effective batch 80, 20 epochs,
# seed 42, the same train_wrap.py, checkpoints every 2nd epoch + the last to the Colab S3 slots ckpt_RUN_eNN_*.
# Links: W/links_RUN.json (presign.py TAG RUN_h200, one day, mode 600, not committed). The first submission starts fresh
# (no --fetch-resume, so nothing a Colab runtime may have left under these names can be picked up); a resubmission after a
# time limit passes "fetch" to resume from the newest checkpoint on S3. No TF32 override (as Colab and PixCell).
set -u
RUN=$1; CFG=$2; FETCH=${3:-}
W=/gpfs/scrubbed/pohaoc2/hfv_retrain_20261118; P=$W/PixCell; D=$W/data/orion-crc33; LK=$W/links_$RUN.json
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
