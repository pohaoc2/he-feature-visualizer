#!/bin/bash
#SBATCH --job-name=tr_concat
#SBATCH --account=bagherilab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/pohaoc2/hfv_retrain_20261118/logs/%x-%j.out
# hfv_retrain_20261118: Concat (config_controlnet_exp_a1_concat) on one full Tillicum H200 (coordinator 2026-11-18;
# time limit 4:00:00 at $0.90 / h = $3.60 at most). Same as the Colab train:tr_concat task: PixCell cb62fa7, the same
# train_wrap.py (effective batch 80 = micro 80 x accumulation 1, 20 epochs, seed 42, epoch-end checkpoints, every 2nd
# epoch and the last uploaded to the same S3 slots ckpt_tr_concat_eNN_* so Colab can resume with --fetch-resume), the
# same data state (tools/setup.sh, run before on the login node), no TF32 override (Colab sets none; PixCell sets none;
# torch defaults). No downloads or pip here. All new files under /gpfs/scrubbed (caches too, not /gpfs/home).
set -u
W=/gpfs/scrubbed/pohaoc2/hfv_retrain_20261118; P=$W/PixCell; D=$W/data/orion-crc33
PY=/gpfs/projects/bagherilab/pohaoc2/envs/pixcell/bin/python; RUN=tr_concat; LOG=$W/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out
export MPLBACKEND=Agg HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache TORCH_HOME=$W/cache/torch TRITON_CACHE_DIR=$W/cache/triton
mkdir -p $W/cache
[ -f $W/SETUP_OK ] || { echo "STOP setup not done"; exit 1; }
echo "== $(date -Is) $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "TF32 override: ${NVIDIA_TF32_OVERRIDE:-none}"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'matmul.allow_tf32', torch.backends.cuda.matmul.allow_tf32, 'cudnn.allow_tf32', torch.backends.cudnn.allow_tf32)"
python3 -c "import json,urllib.request; u=json.load(open('$W/links.json'))['get']['kit']; print('S3 reach', urllib.request.urlopen(urllib.request.Request(u, headers={'Range': 'bytes=0-0'}), timeout=30).status)" || echo "WARNING S3 not reachable from the node: uploads go from the login node later"
( while sleep 120; do echo "gpu $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    python3 $W/kit/s3io.py put $W/links.json log_tr_concat_h200 $LOG >/dev/null 2>&1; done ) & LP=$!
$PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print('index', {k: len(h[k]) for k in h})"
$PY -u $W/kit/train_wrap.py --pixcell $P --config configs/config_controlnet_exp_a1_concat.py --data-root $D \
  --work-dir $W/runs/$RUN --micro-bs 80 --save-every-epoch --upload-every 2 --links $W/links.json --run-name $RUN \
  --resume auto --fetch-resume
rc=$?; echo "== $(date -Is) train $RUN rc $rc"
kill $LP 2>/dev/null; python3 $W/kit/s3io.py put $W/links.json log_tr_concat_h200 $LOG >/dev/null 2>&1
exit $rc
