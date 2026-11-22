#!/bin/bash
#SBATCH --job-name=tr_concat2
#SBATCH --account=bagherilab
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/scrubbed/pohaoc2/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 stage 3 (PREREG.md §5): Concat retrain 2 = model N, a copy of hfv_retrain_20261118/tillicum/
# tr_concat.slurm (model K = tr_concat/step_0006520) with only the data changed: L1 cell channels + their mask latents and
# method C ch7-9 (tillicum/setup2.sh, run before on the login node). Same PixCell cb62fa7, train_wrap.py, config
# config_controlnet_exp_a1_concat, effective batch 80, 20 epochs, seed 42, no TF32 override. One H200, 4:00:00 at most
# ($0.90 / h = $3.60). Checkpoints every 2nd epoch + the last to this run's S3 outputs/ (presign2.py slots).
# NOT submitted by the stage-2 worker: the user logs into Tillicum first.
set -u
W=/gpfs/scrubbed/pohaoc2/hfv_retrain2_20261121; P=$W/PixCell; D=$W/data/orion-crc33
PY=/gpfs/projects/bagherilab/pohaoc2/envs/pixcell/bin/python; RUN=tr_concat2; LOG=$W/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.out
export MPLBACKEND=Agg HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache TORCH_HOME=$W/cache/torch TRITON_CACHE_DIR=$W/cache/triton
mkdir -p $W/cache
[ -f $W/SETUP_OK ] || { echo "STOP setup not done"; exit 1; }
echo "== $(date -Is) $(hostname) job $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "TF32 override: ${NVIDIA_TF32_OVERRIDE:-none}"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'matmul.allow_tf32', torch.backends.cuda.matmul.allow_tf32, 'cudnn.allow_tf32', torch.backends.cudnn.allow_tf32)"
python3 -c "import json,urllib.request; u=json.load(open('$W/links_tr_concat2.json'))['get']['kit']; print('S3 reach', urllib.request.urlopen(urllib.request.Request(u, headers={'Range': 'bytes=0-0'}), timeout=30).status)" || echo "WARNING S3 not reachable from the node: uploads go from the login node later"
( while sleep 120; do echo "gpu $(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    python3 $W/kit/s3io.py put $W/links_tr_concat2.json log_tr_concat2_h200 $LOG >/dev/null 2>&1; done ) & LP=$!
$PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print('index', {k: len(h[k]) for k in h})"
$PY -u $W/kit/train_wrap.py --pixcell $P --config configs/config_controlnet_exp_a1_concat.py --data-root $D \
  --work-dir $W/runs/$RUN --micro-bs 80 --save-every-epoch --upload-every 2 --links $W/links_tr_concat2.json --run-name $RUN \
  --resume auto --fetch-resume
rc=$?; echo "== $(date -Is) train $RUN rc $rc"
kill $LP 2>/dev/null; python3 $W/kit/s3io.py put $W/links_tr_concat2.json log_tr_concat2_h200 $LOG >/dev/null 2>&1
exit $rc
