#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --job-name=r3cv
# hfv_retrain3_cellvit_20261122 (PREREG §3.1, §5.2): official CellViT-256 on real H&E chunks or generated images.
# Copy of cellvit_official_20261122/scripts/cvo_infer.slurm (same runner 467a54f = R/hfv copy, CellViT repo 05097e1,
# checkpoint, env, batch 32, native resolution, NVIDIA_TF32_OVERRIDE=0, cheme CPU), official path only, own dirs.
#   sbatch --array=0-<n-1> --output=$R/logs/%x-%A_%a.out cv3.slurm LISTFILE OUTSUB
#   LISTFILE = one zip path per line (task i = line i); output R/OUTSUB/<zip stem>/<image stem>.json
# Any failure is a STOP: no retry, no requeue.
set -euo pipefail
R=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; CV=/gscratch/cheme/chiu/crc16-20261024/cellvit
LIST=${1:?LIST}; SUB=${2:?OUTSUB}
Z=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST); T=$(basename $Z .zip)
echo "### $(date -Is) job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} node $(hostname) $T hfv $(cat $R/hfv/COMMIT) ckpt $(md5sum $CV/ckpt/CellViT-256.pth | cut -c1-8) runner $(md5sum $R/hfv/stages/run_cellvit_local.py | cut -c1-8)"
test -s $Z || { echo "STOP no $Z"; exit 1; }
OUT=$R/$SUB/$T; test ! -e $OUT || { echo "STOP $OUT exists"; exit 1; }
mkdir -p $R/$SUB
export PYTHONPATH=$CV/CellViT OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1
cd $R/hfv
SECONDS=0
$CV/env/bin/python -u stages/run_cellvit_local.py --zip $Z --out $OUT --checkpoint $CV/ckpt/CellViT-256.pth \
  --cellvit-repo $CV/CellViT --batch-size 32 --no-skip-existing --postprocess official --magnification 40 \
  --bundle-zip $OUT.zip 2>&1 | grep -v -E "Patches:|Zipping:"
n=$(ls $OUT | grep -c '\.json$'); e=$(unzip -Z1 $Z | grep -c '\.png$')
[ "$n" = "$e" ] || { echo "STOP $OUT has $n files, expected $e"; exit 1; }
echo "### $(date -Is) done $T $n files in ${SECONDS}s"
