#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --job-name=cvo_inf
# cellvit_official_20261122 (PREREG.md §3): CellViT-256 inference with hfv run_cellvit_local.py at 467a54f (frozen copy in
# $D/hfv), the same CellViT repo (05097e1), Python env, checkpoint, batch 32, native resolution, NVIDIA_TF32_OVERRIDE=0
# and cheme CPU as every earlier CellViT-256 run (app2_run.slurm, app8_cv.slurm, s3cv.slurm).
#   sbatch --array=0-<n-1> --output=$D/logs/%x-%A_%a.out cvo_infer.slurm RUN PATH NAMES [N_IMG]
#   PATH = official (--postprocess official --magnification 40 --save-instances) or runner (default path, smoke check only)
#   NAMES = comma-separated zip names (see zip_of below); N_IMG = first n members only (smoke test)
# Writes only $D/RUN/{cv,inst,zips}. Any failure is a STOP: no retry, no requeue.
set -euo pipefail
RUN=${1:?RUN}; PP=${2:?PATH}; IFS=, read -ra NAMES <<< "${3:?NAMES}"; NI=${4:-}
CC=/gscratch/cheme/chiu; D=$CC/cellvit_official_20261122; O=$D/$RUN
CV=$CC/crc16-20261024/cellvit; A=$CC/sim2real_apps_20261120; S3=$CC/hfv_retrain2_20261121/stage3
zip_of() {
  case $1 in
    RAW) echo $A/app2/zips/RAW.zip;;
    KP|KD|CP|CD|KE) echo $A/app8/zips/$1.zip;;
    NP|ND|N_crc36ho|K_crc36ho|R_crc36ho|N_crc33sec|K_crc33sec|R_crc33sec) echo $S3/zips/$1.zip;;
    F2_*) echo $D/$RUN/zips/$1.zip;;
    *) echo "STOP unknown name $1" >&2; exit 1;;
  esac
}
T=${NAMES[$SLURM_ARRAY_TASK_ID]}; Z=$(zip_of $T)
echo "### $(date -Is) job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} node $(hostname) $PP $T hfv $(cat $D/hfv/COMMIT) ckpt $(md5sum $CV/ckpt/CellViT-256.pth | cut -c1-8)"
test -s $Z || { echo "STOP no $Z"; exit 1; }
mkdir -p $O/cv $O/inst $O/zips
if [ -n "$NI" ]; then
  ZS=$O/zips/${T}_first$NI.zip
  $CV/env/bin/python -I -c "
import sys, zipfile
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst, 'x') as b:
    for m in sorted(a.namelist())[:n]: b.writestr(m, a.read(m))" $Z $ZS $NI
  Z=$ZS
fi
OUT=$O/cv/$T; test ! -e $OUT || { echo "STOP $OUT exists"; exit 1; }
EXTRA=()
[ $PP = official ] && EXTRA=(--postprocess official --magnification 40 --save-instances $O/inst/$T.npz)
[ $PP = official ] || [ $PP = runner ] || { echo "STOP path $PP"; exit 1; }
export PYTHONPATH=$CV/CellViT OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1
cd $D/hfv
SECONDS=0
$CV/env/bin/python -u stages/run_cellvit_local.py --zip $Z --out $OUT --checkpoint $CV/ckpt/CellViT-256.pth \
  --cellvit-repo $CV/CellViT --batch-size 32 --no-skip-existing "${EXTRA[@]}" 2>&1 | grep -v -E "Patches:|Zipping:"
n=$(ls $OUT/*.json | wc -l); e=$(unzip -Z1 $Z | wc -l)
[ "$n" = "$e" ] || { echo "STOP $OUT has $n files, expected $e"; exit 1; }
echo "### $(date -Is) done $PP $T $n files in ${SECONDS}s"
