#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
# hfv_retrain2 stage 3 (PREREG §9.2, §9.3): CellViT exactly as sim2real app2_run.slurm / app8_cv.slurm (unmodified hfv
# run_cellvit_local.py, CellViT-256.pth, batch 32, NVIDIA_TF32_OVERRIDE=0, native resolution, CPU). One array task per
# zip named in NAMES (comma-separated, without .zip). Writes only G/cv/<name>.
#   sbatch --array=0-<n-1> -J r2cv --output=G/logs/%x_%A_%a.out s3cv.slurm G NAMES
set -euo pipefail
G=${1:?G}; IFS=, read -ra NAMES <<< "${2:?NAMES}"
CV=/gscratch/cheme/chiu/crc16-20261024/cellvit
T=${NAMES[$SLURM_ARRAY_TASK_ID]}; Z=$G/zips/$T.zip; O=$G/cv/$T
echo "start $(date -Is) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=$(hostname) zip=$T hfv=$(cat $CV/hfv/COMMIT) ckpt=$(md5sum $CV/ckpt/CellViT-256.pth | cut -c1-8)"
test -s $Z || { echo "STOP: no $Z"; exit 1; }
test ! -e $O || { echo "STOP: $O exists"; exit 1; }
export PYTHONPATH=$CV/CellViT OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NVIDIA_TF32_OVERRIDE=0
cd $CV/hfv
$CV/env/bin/python -u stages/run_cellvit_local.py --zip $Z --out $O --checkpoint $CV/ckpt/CellViT-256.pth \
  --cellvit-repo $CV/CellViT --batch-size 32 --no-skip-existing 2>&1 | grep -v -E "Patches:|Zipping:"
n=$(ls $O/*.json | wc -l); e=$(unzip -Z1 $Z | wc -l)
[ "$n" = "$e" ] || { echo "STOP: $T has $n CellViT files, expected $e"; exit 1; }
echo "done $T $n files $(date -Is)"
