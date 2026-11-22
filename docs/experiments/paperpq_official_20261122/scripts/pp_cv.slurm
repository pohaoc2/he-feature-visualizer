#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pp_cv
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%A_%a.out
# paperpq_official_20261122 (PREREG.md §6): CellViT-256 on the generated images, both post-processing paths.
# Same settings as cellvit_official_20261122 cvo_infer.slurm: hfv run_cellvit_local.py at 467a54f (its frozen copy in
# cellvit_official_20261122/hfv, read-only), CellViT repo 05097e1, crc16 env, CellViT-256.pth, batch 32, native resolution,
# NVIDIA_TF32_OVERRIDE=0, cheme CPU x 8.
#   sbatch --array=0-<n-1> pp_cv.slurm ITEMS       ITEMS = comma-separated ROOT:VARIANT:PATH (PATH official|runner)
# Images come from D/gen/ROOT.tar (out/tiles/VARIANT/*.png), zipped flat in node /tmp. Writes only D/cv/PATH/ROOT__VARIANT
# (+ D/inst/ROOT__VARIANT.npz for official). Any failure is a STOP: no retry, no requeue.
set -euo pipefail
IFS=, read -ra ITEMS <<< "${1:?ITEMS}"
CC=/gscratch/cheme/chiu; D=$CC/paperpq_official_20261122; CVO=$CC/cellvit_official_20261122; CV=$CC/crc16-20261024/cellvit
IFS=: read -r ROOT V PP <<< "${ITEMS[$SLURM_ARRAY_TASK_ID]}"
Wd=/tmp/pp_cv_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; trap 'rm -rf $Wd' EXIT
echo "### $(date -Is) job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} node $(hostname) $PP $ROOT $V hfv $(cat $CVO/hfv/COMMIT) ckpt $(md5sum $CV/ckpt/CellViT-256.pth | cut -c1-8)"
OUT=$D/cv/$PP/${ROOT}__$V; test ! -e $OUT || { echo "STOP $OUT exists"; exit 1; }
mkdir -p $Wd $D/cv/$PP $D/inst
tar -xf $D/gen/$ROOT.tar -C $Wd out/tiles/$V
Z=$Wd/${ROOT}__$V.zip
e=$($CV/env/bin/python -I -c "
import sys, zipfile, pathlib
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
fs = sorted(src.glob('*.png'))
with zipfile.ZipFile(dst, 'x') as z:
    for f in fs: z.write(f, f.name)
print(len(fs))" $Wd/out/tiles/$V $Z)
[ "$e" = 100 ] || { echo "STOP $Z has $e images"; exit 1; }
EXTRA=()
[ $PP = official ] && EXTRA=(--postprocess official --magnification 40 --save-instances $D/inst/${ROOT}__$V.npz)
[ $PP = official ] || [ $PP = runner ] || { echo "STOP path $PP"; exit 1; }
export PYTHONPATH=$CV/CellViT OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1
cd $CVO/hfv
SECONDS=0
$CV/env/bin/python -u stages/run_cellvit_local.py --zip $Z --out $OUT --checkpoint $CV/ckpt/CellViT-256.pth \
  --cellvit-repo $CV/CellViT --batch-size 32 --no-skip-existing "${EXTRA[@]}" 2>&1 | grep -v -E "Patches:|Zipping:"
n=$(ls $OUT/*.json | wc -l)
[ "$n" = "$e" ] || { echo "STOP $OUT has $n files, expected $e"; exit 1; }
echo "### $(date -Is) done $PP $ROOT $V $n files in ${SECONDS}s"
