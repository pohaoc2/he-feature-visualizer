#!/bin/bash
#SBATCH --no-requeue
# hfv_retrain2 stage 3 (PREREG §9.2): readouts 1-2 (r2score.py metrics through pxscore.py, PixCell cb62fa7 = the app-10
# copy, UNI-2h = sev-conditioning-recovery assets, lpips from hfv_score_20261119/pylib, alexnet weights from its torch_home)
# then stats. CPU. Any failure STOP. Optional \$1 = another G (the dry run with K as a stand-in for N).
set -uo pipefail
CC=/gscratch/cheme/chiu; S=$CC/hfv_retrain2_20261121/stage3/scripts; G=${1:-$CC/hfv_retrain2_20261121/stage3}; PY=$CC/crc16-20261024/cellvit/env/bin/python
PIX=$CC/sim2real_apps_20261120/app10/pixcell; UNI=$CC/sev-conditioning-recovery-20261006/assets/uni-2h
D=$CC/hfv_score_20261119
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) pixcell tile_pipeline md5 $(md5sum $PIX/tools/stage3/tile_pipeline.py | cut -c1-32)"
export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$PIX:$D/pylib TORCH_HOME=$D/torch_home \
  HF_HOME=$G/cache/hf XDG_CACHE_HOME=$G/cache MPLBACKEND=Agg
cd $PIX
$PY -u $S/pxscore.py $S/r2score.py metrics $G $UNI || { echo "STOP metrics"; exit 1; }
$PY -I $S/r2score.py stats $G || { echo "STOP stats"; exit 1; }
echo "### $(date -Is) DONE"
