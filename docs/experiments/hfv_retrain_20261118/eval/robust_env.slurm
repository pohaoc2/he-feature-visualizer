#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_robust_20261120/logs/%x-%j.out
# hfv_retrain_20261118 PQ robustness readouts, step 0 (plan B, coordinator 2026-11-20): build the StarDist CPU venv on
# node-local /tmp (pip into gscratch from the login node stalled on small-file I/O), then copy it back as one tar to the
# new folder R/env. Every later job unpacks the tar at the same path /tmp/rob_sdenv (eval/robust_envsetup.sh), so the
# venv's absolute paths stay valid. Versions as planned: tensorflow-cpu 2.15.1, stardist 0.9.1, numpy < 2.
# Also fetches the 2D_versatile_he weights once into R/keras_home (the StarDist array then loads them from there).
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_robust_20261120; V=/tmp/rob_sdenv
stop() { echo "STOP $*"; exit 1; }
[ -e $R/env ] && stop "$R/env exists (new folders only)"
[ -e $V ] && stop "$V exists on $(hostname)"
mkdir -p $R/env
trap 'rm -rf $V /tmp/rob_pipcache_$SLURM_JOB_ID' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
module load coenv/python/3.11.9 || stop "module"
python3 --version; which python3
python3 -m venv $V || stop venv
$V/bin/pip install -q --cache-dir /tmp/rob_pipcache_$SLURM_JOB_ID --upgrade pip || stop "pip upgrade"
$V/bin/pip install -q --cache-dir /tmp/rob_pipcache_$SLURM_JOB_ID "tensorflow-cpu==2.15.1" "stardist==0.9.1" "numpy<2" \
  scipy pillow scikit-image || stop "pip install"
$V/bin/pip freeze > $R/env/pip_freeze.txt
echo "### $(date -Is) installed"
export KERAS_HOME=$R/keras_home TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg
$V/bin/python -c "
import tensorflow as tf, stardist, csbdeep, numpy, scipy, PIL, matplotlib
from stardist.models import StarDist2D
m = StarDist2D.from_pretrained('2D_versatile_he')
print('versions', tf.__version__, stardist.__version__, csbdeep.__version__, numpy.__version__, scipy.__version__, PIL.__version__)
print('thresholds', m.thresholds)
print('basedir', m.basedir)
" || stop "import / model"
tar -cf $R/env/sdenv.tar -C /tmp rob_sdenv || stop tar
ls -la $R/env; find $R/keras_home -type f | head
echo "### $(date -Is) DONE"
