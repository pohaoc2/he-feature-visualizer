#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --job-name=cvo_s3
# cellvit_official_20261122 (PREREG.md §4): stage-3 readout 3 and the app-8 gate reference on the official CellViT JSONs in
# $D/RUN/cv, with the unchanged scripts that produced the old numbers:
#   1) app8_stats.py stats (sim2real app8/code copy)       -> $D/RUN/s3/results_app8_official.json (K-side gate reference)
#   2) r2_app8.py (hfv_retrain2 stage3/scripts copy)       -> $D/RUN/s3/r2_app8_official.json (decision inputs: Δ NP-KP, ND-KD)
#   3) sensitivity: r2_app8.py with app8_stats NB = 10,000, SEED = 42 (cvo_r2app8_b10k.py), official and old runner JSONs
#   sbatch --output=$D/logs/%x-%j.out cvo_stage3.slurm RUN
# Any failure is a STOP: no retry, no requeue.
set -uo pipefail
RUN=${1:?RUN}
CC=/gscratch/cheme/chiu; D=$CC/cellvit_official_20261122; O=$D/$RUN; CVD=$O/cv; S=$O/s3
A=$CC/sim2real_apps_20261120; S3=$CC/hfv_retrain2_20261121/stage3; PY=$CC/crc16-20261024/cellvit/env/bin/python
S3C=$CC/msi_perturb_20261119/s3b_code; CELLS=$CC/msi_perturb_20261119/s3b_crc16b/cells; L1=$CC/hfv_retrain2_20261121/eval/crc16_L1
SEV=$S3/dry/sev
stop() { echo "$(date -Is) STOP $*"; exit 1; }
test -e $S && stop "$S exists"; mkdir -p $S
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) run $RUN"
cmp $A/app8/code/app8_stats.py $SEV/app8_stats.py || stop "app8_stats.py copies differ"
sha256sum $A/app8/code/app8_stats.py $SEV/app10_stats.py $S3/scripts/r2_app8.py $D/scripts/cvo_r2app8_b10k.py
for a in RAW KP KD CP CD KE NP ND; do test -d $CVD/$a || stop "no $CVD/$a"; done
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONDONTWRITEBYTECODE=1
$PY -u $A/app8/code/app8_stats.py stats $S3C $CELLS $CVD $CVD $S/results_app8_official.json > $S/app8_stats.log || stop app8_stats
tail -3 $S/app8_stats.log
$PY -u $S3/scripts/r2_app8.py $SEV $L1 $S3C $CELLS $CVD $CVD $CVD $CVD/ND $S/results_app8_official.json \
  $S/r2_app8_official.json > $S/r2_app8.log || stop r2_app8
head -1 $S/r2_app8.log
$PY -u $D/scripts/cvo_r2app8_b10k.py $SEV $S3/scripts/r2_app8.py $L1 $S3C $CELLS $CVD $CVD $CVD $CVD/ND \
  $S/results_app8_official.json $S/r2_app8_official_b10k.json > $S/r2_app8_b10k.log || stop "b10k official"
$PY -u $D/scripts/cvo_r2app8_b10k.py $SEV $S3/scripts/r2_app8.py $L1 $S3C $CELLS $A/app2/cv $A/app8/cv $S3/cv $S3/cv/ND \
  $A/app8/results/results_app8.json $S/r2_app8_runner_b10k.json > $S/r2_app8_runner_b10k.log || stop "b10k runner"
echo "### $(date -Is) DONE"
