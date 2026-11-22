#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --job-name=cvo_r1
# cellvit_official_20261122 (PREREG.md §4): stage-3 readouts 1-2 on the official CellViT JSONs. Builds G' = $D/RUN/g3 whose
# inputs / packs / gen / kfig2 are symlinks to the read-only stage-3 directory and whose cv/<src>_<set> are the official
# JSONs, then runs the unchanged hfv_retrain2 s3metrics.slurm body (pxscore.py r2score.py metrics + stats) on G'.
#   sbatch --output=$D/logs/%x-%j.out cvo_r1.slurm RUN
set -uo pipefail
RUN=${1:?RUN}
CC=/gscratch/cheme/chiu; D=$CC/cellvit_official_20261122; O=$D/$RUN; G=$O/g3; S3=$CC/hfv_retrain2_20261121/stage3
stop() { echo "$(date -Is) STOP $*"; exit 1; }
test -e $G && stop "$G exists"
mkdir -p $G/cv
for x in inputs packs gen kfig2; do ln -s $S3/$x $G/$x; done
for s in crc36ho crc33sec; do for m in N K R; do
  test -d $O/cv/${m}_$s || stop "no $O/cv/${m}_$s"; ln -s $O/cv/${m}_$s $G/cv/${m}_$s; done; done
sha256sum $S3/scripts/r2score.py $S3/scripts/pxscore.py $S3/scripts/s3metrics.slurm
bash $S3/scripts/s3metrics.slurm $G || stop "s3metrics"
echo "### $(date -Is) DONE"
