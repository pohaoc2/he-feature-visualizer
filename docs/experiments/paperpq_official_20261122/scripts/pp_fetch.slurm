#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pp_fetch
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%j.out
# paperpq_official_20261122 (PREREG.md §4): fetch the paper-set inputs from own S3 (read-only) into the new Hyak dir.
#   - share_space/02_datasets/orion-crc33.zip (the paper's training / Fig 2 data root: old channels, UNI features)
#   - share_space/03_checkpoints: the paper checkpoints build_cache.py INFERENCE_VARIANTS names
#       production         -> concat_retrain_post_fix/full_seed_42/checkpoint/step_0002600   (Grouped, production)
#       a1_concat          -> concat/full_seed_42/checkpoint/step_0002600 (= tillicum_95470/concat_95470_0)
#       a1_per_channel     -> tillicum_95470/per_channel_95470_3/checkpoints/step_0002600 (+ its config.py)
#       a2_bypass_full_tme -> bypass/full_seed_42/checkpoint/step_0002600
# Writes only $D/in. Any failure is a STOP (no retry).
set -euo pipefail
D=/gscratch/cheme/chiu/paperpq_official_20261122; S=s3://bagherilab-working/pohao/share_space
module load coenv/aws/2.17.32
mkdir -p $D/in/ck
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
SECONDS=0
aws s3 cp --only-show-errors $S/02_datasets/orion-crc33.zip $D/in/orion-crc33.zip
echo "### $(date -Is) orion-crc33.zip $(stat -c %s $D/in/orion-crc33.zip) bytes ${SECONDS}s"
for pair in production:concat_retrain_post_fix/full_seed_42/checkpoint a1_concat:concat/full_seed_42/checkpoint \
            a1_per_channel:tillicum_95470/per_channel_95470_3/checkpoints a2_bypass_full_tme:bypass/full_seed_42/checkpoint; do
  v=${pair%%:*}; src=${pair#*:}
  aws s3 cp --only-show-errors --recursive $S/03_checkpoints/$src/step_0002600/ $D/in/ck/$v/step_0002600/
  echo "### $(date -Is) $v $(du -sb $D/in/ck/$v | cut -f1) bytes ${SECONDS}s"
done
aws s3 cp --only-show-errors $S/03_checkpoints/tillicum_95470/per_channel_95470_3/config.py $D/in/ck/a1_per_channel/config.py
aws s3 cp --only-show-errors $S/04_inference/si_a1_a2/cache.json.bak_n100 $D/in/cache.json.bak_n100
md5sum $D/in/ck/*/step_0002600/* $D/in/ck/a1_per_channel/config.py > $D/in/ck_md5.txt
echo "### $(date -Is) DONE ${SECONDS}s"
