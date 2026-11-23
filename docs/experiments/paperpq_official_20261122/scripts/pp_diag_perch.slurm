#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#SBATCH --job-name=pp_diag
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%j.out
# paperpq_official_20261122 gate diagnosis (Per-channel). Paper sidecars from S3 si_a1_a2/tiles into node /tmp; same PYTHONPATH as pp_score.slurm.
set -euo pipefail
CC=/gscratch/cheme/chiu; D=$CC/paperpq_official_20261122; HFV=$CC/hfv_lintype_20261118/hfv; CA=$CC/he_celltype_audit_20261118
S=s3://bagherilab-working/pohao/share_space/04_inference/si_a1_a2/tiles; Wd=/tmp/pp_diag_$SLURM_JOB_ID; trap 'rm -rf $Wd' EXIT
module load coenv/aws/2.17.32
for v in a1_per_channel a1_concat a2_bypass_full_tme a2_off_shelf; do mkdir -p $Wd/$v
  aws s3 cp --only-show-errors --recursive --exclude "*" --include "*_cellvit_instances.json" $S/$v/ $Wd/$v/; echo "$v $(ls $Wd/$v | wc -l) json"; done
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$CA/pylib:$CC/slidesel-20261023/pylib:$CC/crc16-20261024/pylib:$CC/hfv_figs_20261118/PixCell:$HFV \
  $CC/crc16-20261024/cellvit/env/bin/python -u $D/scripts/pp_diag_perch.py $D $Wd
