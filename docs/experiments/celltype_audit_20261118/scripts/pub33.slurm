#!/bin/bash
# celltype_audit_20261118 task 3: CELESTA and Astir on CRC33_01 (the generator's training slide) with the identical
# setup as sim-exp-validation celltype_pub (CRC16): same scripts, priors and package defaults (pub/ = byte copies of
# celltype_pub_20261118/run, md5 checked), same environments used read-only. Only differences: the input CSV, the
# output dir, and the CELESTA k-d depth (2 -> 4 core tiles of ~111k cells; CRC16 used depth 5 -> ~86k; the R script's
# own guidance is ~1e5 cells per run), plus a whole-slide CELESTA reference run as in celltype_pub.
#   sbatch --export=MODE=prep|tile|whole|ainit|afinal [--array=...] pub33.slurm
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/gscratch/cheme/chiu/he_celltype_audit_20261118/logs/%x-%A_%a.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
W=/gscratch/cheme/chiu/he_celltype_audit_20261118; P=$W/pub
E=/gscratch/cheme/chiu/crc16-20261024/celltype_pub_20261118      # environments only, read only
CSV=$W/lin/CRC33_01__P37_S76_01-CRC33_01.csv
echo "host $(hostname) mode $MODE task ${SLURM_ARRAY_TASK_ID:-} start $(date)"
case $MODE in
  prep)   $E/env_r/bin/Rscript $P/celltype_pub_celesta.R prep $CSV $P/celesta_prior.csv $W/celesta33 2 1000 ;;
  tile)   $E/env_r/bin/Rscript $P/celltype_pub_celesta.R tile $P/celesta_prior.csv $W/celesta33 $SLURM_ARRAY_TASK_ID ;;
  whole)  mkdir -p $W/celesta33_whole && cd $W/celesta33_whole && \
          $E/env_r/bin/Rscript $P/celltype_pub_celesta.R whole $CSV $P/celesta_prior.csv $W/celesta33_whole ;;
  ainit)  CUDA_VISIBLE_DEVICES= $E/env_astir/bin/python -u $P/celltype_pub_astir.py $CSV $P/astir_markers.yml $W/astir33 init $SLURM_ARRAY_TASK_ID ;;
  afinal) CUDA_VISIBLE_DEVICES= $E/env_astir/bin/python -u $P/celltype_pub_astir.py $CSV $P/astir_markers.yml $W/astir33 final ;;
esac
echo "end $(date)"
