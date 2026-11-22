#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pp_score
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%j.out
# paperpq_official_20261122 (PREREG.md §7.1): pp_score.py on Hyak CPU. PYTHONPATH = fidelity_variants fv_hema.slurm /
# mask_truenuc truenuc.slurm (numpy 2.5.4 pylibs, PixCell hfv_figs copy, hfv lintype copy with compare.py md5 932ebd72...).
set -uo pipefail
CC=/gscratch/cheme/chiu; D=$CC/paperpq_official_20261122; HFV=$CC/hfv_lintype_20261118/hfv; CA=$CC/he_celltype_audit_20261118
PY=$CC/crc16-20261024/cellvit/env/bin/python
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
echo "932ebd728e549a21eb668c59e98581f1  $HFV/stages/lin_typing/compare.py" | md5sum -c || { echo "STOP compare.py md5"; exit 1; }
md5sum $CC/hfv_figs_20261118/PixCell/tools/compute_ablation_metrics.py $CC/sim2real_apps_20261120/app10/pixcell/tools/compute_ablation_metrics.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=$CA/pylib:$CC/slidesel-20261023/pylib:$CC/crc16-20261024/pylib:$CC/hfv_figs_20261118/PixCell:$HFV \
  $PY -u $D/scripts/pp_score.py $D $SLURM_CPUS_PER_TASK || { echo "$(date -Is) STOP pp_score"; exit 1; }
echo "### $(date -Is) DONE"
