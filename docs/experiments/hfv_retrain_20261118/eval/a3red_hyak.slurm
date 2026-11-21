#!/bin/bash
#SBATCH --job-name=a3-red
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_a3_20261120/logs/%x-%j.out
# hfv_retrain_20261118 追加 3, extra check AFTER the dec job 43443943 (not in the pre-result decisions; a deviation to be
# confirmed by the coordinator). In 43443943 the paper's Virchow2 R2 was reproduced to a median |diff| of 0.007 with the
# handover's reduced budget (max 10,000 training rows, 50 iterations), while UNI (500,000 / 50,000 rows: 0.316 / 0.115)
# and CTransPath (0.500) were not. This job runs UNI and CTransPath with that same reduced budget on the same features,
# targets and folds (old 800, old 715, new 715), so the README can say whether the paper's UNI / CTransPath runs match it.
# Inputs: H/dec_features.tar (feat/uni33, feat/ctp33), H/dec_results.tar (targets, cv, keep800.txt), paper folds/CSVs from
# S3 share_space. Outputs: S3 figures_rerender/a3/dec_red/ and H/dec_red_results.tar (both must not exist). No retry.
set -uo pipefail
H=/gscratch/cheme/chiu/hfv_a3_20261120; F=/tmp/a3r_$SLURM_JOB_ID; I=$F/dec; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
S=s3://bagherilab-working/pohao/hfv_retrain_20261118; OLD=s3://bagherilab-working/pohao/share_space; SD=$S/figures_rerender/a3/dec_red
PY=$C/env/bin/python
module load coenv/aws/2.17.32
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
[ -e $F ] && { echo "STOP $F exists"; exit 1; }
[ -e $H/dec_red_results.tar ] && { echo "STOP $H/dec_red_results.tar exists"; exit 1; }
[ -n "$(aws s3 ls $SD/ 2>/dev/null)" ] && { echo "STOP $SD exists"; exit 1; }
mkdir -p $I/paper $I/logs $I/probe $F/pix_d1fc8a5
trap 'rm -rf $F' EXIT
tar -xzf $H/pixcell_d1fc8a5.tar.gz -C $F/pix_d1fc8a5 || { echo "STOP d1fc8a5 tar"; exit 1; }
tar -xf $H/dec_features.tar -C $I feat/uni33 feat/ctp33 || { echo "STOP features tar"; exit 1; }
tar -xf $H/dec_results.tar -C $I targets/old16_800 targets/old16_715 targets/new16_715 targets/old7_800 targets/old7_715 \
    targets/new7_715 cv keep800.txt || { echo "STOP results tar"; exit 1; }
for n in uni_16 ctranspath_07; do
  aws s3 cp --only-show-errors $OLD/05_results/results/a1_probe_mlp_spatial/$n/cv_splits.json $I/paper/cv_$n.json || { echo "STOP cv $n"; exit 1; }
done
links () { local list=$1 src=$2 dst=$3 suf=$4; [ -e $dst ] && { echo "STOP $dst exists"; exit 1; }; mkdir -p $dst
  while read -r t; do [ -e $src/$t$suf ] || { echo "STOP missing $src/$t$suf"; exit 1; }; ln -s $src/$t$suf $dst/$t$suf; done < $list; }
links $I/keep800.txt $I/feat/uni33 $I/fsub/uni715 _uni_tokens.npy
links $I/keep800.txt $I/feat/ctp33 $I/fsub/ctp715 _patches.npy
export PYTHONPATH=$F/pix_d1fc8a5:$H/pylib
cd $F/pix_d1fc8a5 || exit 1
PRB () { local name=$1 fd=$2 tg=$3 cv=$4 suf=$5; shift 5; [ -e $I/probe/$name ] && { echo "STOP probe/$name exists"; return 1; }
  echo "### $(date -Is) probe $name start"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY -m src.a1_probe_mlp_spatial.main \
    --features-dir $fd --targets-path $I/targets/$tg/mask_targets_T1_spatial.npy --tile-ids-path $I/targets/$tg/tile_ids.txt \
    --target-names-path $I/targets/$tg/target_names_T1_spatial.json --cv-splits-path $cv --out-dir $I/probe/$name \
    --feature-suffix $suf --batch-size 2048 "$@" > $I/logs/probe_$name.log 2>&1; local rc=$?
  rm -f $I/probe/$name/_X_memmap.npy; echo "### $(date -Is) probe $name rc $rc"; [ $rc -eq 0 ] || echo "probe_$name rc $rc" >> $I/A3RED_FAILED; return $rc; }
export -f PRB; export PY I
RED="--max-train-rows 10000 --max-iter 50 --hidden-layer-sizes 128,32 --n-jobs 2"
{
  echo "uni_old800_red $I/feat/uni33 old16_800 $I/paper/cv_uni_16.json _uni_tokens.npy $RED"
  echo "uni_old715_red $I/fsub/uni715 old16_715 $I/cv/cv_uni_16_kept.json _uni_tokens.npy $RED"
  echo "uni_new715_red $I/fsub/uni715 new16_715 $I/cv/cv_uni_16_kept.json _uni_tokens.npy $RED"
  echo "ctp_old800_red $I/feat/ctp33 old7_800 $I/paper/cv_ctranspath_07.json _patches.npy $RED"
  echo "ctp_old715_red $I/fsub/ctp715 old7_715 $I/cv/cv_ctranspath_07_kept.json _patches.npy $RED"
  echo "ctp_new715_red $I/fsub/ctp715 new7_715 $I/cv/cv_ctranspath_07_kept.json _patches.npy $RED"
} > $I/probes_red.txt
xargs -P 3 -L 1 bash -c 'PRB "$@"' _ < $I/probes_red.txt
[ -f $I/A3RED_FAILED ] && { echo "failed probes:"; cat $I/A3RED_FAILED; }
cd $I && aws s3 sync --only-show-errors $I/probe $SD/probe && aws s3 sync --only-show-errors $I/logs $SD/logs && echo "upload ok"
tar -cf $H/dec_red_results.tar probe logs probes_red.txt $( [ -f A3RED_FAILED ] && echo A3RED_FAILED )
echo "### $(date -Is) done"
[ ! -f $I/A3RED_FAILED ]
