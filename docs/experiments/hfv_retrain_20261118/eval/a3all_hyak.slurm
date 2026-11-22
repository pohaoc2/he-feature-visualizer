#!/bin/bash
#SBATCH --job-name=a3-all
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=40
#SBATCH --mem=170G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_a3_20261120/logs/%x-%j.out
# hfv_retrain_20261118 追加 3b (user decision 2026-11-21, README written before any result): decodability probes with all
# training rows (d1fc8a5 --max-train-rows 0). Same features / targets / folds / settings as dec job 43443943 otherwise.
#   rows : eval/a3all.py rows -- finite training rows per fold for every run; STOP if a reused run's cap would have bound
#   probes (new): Virchow2 old200 / old184 / new184 (handover budget except rows: 50 iterations, 128,32), then C34 UNI new
#          targets, 9,958 tiles, kept paper folds, shuffle baseline, 3 processes x 12 threads (memory estimate in README)
#   summ : eval/a3all.py summ -> r2all/ (render input) and a3all.json
# Inputs: H/dec_features.tar (feat/uni34, feat/vir33), H/dec_results.tar (targets, cv, probe, keep200.txt), paper folds and
# CSVs from S3 share_space. Work in node-local /tmp (gscratch file creation from compute nodes stalled earlier jobs).
# Outputs: S3 figures_rerender/a3/dec_all/ and H/dec_all_results.tar (both must not exist). STOP on failure; no retry.
set -uo pipefail
H=/gscratch/cheme/chiu/hfv_a3_20261120; F=/tmp/a3a_$SLURM_JOB_ID; I=$F/dec; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
S=s3://bagherilab-working/pohao/hfv_retrain_20261118; OLD=s3://bagherilab-working/pohao/share_space; SD=$S/figures_rerender/a3/dec_all
PY=$C/env/bin/python
module load coenv/aws/2.17.32
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"; df -h /tmp | tail -1
[ -e $F ] && { echo "STOP $F exists"; exit 1; }
[ -e $H/dec_all_results.tar ] && { echo "STOP $H/dec_all_results.tar exists"; exit 1; }
[ -n "$(aws s3 ls $SD/ 2>/dev/null)" ] && { echo "STOP $SD exists"; exit 1; }
mkdir -p $I/paper/results $I/logs $I/proball $F/pix_d1fc8a5
trap 'rm -rf $F' EXIT
tar -xzf $H/pixcell_d1fc8a5.tar.gz -C $F/pix_d1fc8a5 || { echo "STOP d1fc8a5 tar"; exit 1; }
tar -xf $H/dec_results.tar -C $I targets cv probe keep200.txt || { echo "STOP results tar"; exit 1; }
for n in virchow2_16 resnet50_07 uni_16_crc34; do
  aws s3 cp --only-show-errors $OLD/05_results/results/a1_probe_mlp_spatial/$n/cv_splits.json $I/paper/cv_$n.json || { echo "STOP cv $n"; exit 1; }
  mkdir -p $I/paper/results/$n
  aws s3 cp --only-show-errors $OLD/05_results/results/a1_probe_mlp_spatial/$n/mlp_spatial_probe_results.csv $I/paper/results/$n/ || { echo "STOP csv $n"; exit 1; }
done
$PY $H/scripts/a3all.py rows $I $I/rows.json > $I/logs/rows.log 2>&1; rc=$?; cat $I/logs/rows.log
[ $rc -eq 0 ] || { echo "STOP rows check"; exit 1; }
t0=$(date +%s); tar -xf $H/dec_features.tar -C $I feat/uni34 feat/vir33 || { echo "STOP features tar"; exit 1; }
echo "### features extracted in $(( $(date +%s)-t0 )) s"; du -sh $I/feat/*
mkdir -p $I/fsub/vir184
while read -r t; do [ -e $I/feat/vir33/${t}_patches.npy ] || { echo "STOP missing vir33/$t"; exit 1; }
  ln -s $I/feat/vir33/${t}_patches.npy $I/fsub/vir184/${t}_patches.npy; done < $I/keep200.txt
export PYTHONPATH=$F/pix_d1fc8a5:$H/pylib
cd $F/pix_d1fc8a5 || exit 1
PRB () { local name=$1 thr=$2 fd=$3 tg=$4 cv=$5 suf=$6; shift 6; [ -e $I/proball/$name ] && { echo "STOP proball/$name exists"; return 1; }
  echo "### $(date -Is) probe $name start"; local t0=$(date +%s)
  OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr $PY -m src.a1_probe_mlp_spatial.main \
    --features-dir $fd --targets-path $I/targets/$tg/mask_targets_T1_spatial.npy --tile-ids-path $I/targets/$tg/tile_ids.txt \
    --target-names-path $I/targets/$tg/target_names_T1_spatial.json --cv-splits-path $cv --out-dir $I/proball/$name \
    --feature-suffix $suf --batch-size 2048 --max-train-rows 0 "$@" > $I/logs/probe_$name.log 2>&1; local rc=$?
  rm -f $I/proball/$name/_X_memmap.npy; echo "### $(date -Is) probe $name rc $rc ($(( $(date +%s)-t0 )) s)"
  [ $rc -eq 0 ] || { echo "probe_$name rc $rc" >> $I/A3ALL_FAILED; tail -20 $I/logs/probe_$name.log; }; return $rc; }
RED="--max-iter 50 --hidden-layer-sizes 128,32 --n-jobs 2"
PRB vir_old200_all 10 $I/feat/vir33 old16_200 $I/paper/cv_virchow2_16.json _patches.npy $RED || exit 1
PRB vir_old184_all 10 $I/fsub/vir184 old16_184 $I/cv/cv_virchow2_16_kept.json _patches.npy $RED || exit 1
PRB vir_new184_all 10 $I/fsub/vir184 new16_184 $I/cv/cv_virchow2_16_kept.json _patches.npy $RED || exit 1
# memory sampler for the C34 probe (node RSS of this job's processes every 60 s)
( while sleep 60; do echo "$(date +%H:%M:%S) $(ps -u $USER -o rss= | awk '{s+=$1} END {printf "%.1f GB", s/1048576}') $(free -g | awk '/Mem:/{print "used "$3" GB"}')"; done ) > $I/logs/mem_c34.log 2>&1 &
MEMPID=$!
PRB c34_new_all 12 $I/feat/uni34 new16_c34 $I/cv/cv_uni_16_crc34_kept.json _uni_tokens.npy --compute-shuffle-baseline --n-jobs 3
c34rc=$?; kill $MEMPID 2>/dev/null; sort -k2 -n $I/logs/mem_c34.log | tail -1 | sed 's/^/### peak sampled RSS: /'
[ $c34rc -eq 0 ] || exit 1
$PY $H/scripts/a3all.py summ $I $I/rows.json $I/a3all.json $I/r2all > $I/logs/summ.log 2>&1 || { cat $I/logs/summ.log; echo "STOP summ"; exit 1; }
cat $I/logs/summ.log
cd $I
aws s3 cp --only-show-errors a3all.json $SD/a3all.json && aws s3 cp --only-show-errors rows.json $SD/rows.json \
  && aws s3 sync --only-show-errors r2all $SD/r2all && aws s3 sync --only-show-errors proball $SD/proball \
  && aws s3 sync --only-show-errors logs $SD/logs && echo "upload ok" || { echo "STOP upload"; exit 1; }
tar -cf $H/dec_all_results.tar a3all.json rows.json r2all proball logs || echo "results tar failed"
echo "### $(date -Is) done"
