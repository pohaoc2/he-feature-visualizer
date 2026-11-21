#!/bin/bash
#SBATCH --job-name=a3-render
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_a3_20261120/logs/%x-%j.out
# hfv_retrain_20261118 追加 3: the figures job 43427989 never ran -- a6 (cohort_invariance_metrics.json), Fig 4
# (fig_combined_per_channel, which also writes si_channel_decodability_impact), si_threshold_sensitivity and
# si_patient_generalization. PixCell cb62fa7 scripts run unchanged through eval/figrender.py (guard before every script)
# in a NEW checkout F/PixCell with its own stand-in F/share-space-paper. Two versions, one per PQ reference:
#   pri = CellViT-256 on the real H&E (realref trees; C34 LOO tree made here with eval/realref.py tree loo)
#   sec = Lin input cell mask (PixCell's default scoring; scoreres trees)
# C10 LOO and C10 R2 are always the old inputs (no rebuilt C10 data).
# R2 source: A3_R2=old -> S3 share_space/05_results a1_probe_mlp_spatial (all seven folders, old channels);
#            A3_R2=<dir or s3:// prefix> -> <dir>/<name>/mlp_spatial_probe_results.csv for every folder present there, old for
#            the rest.
# Outputs: S3 hfv_retrain_20261118/figures_rerender/a3/<A3_TAG>/<ref>/... and H/out_<A3_TAG>.tar (both must not exist yet).
# A failing step is recorded (A3_FAILED) and reported, never retried.
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) --export=ALL,A3_R2=old,A3_TAG=r2old a3render_hyak.slurm
# Work root F = node-local /tmp/a3_<job> (new each job). On 2026-11-20/21 file creation under /gscratch/cheme from compute
# nodes took ~0.25-1.4 s per file (login node ~1 ms) and stalled jobs 43440886, 43441634; /gscratch/scrubbed was 98 % full
# (43441377). Only one tar is written to H. The job prints the time to create 20 files in H as a record.
set -uo pipefail
H=/gscratch/cheme/chiu/hfv_a3_20261120; F=/tmp/a3_$SLURM_JOB_ID; R=$F/PixCell; I=$F/inputs; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
S=s3://bagherilab-working/pohao/hfv_retrain_20261118; OLD=s3://bagherilab-working/pohao/share_space
A3_R2=${A3_R2:-old}; A3_TAG=${A3_TAG:-r2old}
export MPLBACKEND=Agg PYTHONPATH=$H/pylib:$H/scripts XDG_CACHE_HOME=$F/cache MPLCONFIGDIR=$F/cache/mpl A3_FONT_DIR=$H/fonts
PY=$C/env/bin/python
module load coenv/aws/2.17.32
mkdir -p $H/.iot_$SLURM_JOB_ID && t0=$(date +%s%N) && for i in $(seq 20); do : > $H/.iot_$SLURM_JOB_ID/f$i; done && echo "### create 20 files in H: $(( ($(date +%s%N)-t0)/1000000 )) ms"; rm -rf $H/.iot_$SLURM_JOB_ID
[ -e $F ] && { echo "STOP $F exists"; exit 1; }; mkdir -p $F; df -h /tmp | tail -1
trap '[ -d $F/out ] && [ ! -e $H/out_$A3_TAG.tar ] && tar -cf $H/out_${A3_TAG}_partial.tar -C $F out; rm -rf $F' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) R2=$A3_R2 tag=$A3_TAG"
O=$F/out/$A3_TAG
[ -e $H/out_$A3_TAG.tar ] && { echo "STOP $H/out_$A3_TAG.tar exists"; exit 1; }
[ -n "$(aws s3 ls $S/figures_rerender/a3/$A3_TAG/ 2>/dev/null)" ] && { echo "STOP S3 figures_rerender/a3/$A3_TAG exists"; exit 1; }
if [[ $A3_R2 == s3://* ]]; then mkdir -p $F/r2 && aws s3 sync --only-show-errors $A3_R2 $F/r2 || { echo "STOP r2 download"; exit 1; }; A3_R2=$F/r2; fi
mkdir -p $O $H/logs
# ---- inputs (once)
if [ ! -f $I/.ok ]; then
  [ -e $F/share-space-paper ] && { echo "STOP stand-in already exists before first setup"; exit 1; }
  mkdir $F/share-space-paper   # empty stand-in (plain directory); the real paper repo is only on the Mac
  mkdir -p $R $I/score $I/scoreres $I/realref $I/eval $I/old/results
  aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $R || { echo "STOP pixcell tar"; exit 1; }
  for t in score_loo_crc34eval score_abl_crc33sec score_crc34_crc34eval realcv_crc34eval; do
    aws s3 cp --only-show-errors $S/outputs/$t.tar - | tar -x -C $I/score || { echo "STOP input $t"; exit 1; }; done
  for t in scoreres_loo_crc33sec scoreres_loo_crc34eval scoreres_abl_crc33sec scoreres_crc34_crc34eval; do
    aws s3 cp --only-show-errors $S/outputs/$t.tar - | tar -x -C $I/scoreres || { echo "STOP input $t"; exit 1; }; done
  for t in realref_loo_crc33sec realref_abl_crc33sec realref_crc34_crc34eval; do
    aws s3 cp --only-show-errors $S/outputs/$t.tar - | tar -x -C $I/realref || { echo "STOP input $t"; exit 1; }; done
  for p in crc33sec crc34eval; do
    aws s3 cp --only-show-errors $S/inputs/eval/evalpack_$p.tar - | tar -x -C $I/eval || { echo "STOP evalpack $p"; exit 1; }; done
  aws s3 cp --only-show-errors $S/inputs/eval/crc33sec_p300_unpaired.json $I/eval/ || exit 1
  aws s3 sync --only-show-errors $OLD/04_inference/subchannel_loo_n300_crc10 $I/old/loo_crc10 --exclude "*" \
    --include "per_subchannel_summary.csv" --include "per_subchannel_layout_summary.csv" --include "group_vs_subchannel_consistency.csv" || exit 1
  aws s3 sync --only-show-errors $OLD/05_results/results/a1_probe_mlp_spatial $I/old/results/a1_probe_mlp_spatial || exit 1
  # C34 LOO against the primary reference (the tree was never produced): realref masks from CellViT on the real H&E
  PIXCELL=$R PYTHONPATH=$R:$H/pylib $PY $H/scripts/realref.py masks $I/score/realcv_crc34eval $I/refroot_crc34eval || { echo "STOP realref masks"; exit 1; }
  PIXCELL=$R PYTHONPATH=$R:$H/pylib $PY $H/scripts/realref.py tree loo $I/score/loo_crc34eval $I/refroot_crc34eval $I/realref/loo_crc34eval \
    || { echo "STOP realref tree loo crc34eval"; tail -30 $I/realref/loo_crc34eval/realref.log; exit 1; }
  tail -3 $I/realref/loo_crc34eval/realref.log
  touch $I/.ok
fi
for f in $I/scoreres/loo_crc33sec $I/scoreres/loo_crc34eval $I/realref/loo_crc33sec $I/realref/loo_crc34eval $I/old/loo_crc10; do
  for c in per_subchannel_summary.csv per_subchannel_layout_summary.csv; do [ -f $f/$c ] || { echo "STOP missing $f/$c"; exit 1; }; done
done
# ---- links shared by both versions (PixCell hard-coded paths)
mkdir -p $R/data $R/inference_output/concat_ablation_1000 $R/inference_output/CRC34_metrics $R/results/a1_probe_mlp_spatial $R/results/a6_cohort_invariance
ln -sfn $I/eval/orion-crc33sec $R/data/orion-crc33
ln -sfn $I/eval/orion-crc34eval $R/data/orion-CRC34
for m in paired unpaired; do ln -sfn $I/score/abl_crc33sec/${m}_ablation $R/inference_output/concat_ablation_1000/${m}_ablation; done
mkdir -p $I/score/abl_crc33sec/unpaired_ablation/metadata
cp $I/eval/crc33sec_p300_unpaired.json $I/score/abl_crc33sec/unpaired_ablation/metadata/unpaired_mapping.json
ln -sfn $I/score/crc34_crc34eval $R/inference_output/CRC34_paired
ln -sfn $I/scoreres/crc34_crc34eval/unpaired_cache $R/inference_output/CRC34_metrics/unpaired_cache
ln -sfn $I/old/loo_crc10 $R/inference_output/subchannel_loo_n300_crc10
: > $O/r2_sources.txt
for n in uni_16 virchow2_16 ctranspath_07 resnet50_07 t2_spatial uni_16_crc34 uni_16_crc10; do
  src=$I/old/results/a1_probe_mlp_spatial/$n; lab=old
  if [ "$A3_R2" != old ] && [ -f $A3_R2/$n/mlp_spatial_probe_results.csv ]; then src=$A3_R2/$n; lab=new; fi
  [ -f $src/mlp_spatial_probe_results.csv ] || { echo "STOP no R2 csv for $n"; exit 1; }
  ln -sfn $src $R/results/a1_probe_mlp_spatial/$n; echo "$n $lab $src" | tee -a $O/r2_sources.txt
done
$PY $H/scripts/figrender.py guard $R || exit 1
# ---- figure steps
fails=0
step () { local name=$1; shift; echo "### $(date -Is) step $name: $*"
  (cd $R && $PY -u $H/scripts/figrender.py run $R "$@") > $O/step_$name.log 2>&1; local rc=$?
  echo "### $(date -Is) step $name rc $rc"; [ $rc -eq 0 ] || { fails=$((fails+1)); echo "$name rc $rc" >> $O/A3_FAILED; tail -20 $O/step_$name.log; }; return $rc; }
P=$F/share-space-paper
for ref in pri sec; do
  if [ $ref = pri ]; then T=$I/realref; else T=$I/scoreres; fi
  ln -sfn $T/loo_crc33sec $R/inference_output/subchannel_loo_n300
  ln -sfn $T/loo_crc34eval $R/inference_output/subchannel_loo_n300_crc34
  stamp=$O/.stamp_$ref; touch $stamp; sleep 1
  step a6_$ref src.a6_cohort_invariance && cp $P/figures/data/cohort_invariance_metrics.json $R/results/a6_cohort_invariance/
  step fig4_$ref src.paper_figures.fig_combined_per_channel
  A3_LAYOUT_LAX=1 step thr_$ref src.paper_figures.fig_si_threshold_sensitivity   # label-overlap guard stopped r2old (43443941)
  if [ $ref = pri ]; then A=$I/realref/abl_crc33sec; Z=$I/realref/crc34_crc34eval; else A=$I/scoreres/abl_crc33sec; Z=$I/scoreres/crc34_crc34eval; fi
  mkdir -p $P/figures/data
  if $PY $H/scripts/figrender.py patient $A/paired_ablation/ablation_results $A/unpaired_ablation/ablation_results \
      $Z/paired_cache $Z/unpaired_cache $P/figures/data/patient_generalization_metrics.json 299 299 1000 1000 > $O/patient_json_$ref.log 2>&1; then
    step patient_$ref src.paper_figures.fig_si_patient_generalization
  else
    echo "patient_json_$ref failed"; tail -5 $O/patient_json_$ref.log; fails=$((fails+1)); echo "patient_json_$ref" >> $O/A3_FAILED
  fi
  mkdir -p $O/$ref
  find $R/results/paper_figures $R/results/a6_cohort_invariance $R/figures/pngs_updated $P/figures -newer $stamp -type f \
       \( -name "*.png" -o -name "*.pdf" -o -name "*.json" \) 2>/dev/null | sort > $O/$ref/written.txt
  while read -r f; do d=$O/$ref/$(dirname ${f#$F/}); mkdir -p $d; cp -p "$f" $d/; done < $O/$ref/written.txt
done
cp $I/realref/loo_crc34eval/per_subchannel_*summary.csv $O/ 2>/dev/null; mkdir -p $O/c34_pri_loo && mv $O/per_subchannel_*summary.csv $O/c34_pri_loo/ 2>/dev/null
aws s3 sync --only-show-errors $O $S/figures_rerender/a3/$A3_TAG && echo "upload ok" || { echo "upload FAILED"; fails=$((fails+1)); }
tar -cf $H/out_$A3_TAG.tar -C $F/out $A3_TAG
echo "### $(date -Is) done, $fails failed steps"
exit $([ $fails -eq 0 ] && echo 0 || echo 1)
