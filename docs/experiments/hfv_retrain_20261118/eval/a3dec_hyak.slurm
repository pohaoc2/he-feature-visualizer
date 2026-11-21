#!/bin/bash
#SBATCH --job-name=a3-dec
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=40
#SBATCH --mem=170G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_a3_20261120/logs/%x-%j.out
# hfv_retrain_20261118 追加 3, steps B and C in one job on node-local /tmp (README 追加 3 decisions 1-4, written before any
# result). On 2026-11-20/21 file creation under /gscratch/cheme from compute nodes took ~0.25-1.4 s per file (login ~1 ms)
# and stalled 43440941 / 43441635; /gscratch/scrubbed was 98 % full (43441378). So every file is made in /tmp/a3d_<job>;
# results go to S3 figures_rerender/a3/dec/ and one tar per kind to H (features, results).
#   B  inputs: C33 old channels + H&E + CLS from S3 share_space/02_datasets/orion-crc33.zip (paper tile sets only); C33 new
#      cell channels from S3 inputs/new_exp_channels_crc33_01.tar (vasculature/oxygen/glucose = the old files, as in
#      training); C34 from Hyak hfv_crc34_20261118/data/*.tar (read only), paper 10,000 tiles that have new channels;
#      paper tokens (250 tiles) from S3 05_results/bundles_202607/decodability_cpu_bundle.tgz.
#      a3tok.py: decision 4(b) token check (picks the UNI transform) + CPU s/image of the four encoders.
#   placement rule (written before timing): extraction on this job's CPUs if the extrapolated extraction time
#      (s/image x images, per encoder) is at most half of the walltime left (the other half for targets and probes, whose
#      time is unknown beforehand); otherwise STOP, and a Colab L4 kit is prepared instead.
#   C  features (a3feat.py, PixCell cb62fa7): UNI C33 800, Virchow2 C33 200, CTransPath C33 800, ResNet-50 C33 200, UNI C34
#      9,958. Targets (d1fc8a5 src.a1_mask_targets_spatial) and probes (d1fc8a5 src.a1_probe_mlp_spatial, sklearn MLP):
#      C33 old targets on the full paper set (4c) and old/new on the kept set (4d), UNI and CTransPath with max_train_rows
#      500,000 and 50,000 (decision 3), Virchow2 / ResNet-50 with the handover's reduced budget; C34 new targets, 9,958
#      tiles, paper folds minus dropped tiles, shuffle baseline, 500,000 rows. a3summ.py: 4(a), 4(c), the setting choice,
#      4(d), and the folder of selected new R2 CSVs for the render (S3 figures_rerender/a3/dec/r2new/).
# STOP on failure; nothing is retried.
set -uo pipefail
H=/gscratch/cheme/chiu/hfv_a3_20261120; F=/tmp/a3d_$SLURM_JOB_ID; I=$F/dec; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
S=s3://bagherilab-working/pohao/hfv_retrain_20261118; OLD=s3://bagherilab-working/pohao/share_space; SD=$S/figures_rerender/a3/dec
D34=/gscratch/cheme/chiu/hfv_crc34_20261118/data
UNI=/gscratch/cheme/chiu/hfv_retrain_20261118/PixCell/pretrained_models/uni-2h
VIR=/gscratch/cheme/chiu/sev-channel-predicted-20261005/virchow_weights/model.safetensors
CTP=/gscratch/cheme/chiu/msi_perturb_20261119/weights/ctranspath/ctranspath.pth
export TORCH_HOME=$H/torch_home XDG_CACHE_HOME=$F/cache HF_HUB_OFFLINE=1
PY=$C/env/bin/python
module load coenv/aws/2.17.32
T0=$(date +%s); LIMIT=$((24*3600))
mkdir -p $H/.iot_$SLURM_JOB_ID && t0=$(date +%s%N) && for i in $(seq 20); do : > $H/.iot_$SLURM_JOB_ID/f$i; done && echo "### create 20 files in H: $(( ($(date +%s%N)-t0)/1000000 )) ms"; rm -rf $H/.iot_$SLURM_JOB_ID
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"; df -h /tmp | tail -1
[ -e $F ] && { echo "STOP $F exists"; exit 1; }
for t in dec_results dec_features; do [ -e $H/$t.tar ] && { echo "STOP $H/$t.tar exists"; exit 1; }; done
[ -n "$(aws s3 ls $SD/ 2>/dev/null)" ] && { echo "STOP $SD exists"; exit 1; }
mkdir -p $I/raw $I/c33old $I/c33new $I/c34 $I/paper $I/logs $I/targets $I/probe $I/cv
trap 'cd /; tar -cf $H/dec_results_partial_$SLURM_JOB_ID.tar -C $I --exclude="*.npy" --exclude=./raw --exclude=./c33old --exclude=./c33new --exclude=./c34 --exclude=./feat --exclude=./img --exclude=./fsub --exclude=./paper . 2>/dev/null; rm -rf $F' EXIT
# ---------------- B: inputs
mkdir -p $F/pix_cb62fa7 $F/pix_d1fc8a5
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $F/pix_cb62fa7 || { echo "STOP pixcell tar"; exit 1; }
tar -xzf $H/pixcell_d1fc8a5.tar.gz -C $F/pix_d1fc8a5 || { echo "STOP d1fc8a5 tar"; exit 1; }
cd $I
sort -u $H/ids/bundle250.txt $H/ids/paper_800_ids.txt $H/ids/paper_200_ids.txt > ids_he33.txt
sort -u $H/ids/paper_800_ids.txt $H/ids/paper_200_ids.txt > ids_ch33.txt
comm -12 ids_ch33.txt <(sort $H/ids/new33_ids.txt) > ids_ch33_new.txt
comm -12 <(sort $H/ids/paper_c34_10000_ids.txt) <(sort $H/ids/tiles_keep34.txt) > ids_c34_new.txt
comm -12 <(sort $H/ids/paper_800_ids.txt) <(sort $H/ids/new33_ids.txt) > keep800.txt
comm -12 <(sort $H/ids/paper_200_ids.txt) <(sort $H/ids/new33_ids.txt) > keep200.txt
wc -l ids_*.txt keep*.txt
aws s3 cp --only-show-errors $OLD/02_datasets/orion-crc33.zip raw/orion-crc33.zip || { echo "STOP zip"; exit 1; }
$PY $H/scripts/a3prep.py zip raw/orion-crc33.zip ids_he33.txt ids_ch33.txt c33old || exit 1
rm -f raw/orion-crc33.zip
aws s3 cp --only-show-errors $S/inputs/new_exp_channels_crc33_01.tar raw/new33.tar || { echo "STOP new33"; exit 1; }
$PY $H/scripts/a3prep.py tar raw/new33.tar ids_ch33_new.txt c33new 1 || exit 1
rm -f raw/new33.tar
for ch in vasculature oxygen glucose; do mkdir -p c33new/exp_channels/$ch
  while read -r t; do cp c33old/orion-crc33-old/exp_channels/$ch/$t.* c33new/exp_channels/$ch/ || { echo "STOP cp $ch $t"; exit 1; }; done < ids_ch33_new.txt; done
aws s3 cp --only-show-errors $OLD/05_results/bundles_202607/decodability_cpu_bundle.tgz - | tar -xz -C paper || { echo "STOP bundle"; exit 1; }
for n in uni_16 virchow2_16 ctranspath_07 resnet50_07 uni_16_crc34; do
  aws s3 cp --only-show-errors $OLD/05_results/results/a1_probe_mlp_spatial/$n/cv_splits.json paper/cv_$n.json || { echo "STOP cv $n"; exit 1; }
  mkdir -p paper/results/$n; aws s3 cp --only-show-errors $OLD/05_results/results/a1_probe_mlp_spatial/$n/mlp_spatial_probe_results.csv paper/results/$n/ || exit 1
done
for f in mask_targets_T1_spatial.npy tile_ids.txt target_names_T1_spatial.json manifest.json; do
  aws s3 cp --only-show-errors $OLD/05_results/results/a1_mask_targets_spatial/$f paper/targets16_$f || { echo "STOP targets16 $f"; exit 1; }; done
$PY $H/scripts/a3prep.py tar $D34/crc34_he_tiles.tar ids_c34_new.txt c34 0 || exit 1
$PY $H/scripts/a3prep.py tar $D34/crc34_exp_channels.tar ids_c34_new.txt c34 1 || exit 1
$PY $H/scripts/a3prep.py tar $D34/crc34_mx_channels.tar ids_c34_new.txt c34 1 || exit 1
for d in c33new/exp_channels/* c34/exp_channels/* c34/he; do echo "$d $(ls $d | wc -l)"; done
BUNDLE=$(dirname $(find paper -type d -name uni_tokens | head -1))
echo "### $(date -Is) B: token check + timing (bundle $BUNDLE)"
PYTHONPATH=$F/pix_cb62fa7:$H/pylib OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $PY -u $H/scripts/a3tok.py $BUNDLE c33old/orion-crc33-old/he $UNI $VIR $CTP $I/a3tok.json \
  || { echo "STOP a3tok"; exit 1; }
aws s3 cp --only-show-errors $I/a3tok.json $SD/a3tok.json
# ---------------- placement rule
LEFT=$(( LIMIT - ($(date +%s) - T0) ))
$PY - $I/a3tok.json $LEFT <<'EOF' || { echo "STOP placement rule: extraction does not fit this job -> Colab L4 kit"; exit 1; }
import json, sys
r = json.load(open(sys.argv[1])); left = int(sys.argv[2]); t = r["timing_s_per_image"]
uni = t["uni_tokens_" + r["uni_transform_used"]]
est = uni * (800 + 9958) + t["virchow2"] * 200 + t["ctranspath"] * 800 + t["resnet50"] * 200
print(f"extrapolated extraction {est/3600:.2f} h; walltime left {left/3600:.2f} h; rule: extraction <= half of left")
sys.exit(0 if est <= left / 2 else 1)
EOF
# ---------------- C: features
links () { local list=$1 src=$2 dst=$3 suf=$4; [ -e $dst ] && { echo "STOP $dst exists"; exit 1; }; mkdir -p $dst
  while read -r t; do [ -e $src/$t$suf ] || { echo "STOP missing $src/$t$suf"; exit 1; }; ln -s $src/$t$suf $dst/$t$suf; done < $list; }
links $H/ids/paper_800_ids.txt $I/c33old/orion-crc33-old/he $I/img/he33_800 .png
links $H/ids/paper_200_ids.txt $I/c33old/orion-crc33-old/he $I/img/he33_200 .png
links $I/ids_c34_new.txt $I/c34/he $I/img/he34 .png
feat () { echo "### $(date -Is) feat $*"; PYTHONPATH=$F/pix_cb62fa7:$H/pylib OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $PY -u $H/scripts/a3feat.py "$@" \
  > $I/logs/feat_$(basename $3).log 2>&1 || { echo "STOP feat $1 $3"; tail -20 $I/logs/feat_$(basename $3).log; exit 1; }; tail -1 $I/logs/feat_$(basename $3).log; }
feat uni $I/img/he33_800 $I/feat/uni33 $UNI - $I/a3tok.json
feat virchow2 $I/img/he33_200 $I/feat/vir33 $UNI $VIR
feat ctranspath $I/img/he33_800 $I/feat/ctp33 $UNI $CTP
feat resnet50 $I/img/he33_200 $I/feat/res33 $UNI -
feat uni $I/img/he34 $I/feat/uni34 $UNI - $I/a3tok.json
echo "### $(date -Is) features done"
# ---------------- C: targets
export PYTHONPATH=$F/pix_d1fc8a5:$H/pylib
cd $F/pix_d1fc8a5 || exit 1
TGT () { local name=$1 fd=$2 ex=$3 grid=$4 suf=$5; echo "### $(date -Is) targets $name"
  OMP_NUM_THREADS=4 $PY -m src.a1_mask_targets_spatial.main --features-dir $fd --exp-channels-dir $ex --out-dir $I/targets/$name \
    --grid $grid --feature-suffix $suf > $I/logs/targets_$name.log 2>&1 || { echo "STOP targets $name"; tail -20 $I/logs/targets_$name.log; exit 1; }; }
O33=$I/c33old/orion-crc33-old/exp_channels; N33=$I/c33new/exp_channels
links $I/keep800.txt $I/feat/uni33 $I/fsub/uni715 _uni_tokens.npy
links $I/keep800.txt $I/feat/ctp33 $I/fsub/ctp715 _patches.npy
links $I/keep200.txt $I/feat/vir33 $I/fsub/vir184 _patches.npy
links $I/keep200.txt $I/feat/res33 $I/fsub/res184 _patches.npy
for p in "uni_16 paper_800_ids keep800" "ctranspath_07 paper_800_ids keep800" "virchow2_16 paper_200_ids keep200" "resnet50_07 paper_200_ids keep200"; do
  set -- $p; $PY $H/scripts/a3prep.py splits $I/paper/cv_$1.json $H/ids/$2.txt $I/$3.txt $I/cv/ids_$1_kept.txt $I/cv/cv_$1_kept.json || exit 1
done
$PY $H/scripts/a3prep.py splits $I/paper/cv_uni_16_crc34.json $H/ids/paper_c34_10000_ids.txt $I/ids_c34_new.txt \
    $I/cv/ids_uni_16_crc34_kept.txt $I/cv/cv_uni_16_crc34_kept.json || exit 1
TGT old16_800 $I/feat/uni33 $O33 16 _uni_tokens.npy
TGT old16_715 $I/fsub/uni715 $O33 16 _uni_tokens.npy;  TGT new16_715 $I/fsub/uni715 $N33 16 _uni_tokens.npy
TGT old16_200 $I/feat/vir33 $O33 16 _patches.npy;      TGT old16_184 $I/fsub/vir184 $O33 16 _patches.npy
TGT new16_184 $I/fsub/vir184 $N33 16 _patches.npy
TGT old7_800 $I/feat/ctp33 $O33 7 _patches.npy;        TGT old7_715 $I/fsub/ctp715 $O33 7 _patches.npy
TGT new7_715 $I/fsub/ctp715 $N33 7 _patches.npy
TGT old7_200 $I/feat/res33 $O33 7 _patches.npy;        TGT old7_184 $I/fsub/res184 $O33 7 _patches.npy
TGT new7_184 $I/fsub/res184 $N33 7 _patches.npy
TGT new16_c34 $I/feat/uni34 $I/c34/exp_channels 16 _uni_tokens.npy
# the target builders list tiles from the feature folder; their order must hash to the paper (full sets) / kept folds
hashchk () { local cv=$1 t=$2
  [ "$($PY -c "import json;print(json.load(open('$cv'))['tile_ids_sha1'])")" = \
    "$($PY -c "from src._tasklib.tile_ids import tile_ids_sha1;print(tile_ids_sha1([l.strip() for l in open('$I/targets/$t/tile_ids.txt') if l.strip()]))")" ] \
    || { echo "STOP $t tile ids do not hash to $cv"; exit 1; }; }
hashchk $I/paper/cv_uni_16.json old16_800;        hashchk $I/cv/cv_uni_16_kept.json old16_715;  hashchk $I/cv/cv_uni_16_kept.json new16_715
hashchk $I/paper/cv_ctranspath_07.json old7_800;  hashchk $I/cv/cv_ctranspath_07_kept.json old7_715; hashchk $I/cv/cv_ctranspath_07_kept.json new7_715
hashchk $I/paper/cv_virchow2_16.json old16_200;   hashchk $I/cv/cv_virchow2_16_kept.json old16_184; hashchk $I/cv/cv_virchow2_16_kept.json new16_184
hashchk $I/paper/cv_resnet50_07.json old7_200;    hashchk $I/cv/cv_resnet50_07_kept.json old7_184;  hashchk $I/cv/cv_resnet50_07_kept.json new7_184
hashchk $I/cv/cv_uni_16_crc34_kept.json new16_c34
echo "### $(date -Is) targets done, hashes match"
# ---------------- C: probes (C34 in the background, 5 processes x 4 threads; C33 two at a time, 2 processes x 4 threads;
#                  cheme nodes have 40 cores)
PRB () { local name=$1 fd=$2 tg=$3 cv=$4 suf=$5; shift 5; [ -e $I/probe/$name ] && { echo "STOP probe/$name exists"; return 1; }
  echo "### $(date -Is) probe $name start"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY -m src.a1_probe_mlp_spatial.main \
    --features-dir $fd --targets-path $I/targets/$tg/mask_targets_T1_spatial.npy --tile-ids-path $I/targets/$tg/tile_ids.txt \
    --target-names-path $I/targets/$tg/target_names_T1_spatial.json --cv-splits-path $cv --out-dir $I/probe/$name \
    --feature-suffix $suf --batch-size 2048 "$@" > $I/logs/probe_$name.log 2>&1; local rc=$?
  rm -f $I/probe/$name/_X_memmap.npy; echo "### $(date -Is) probe $name rc $rc"; [ $rc -eq 0 ] || echo "probe_$name rc $rc" >> $I/A3DEC_FAILED; return $rc; }
export -f PRB; export PY I
PRB c34_new_r500000 $I/feat/uni34 new16_c34 $I/cv/cv_uni_16_crc34_kept.json _uni_tokens.npy \
    --compute-shuffle-baseline --max-train-rows 500000 --n-jobs 5 &
C34PID=$!
RED="--max-train-rows 10000 --max-iter 50 --hidden-layer-sizes 128,32 --n-jobs 2"
{
  for r in 500000 50000; do
    echo "uni_old800_r$r $I/feat/uni33 old16_800 $I/paper/cv_uni_16.json _uni_tokens.npy --max-train-rows $r --n-jobs 2"
    echo "uni_old715_r$r $I/fsub/uni715 old16_715 $I/cv/cv_uni_16_kept.json _uni_tokens.npy --max-train-rows $r --n-jobs 2"
    echo "uni_new715_r$r $I/fsub/uni715 new16_715 $I/cv/cv_uni_16_kept.json _uni_tokens.npy --max-train-rows $r --n-jobs 2"
    echo "ctp_old800_r$r $I/feat/ctp33 old7_800 $I/paper/cv_ctranspath_07.json _patches.npy --max-train-rows $r --n-jobs 2"
    echo "ctp_old715_r$r $I/fsub/ctp715 old7_715 $I/cv/cv_ctranspath_07_kept.json _patches.npy --max-train-rows $r --n-jobs 2"
    echo "ctp_new715_r$r $I/fsub/ctp715 new7_715 $I/cv/cv_ctranspath_07_kept.json _patches.npy --max-train-rows $r --n-jobs 2"
  done
  echo "vir_old200 $I/feat/vir33 old16_200 $I/paper/cv_virchow2_16.json _patches.npy $RED"
  echo "vir_old184 $I/fsub/vir184 old16_184 $I/cv/cv_virchow2_16_kept.json _patches.npy $RED"
  echo "vir_new184 $I/fsub/vir184 new16_184 $I/cv/cv_virchow2_16_kept.json _patches.npy $RED"
  echo "res_old200 $I/feat/res33 old7_200 $I/paper/cv_resnet50_07.json _patches.npy $RED"
  echo "res_old184 $I/fsub/res184 old7_184 $I/cv/cv_resnet50_07_kept.json _patches.npy $RED"
  echo "res_new184 $I/fsub/res184 new7_184 $I/cv/cv_resnet50_07_kept.json _patches.npy $RED"
} > $I/probes_c33.txt
xargs -P 2 -L 1 bash -c 'PRB "$@"' _ < $I/probes_c33.txt
wait $C34PID
[ -f $I/A3DEC_FAILED ] && { echo "failed probes:"; cat $I/A3DEC_FAILED; }
# ---------------- summary (4a, 4c, choice, 4d) and the selected new R2 folder
$PY $H/scripts/a3summ.py $I $I/paper/results $I/a3summ.json $I/r2new > $I/logs/a3summ.log 2>&1 || echo "a3summ failed" >> $I/A3DEC_FAILED
cat $I/logs/a3summ.log
# ---------------- outputs
cd $I
aws s3 cp --only-show-errors a3summ.json $SD/a3summ.json
aws s3 sync --only-show-errors r2new $SD/r2new
aws s3 sync --only-show-errors probe $SD/probe --exclude "*_X_memmap.npy"
aws s3 sync --only-show-errors targets $SD/targets
aws s3 sync --only-show-errors cv $SD/cv
aws s3 sync --only-show-errors logs $SD/logs
tar -cf $H/dec_results.tar a3tok.json a3summ.json r2new probe targets cv logs ids_*.txt keep*.txt probes_c33.txt $( [ -f A3DEC_FAILED ] && echo A3DEC_FAILED ) || echo "results tar failed"
tar -cf $H/dec_features.tar feat || echo "features tar failed"
trap 'rm -rf $F' EXIT
echo "### $(date -Is) done"
[ -f $I/A3DEC_FAILED ] && exit 1 || exit 0
