#!/bin/bash
# usage: pp_pull.sh ROOT  -> copies gen_ROOT from S3 to D/gen/ROOT.tar, checks 100 png per variant, submits pp_cv array (official+runner)
set -euo pipefail
ROOT=$1; D=/gscratch/cheme/chiu/paperpq_official_20261122
module load coenv/aws/2.17.32 >/dev/null 2>&1
mkdir -p $D/gen
test ! -e $D/gen/$ROOT.tar || { echo "exists $D/gen/$ROOT.tar"; exit 1; }
aws s3 cp --only-show-errors s3://bagherilab-working/pohao/paperpq_official_20261122/outputs/gen_$ROOT $D/gen/$ROOT.tar
case $ROOT in P_*) VS="a1_concat production a2_bypass_full_tme a1_per_channel a2_off_shelf";; *) VS="a1_concat production a2_bypass_full_tme a1_per_channel";; esac
tar -tf $D/gen/$ROOT.tar > /tmp/pp_list_$ROOT.txt
IT=""
for v in $VS; do n=$(grep -c "^out/tiles/$v/.*\.png$" /tmp/pp_list_$ROOT.txt || true); echo "$ROOT $v $n"
  [ "$n" = 100 ] || { echo "STOP $v has $n"; exit 1; }
  IT="$IT,$ROOT:$v:official,$ROOT:$v:runner"; done
grep -E "out/(DONE|fud.json|appear_per_tile.csv)" /tmp/pp_list_$ROOT.txt
IT=${IT#,}; N=$(echo $IT | tr , '\n' | wc -l)
sbatch --parsable --array=0-$((N-1)) --job-name=pp_cv_$ROOT $D/scripts/pp_cv.slurm $IT
