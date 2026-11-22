#!/bin/bash
# hfv_retrain2 stage 3 (PREREG §9), login node, after both r2 Colab jobs returned. One shot, any failure STOP.
#   1) G/colab/.get.txt ("<jobs_out name> <presigned GET>" x 3, mode 600, r2_presign.py get; deleted here) -> curl into
#      G/colab; only r2ND/ and fig2c_<set>/ paths accepted; extract into G/gen; status / DONE checks
#   2) r2score.py zip G N (cheme) -> CellViT array N_crc36ho, N_crc33sec, ND (cheme CPU, s3cv.slurm)
#   3) metrics (pxscore.py r2score.py metrics, cheme CPU) -> stats
#   bash s3post.sh
set -uo pipefail
CC=/gscratch/cheme/chiu; G=$CC/hfv_retrain2_20261121/stage3; PY=$CC/crc16-20261024/cellvit/env/bin/python
stop() { echo "$(date -Is) STOP: $*" | tee -a $G/STAGE3.txt; rm -f $G/colab/.get.txt; exit 1; }
sb() { local o; o=$(sbatch --parsable "$@" 2>&1); [[ $o =~ ^[0-9]+$ ]] || { echo "sbatch: $o" >&2; return 1; }; echo $o; }
test -s $G/colab/.get.txt || stop "no .get.txt"
test -e $G/gen && stop "$G/gen exists"
mkdir -p $G/gen
while read -r name url; do
  [[ $name =~ ^jobs_out_(r2ND|fig2c_crc36ho|fig2c_crc33sec)\.tar$ ]] || stop "name $name"
  o=${BASH_REMATCH[1]}
  curl -sf -o $G/colab/$name "$url" || stop "download $name"
  bad=$(tar -tf $G/colab/$name | grep -v -E "^$o(/|\$)" | head -3)
  [ -z "$bad" ] || stop "unexpected paths in $name: $bad"
  tar -xf $G/colab/$name -C $G/gen || stop "extract $name"
  echo "$(date -Is) fetched $name $(sha256sum $G/colab/$name | cut -c1-16)" | tee -a $G/STAGE3.txt
done < $G/colab/.get.txt
rm -f $G/colab/.get.txt
grep -q '"state": "complete"' $G/gen/r2ND/status.json && grep -q '"images": 384' $G/gen/r2ND/status.json || stop "r2ND not complete"
for s in crc36ho crc33sec; do
  test -f $G/gen/fig2c_$s/DONE || stop "fig2c_$s not DONE"
  n=$(ls $G/gen/fig2c_$s/tiles/a1_concat/ | grep -c -E '^[0-9]+_[0-9]+\.png$'); e=$(wc -l < $G/inputs/lists/$([ $s = crc36ho ] && echo crc36ho_p300_ids.txt || echo crc33sec_fig2_ids.txt))
  [ "$n" = "$e" ] || stop "fig2c_$s has $n images, expected $e"
done
free=$(hyakalloc 2>/dev/null | awk -F'│' '/FREE/ && !d {gsub(/ /,"",$4); print $4; d=1}')
[[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge $((120 + 24 + 8)) ] || stop "cheme free ${free:-?} < 120 + 32"
CPU=(--account=cheme --partition=compute --kill-on-invalid-dep=yes)
z=$(sb -J r2zipN "${CPU[@]}" -c 2 --mem=8G --time=00:30:00 --output=$G/logs/%x-%j.out \
    --wrap "$PY -I $G/scripts/r2score.py zip $G N") || stop "sbatch zip"
cv=$(sb -J r2cvN --array=0-2 --dependency=afterok:$z --kill-on-invalid-dep=yes --output=$G/logs/%x_%A_%a.out \
     $G/scripts/s3cv.slurm $G N_crc36ho,N_crc33sec,ND) || stop "sbatch cv"
mt=$(sb -J r2metrics "${CPU[@]}" -c 8 --mem=32G --time=03:00:00 --dependency=afterok:$cv --output=$G/logs/%x-%j.out \
     $G/scripts/s3metrics.slurm) || stop "sbatch metrics"
echo "$(date -Is) submitted zip $z cv $cv metrics $mt (cheme free $free)" | tee -a $G/STAGE3.txt
