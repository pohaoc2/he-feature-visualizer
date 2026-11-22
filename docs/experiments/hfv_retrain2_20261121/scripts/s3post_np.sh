#!/bin/bash
# hfv_retrain2 stage 3, arm NP as its own job (2026-11-22: the ch6 rule stopped app-10 generation, so the app-10 KP arm that
# was to serve as NP is not generated). Login node, after colab session r2C returned. One shot, any failure STOP.
#   G/colab/.get_np.txt = "jobs_out_r2NP.tar <presigned GET>" (mode 600, r2_presign.py get OUT jobs_out_r2NP.tar; deleted here)
#   -> G/gen/r2NP -> r2score.py zip G NP -> CellViT G/cv/NP (s3cv.slurm) -> readout 3 (r2_app8.py, CV10 = G/cv)
#   bash s3post_np.sh SEV APP8_JSON OUT_JSON
set -uo pipefail
CC=/gscratch/cheme/chiu; G=$CC/hfv_retrain2_20261121/stage3; PY=$CC/crc16-20261024/cellvit/env/bin/python
SEV=${1:?SEV}; A8J=${2:?APP8_JSON}; OUT=${3:?OUT_JSON}
stop() { echo "$(date -Is) STOP: $*" | tee -a $G/STAGE3.txt; rm -f $G/colab/.get_np.txt; exit 1; }
sb() { local o; o=$(sbatch --parsable "$@" 2>&1); [[ $o =~ ^[0-9]+$ ]] || { echo "sbatch: $o" >&2; return 1; }; echo $o; }
test -s $G/colab/.get_np.txt || stop "no .get_np.txt"
test -e $G/gen/r2NP && stop "$G/gen/r2NP exists"
test -e $OUT && stop "$OUT exists"
read -r name url < $G/colab/.get_np.txt
[ "$name" = jobs_out_r2NP.tar ] || stop "name $name"
curl -sf -o $G/colab/$name "$url" || stop "download $name"
rm -f $G/colab/.get_np.txt
bad=$(tar -tf $G/colab/$name | grep -v -E "^r2NP(/|\$)" | head -3)
[ -z "$bad" ] || stop "unexpected paths in $name: $bad"
tar -xf $G/colab/$name -C $G/gen || stop "extract $name"
echo "$(date -Is) fetched $name $(sha256sum $G/colab/$name | cut -c1-16)" | tee -a $G/STAGE3.txt
CPU=(--account=cheme --partition=compute --kill-on-invalid-dep=yes)
z=$(sb -J r2zipNP "${CPU[@]}" -c 2 --mem=8G --time=00:30:00 --output=$G/logs/%x-%j.out \
    --wrap "$PY -I $G/scripts/r2score.py zip $G NP") || stop "sbatch zip"
cv=$(sb -J r2cvNP --array=0 --dependency=afterok:$z --kill-on-invalid-dep=yes --output=$G/logs/%x_%A_%a.out \
     $G/scripts/s3cv.slurm $G NP) || stop "sbatch cv"
r8=$(sb -J r2app8 "${CPU[@]}" -c 4 --mem=24G --time=00:40:00 --dependency=afterok:$cv --output=$G/logs/%x-%j.out \
     --wrap "OMP_NUM_THREADS=4 $PY -u $G/scripts/r2_app8.py $SEV $CC/hfv_retrain2_20261121/eval/crc16_L1 \
$CC/msi_perturb_20261119/s3b_code $CC/msi_perturb_20261119/s3b_crc16b/cells $CC/sim2real_apps_20261120/app2/cv \
$CC/sim2real_apps_20261120/app8/cv $G/cv $G/cv/ND $A8J $OUT") || stop "sbatch r2_app8"
echo "$(date -Is) NP: zip $z cv $cv r2_app8 $r8" | tee -a $G/STAGE3.txt
