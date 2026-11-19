#!/bin/bash
#SBATCH --job-name=ev-pack33
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_eval_20261118/logs/%x-%j.out
# hfv_retrain_20261118 evaluation: the CRC33_01 secondary pack (README rules: comparable with the paper; training tiles,
# not held out). Same inputs as the training root: the new cell channels (S3 inputs/new_exp_channels_crc33_01.tar),
# vasculature / oxygen / glucose, stored UNI, SD3.5 VAE H&E latents and H&E from orion-crc33.zip, the new mask latents
# (S3 outputs/masklat_crc33_01). Tile sets: eval/sets33.py from the paper lists (S3 inputs/eval/paper_lists/).
# The zip is copied to this job's folder, read, then deleted. set -e, no retry.
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/pack33.slurm
set -euo pipefail
E=/gscratch/cheme/chiu/hfv_eval_20261118; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
module load coenv/aws/2.17.32
echo "== start $(date -Is) $(hostname)"
mkdir -p $E/src33 $E/packs; cd $E/src33
aws s3 cp --only-show-errors $S/inputs/new_exp_channels_crc33_01.tar newch33.tar
aws s3 cp --only-show-errors $S/outputs/masklat_crc33_01 masklat33.tar
aws s3 cp --only-show-errors s3://bagherilab-working/pohao/share_space/02_datasets/orion-crc33.zip orion-crc33.zip
aws s3 cp --only-show-errors --recursive $S/inputs/eval/paper_lists/ lists/
md5sum newch33.tar masklat33.tar orion-crc33.zip lists/*
tar -tf newch33.tar | sed -n 's#.*/cell_masks/\([0-9]*_[0-9]*\)\.png$#\1#p' | sort > eligible33.txt; wc -l eligible33.txt
$PY $E/scripts/sets33.py eligible33.txt lists/metric_tile_ids.txt lists/paired_1000_tile_ids.txt lists/unpaired_mapping.json $E/packs
# members of the needed tiles from the zip (mx channels, features, vae_features, he) into one tar
$PY - <<'PYZ'
import re, tarfile, zipfile
E = "/gscratch/cheme/chiu/hfv_eval_20261118"
T = {t.strip() for f in ("crc33sec_pack_ids.txt", "crc33sec_style_ids.txt") for t in open(f"{E}/packs/{f}") if t.strip()}
rx = re.compile(r"/(exp_channels/(vasculature|oxygen|glucose)/|features/|vae_features/|he/)(\d+_\d+)(_uni|_sd3_vae)?\.(png|npy)$")
n = 0
with zipfile.ZipFile("orion-crc33.zip") as z, tarfile.open("z33.tar", "w") as out:
    for i in z.infolist():
        m = rx.search("/" + i.filename)
        if m is None or m.group(3) not in T:
            continue
        ti = tarfile.TarInfo(i.filename); ti.size = i.file_size; ti.mode = 0o644
        with z.open(i) as f:
            out.addfile(ti, f)
        n += 1
print("zip members copied", n)
PYZ
rm orion-crc33.zip
cd $E/packs
$PY $E/scripts/evalpack.py orion-crc33sec crc33sec_pack_ids.txt evalpack_crc33sec.tar $E/src33/newch33.tar \
  $E/src33/z33.tar $E/src33/z33.tar $E/src33/z33.tar $E/src33/masklat33.tar $E/src33/z33.tar crc33sec_style_ids.txt
md5sum evalpack_crc33sec.tar crc33sec_* | tee evalpack_crc33sec.md5
for f in evalpack_crc33sec.tar evalpack_crc33sec.tar.json sets33.json crc33sec_*; do aws s3 cp --only-show-errors $f $S/inputs/eval/$f; done
aws s3 ls $S/inputs/eval/
echo "== done $(date -Is)"
