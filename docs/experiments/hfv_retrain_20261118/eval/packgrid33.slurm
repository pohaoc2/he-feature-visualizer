#!/bin/bash
#SBATCH --job-name=ev-packgrid33
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_eval_20261118/logs/%x-%j.out
# hfv_retrain_20261118 figure re-render (README, coordinator 2026-11-19): the unpaired ablation-grid tile of
# ablation_grids_combined panel b, 13056_27392 with its paper style source (paper unpaired_mapping.json; 12032_15104),
# as a one-tile pack orion-crc33grid. It is the 1,001st tile of the paper's unpaired list, so it is not in
# evalpack_crc33sec. Same sources and code as pack33.slurm. Generated with
#   abl:orion-crc33grid:crc33grid_ids.txt:crc33grid_unpaired.json
# in the same job / session (same GPU type) as the crc33sec ablation; not part of any metric set.
# If the grid tile has no new channels in CRC33_01 (ink exclusion), writes NO_CHANNELS and exits 0: panel b then keeps
# the old model input. Work in node-local /tmp (many small members). set -e, no retry.
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/packgrid33.slurm
set -euo pipefail
E=/gscratch/cheme/chiu/hfv_eval_20261118; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
S=s3://bagherilab-working/pohao/hfv_retrain_20261118
module load coenv/aws/2.17.32
W=/tmp/packgrid33_$SLURM_JOB_ID; mkdir -p $W; trap 'rm -rf $W' EXIT; cd $W
echo "== start $(date -Is) $(hostname)"
aws s3 cp --only-show-errors $S/inputs/eval/paper_lists/unpaired_mapping.json paper_unpaired.json
G=13056_27392
STY=$($PY -c "import json;print(json.load(open('paper_unpaired.json'))['style_mapping']['$G'])")
echo "grid $G style $STY"
aws s3 cp --only-show-errors $S/inputs/new_exp_channels_crc33_01.tar newch33.tar
aws s3 cp --only-show-errors $S/outputs/masklat_crc33_01 masklat33.tar
md5sum newch33.tar masklat33.tar paper_unpaired.json
if ! tar -tf newch33.tar | grep -q "/cell_masks/$G\.png$"; then
  echo "NO_CHANNELS: $G has no new cell channels in CRC33_01; panel b keeps the old model input"
  echo "{\"tile\": \"$G\", \"status\": \"NO_CHANNELS\"}" > $E/packs/crc33grid_NO_CHANNELS.json
  aws s3 cp --only-show-errors $E/packs/crc33grid_NO_CHANNELS.json $S/inputs/eval/crc33grid_NO_CHANNELS.json; exit 0
fi
aws s3 cp --only-show-errors s3://bagherilab-working/pohao/share_space/02_datasets/orion-crc33.zip orion-crc33.zip
md5sum orion-crc33.zip
echo $G > crc33grid_ids.txt; echo $STY > crc33grid_style_ids.txt
$PY - "$G" "$STY" <<'PYZ'
import json, re, sys, tarfile, zipfile
G, STY = sys.argv[1], sys.argv[2]
T = {G, STY}
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
json.dump({"seed": 42, "source": "paper unpaired_mapping.json (concat_ablation_1000), entry for the grid tile",
           "tile_ids": [G], "style_mapping": {G: STY}}, open("crc33grid_unpaired.json", "w"), indent=1)
PYZ
rm orion-crc33.zip
$PY $E/scripts/evalpack.py orion-crc33grid crc33grid_ids.txt evalpack_crc33grid.tar newch33.tar \
  z33.tar z33.tar z33.tar masklat33.tar z33.tar crc33grid_style_ids.txt
md5sum evalpack_crc33grid.tar crc33grid_* | tee evalpack_crc33grid.md5
mkdir -p $E/packs
for f in evalpack_crc33grid.tar evalpack_crc33grid.tar.json evalpack_crc33grid.md5 crc33grid_ids.txt crc33grid_style_ids.txt crc33grid_unpaired.json; do
  cp $f $E/packs/; aws s3 cp --only-show-errors $f $S/inputs/eval/$f; done
aws s3 ls $S/inputs/eval/ | grep crc33grid
echo "== done $(date -Is)"
