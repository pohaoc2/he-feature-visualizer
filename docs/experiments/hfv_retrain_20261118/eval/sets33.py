"""hfv_retrain_20261118 evaluation: the CRC33_01 secondary tile sets ("comparable with the paper; training tiles, not
held out", README rules). Paper lists from inference_output_20260527.zip (range reads): si_a1_a2/metric_tile_ids.txt
(Fig 2, 100), concat_ablation_1000/tile_lists/paired_1000_tile_ids.txt (ablation + Fig 4: first 300) and
unpaired_ablation/metadata/unpaired_mapping.json (seed 42 derangement). Decomposition: the paper's rule, the first 500
tile ids of the data root in PixCell's numeric order (row, col as integers) (here: the tiles that have the new channels).

  python sets33.py ELIGIBLE_TXT METRIC_TXT PAIRED1000_TXT UNPAIRED_JSON OUT_DIR
ELIGIBLE_TXT = the CRC33_01 tiles with new cell channels (ink tiles excluded). Every set is cut to eligible tiles; the
dropped ids are written to sets33.json. Unpaired style tiles keep the paper mapping even when the style tile is an ink
tile (only its UNI and H&E are used).
"""
import json
import sys

el_txt, met_txt, p1k_txt, unp_json, out = sys.argv[1:6]
rd = lambda p: [t.strip() for t in open(p) if t.strip()]  # noqa: E731
E = set(rd(el_txt))
met, p1k = rd(met_txt), rd(p1k_txt)
m = json.load(open(unp_json))["style_mapping"]
S = {}
S["fig2"] = [t for t in met if t in E]
S["p300"] = [t for t in p1k[:300] if t in E]
S["decomp"] = sorted(E, key=lambda t: tuple(map(int, t.split("_"))))[:500]   # PixCell sort_tile_ids_numeric order
style = {t: m[t] for t in S["p300"]}
R = dict(eligible=len(E), counts={k: len(v) for k, v in S.items()},
         dropped=dict(fig2=[t for t in met if t not in E], p300=[t for t in p1k[:300] if t not in E]),
         style_tiles_without_channels=sorted({s for s in style.values() if s not in E}))
for k, v in S.items():
    open(f"{out}/crc33sec_{k}_ids.txt", "w").write("".join(f"{t}\n" for t in v))
json.dump(dict(seed=42, source="paper unpaired_mapping.json restricted to the eligible first-300 tiles",
               tile_ids=S["p300"], style_mapping=style), open(f"{out}/crc33sec_p300_unpaired.json", "w"), indent=1)
union = sorted(set(S["fig2"]) | set(S["p300"]) | set(S["decomp"]))
open(f"{out}/crc33sec_pack_ids.txt", "w").write("".join(f"{t}\n" for t in union))
open(f"{out}/crc33sec_style_ids.txt", "w").write("".join(f"{t}\n" for t in sorted(set(style.values()) - set(union))))
R["pack_tiles"], R["style_only"] = len(union), len(set(style.values()) - set(union))
json.dump(R, open(f"{out}/sets33.json", "w"), indent=1)
print(json.dumps(R))
