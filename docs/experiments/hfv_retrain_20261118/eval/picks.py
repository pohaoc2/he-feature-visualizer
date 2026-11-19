"""hfv_retrain_20261118 evaluation tile picks (README rules, fixed before any evaluation result).

  python picks.py POOL_TXT N OUT_PREFIX
POOL_TXT  every eligible tile of one slide, one id per line (training filters applied; CRC36: the hold-out tiles, guard
          band already excluded)
N         300
Picks:    all of the pool when it has <= N tiles, else numpy default_rng(42).choice(sorted(pool), N, replace=False)
          (the same call as the CellViT gate, crc34_render_20261118/scripts/cvprep34.py), written sorted.
Unpaired: a derangement of the picks with PixCell's own routine (tools/stage3/prepare_unpaired_ablation_dataset.py
          _derangement, random.Random(42) shuffled until no tile maps to itself), copied here unchanged so the kit does not
          need the PixCell tree to rebuild it.
Writes OUT_PREFIX_ids.txt and OUT_PREFIX_unpaired.json ({"seed", "tile_ids", "style_mapping"}, the format
load_style_mapping reads).
"""
import json
import random
import sys

import numpy as np


def _derangement(items, seed):
    if len(items) < 2:
        raise ValueError("need at least 2 items to build an unpaired mapping")
    rng = random.Random(seed)
    shuffled = list(items)
    for _ in range(10_000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(items, shuffled, strict=True)):
            return list(shuffled)
    shuffled = sorted(items)
    offset = (seed % (len(items) - 1)) + 1
    return shuffled[offset:] + shuffled[:offset]


pool_txt, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pool = sorted({t.strip() for t in open(pool_txt) if t.strip()})
picks = pool if len(pool) <= n else sorted(np.random.default_rng(42).choice(pool, n, replace=False).tolist())
style = _derangement(picks, 42)
assert all(a != b for a, b in zip(picks, style))
open(f"{out}_ids.txt", "w").write("".join(f"{t}\n" for t in picks))
json.dump(dict(seed=42, pool=len(pool), rule="default_rng(42).choice(sorted(pool), N) / PixCell _derangement seed 42",
               tile_ids=picks, style_mapping=dict(zip(picks, style))), open(f"{out}_unpaired.json", "w"), indent=1)
print(f"pool {len(pool)} picks {len(picks)} first {picks[:3]}")
