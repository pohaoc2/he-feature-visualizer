"""crc36_render_20261118 (envsmc; numpy/pandas): CRC33_01 + CRC36 training set numbers and a spatial CRC36 hold-out.

  combine.py N L        N = hfv_crc36_20261118 dir, L = hfv_lintype_20261118 dir (CRC33_01 run), both read only except N/out
Hold-out rule: one CELESTA k-d core rectangle of CRC36 (16 rectangles of equal cell count, N/celesta/tiles.tsv), the one
whose share of kept-tile TUM-class cells is closest to the slide's; a tile belongs to the rectangle holding its centre.
Kept tiles that touch the hold-out (8-neighbour) are a guard band, left out of training so no training tile borders it.
Training numbers use bs 80 and 20 epochs of the CRC33_01 run (config_controlnet_exp_a1_concat.py), and the measured
original rate (2,600 steps x 80 in 38 min on one H200).
"""
import json, math, sys
import numpy as np
import pandas as pd

N, L = sys.argv[1:3]
NAMES = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
         "cell_state_nonprolif", "cell_state_dead"]


def frac(path):
    r = np.load(path)
    return dict(zip(r["tile"].tolist(), r["frac"]))


f33 = frac(f"{L}/out/render_tiles.npz")
f01 = frac(f"{N}/out/render_tiles.npz")
S33 = json.load(open(f"{L}/out/summary.json"))
old = {n: S33["channels_keep"][n]["old_mean"] for n in NAMES}

kt = pd.read_csv(f"{N}/celesta/tiles.tsv", sep="\t")
c = np.load(f"{N}/out/cells_labels.npz")
keep = sorted(f01)
xy = np.array([list(map(int, t.split("_"))) for t in keep])
cx, cy = xy[:, 0] + 128, xy[:, 1] + 128


def rect_of(x, y):
    r = np.zeros(len(x), int)
    for _, row in kt.iterrows():
        m = (x >= row.x0) & (x <= row.x1) & (y >= row.y0) & (y <= row.y1) & (r == 0)
        r[m] = int(row.k)
    return r


rt = rect_of(cx, cy)
tum = (c["lin_cls"] == 3) & c["in_keep"]
rc = rect_of(c["x"], c["y"])
share = {int(k): float(tum[rc == k].sum() / max((c["in_keep"] & (rc == k)).sum(), 1)) for k in kt.k}
slide_share = float(tum.sum() / c["in_keep"].sum())
hold = min(share, key=lambda k: abs(share[k] - slide_share))
H = rt == hold
hs = {(int(x), int(y)) for x, y in xy[H]}
guard = np.array([(not H[i]) and any((x + dx, y + dy) in hs for dx in (-256, 0, 256) for dy in (-256, 0, 256))
                  for i, (x, y) in enumerate(xy)])
train01 = ~H & ~guard


def mean_of(fr, tiles):
    a = np.array([fr[t] for t in tiles])
    return a.mean(0), len(tiles)


m33, n33 = mean_of(f33, sorted(f33))
m01_all, n01 = mean_of(f01, keep)
m01_tr, n01_tr = mean_of(f01, [t for t, k in zip(keep, train01) if k])
both = (m33 * n33 + m01_tr * n01_tr) / (n33 + n01_tr)
both_all = (m33 * n33 + m01_all * n01) / (n33 + n01)


def plan(n):
    spe = math.ceil(n / 80)
    steps = spe * 20
    rate = 2600 * 80 / (38 * 60)
    return dict(tiles=n, steps_per_epoch=spe, steps_20_epochs=steps, samples=steps * 80,
                h200_min=round(steps * 80 / rate, 1), h200_usd_at_0p90=round(steps * 80 / rate / 60 * 0.90, 2))


R = dict(
    holdout=dict(rule="k-d core rectangle with TUM share closest to the slide's; 8-neighbour guard band",
                 rect=int(hold), rect_tum_share=round(share[hold], 4), slide_tum_share=round(slide_share, 4),
                 tum_share_by_rect={k: round(v, 4) for k, v in share.items()},
                 holdout_tiles=int(H.sum()), guard_tiles=int(guard.sum()), train_tiles=int(train01.sum()),
                 holdout_tile_ids=[t for t, h in zip(keep, H) if h]),
    mean_frac=dict(
        current_crc33_01_old=old,
        crc33_01_new=dict(zip(NAMES, np.round(m33, 4).tolist())),
        crc36_all_kept=dict(zip(NAMES, np.round(m01_all, 4).tolist())),
        crc36_train=dict(zip(NAMES, np.round(m01_tr, 4).tolist())),
        crc33_01_new_plus_crc36_train=dict(zip(NAMES, np.round(both, 4).tolist())),
        crc33_01_new_plus_crc36_all=dict(zip(NAMES, np.round(both_all, 4).tolist()))),
    tiles_with_cancer=dict(crc33_01_new=int(sum(f33[t][2] > 0 for t in f33)),
                           crc36_kept=int(sum(f01[t][2] > 0 for t in f01))),
    plan=dict(crc33_01_only=plan(n33), crc33_01_plus_crc36_train=plan(n33 + n01_tr),
              crc33_01_plus_crc36_all=plan(n33 + n01),
              fixed_2360_steps_note="keeping 2,360 steps with the larger set = 20 epochs -> about "
                                    f"{round(2360 * 80 / (n33 + n01_tr), 1)} epochs"))
json.dump(R, open(f"{N}/out/combine.json", "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "holdout"}, indent=1))
print(json.dumps({k: v for k, v in R["holdout"].items() if k != "holdout_tile_ids"}))
