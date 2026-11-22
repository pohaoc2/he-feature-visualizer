"""PREREG §8 stats: L1b cross-PQ with the diag.py bootstrap (same row order, rng(42), B = 10,000), paired gains.
  python l1b_stats.py DATA_DIR   (tiles_<set>.csv from diag, l1b_<set>.csv from l1b.py) -> DATA_DIR/l1b_stats.json"""
import csv, json, sys
import numpy as np
D = sys.argv[1]
rows = []
for s in ("crc33sec", "crc36ho"):
    l = {r["tile"]: r for r in csv.DictReader(open(f"{D}/l1b_{s}.csv"))}
    for r in csv.DictReader(open(f"{D}/tiles_{s}.csv")):
        r.update(l[r["tile"]]); rows.append(r)
f = lambda a, b: np.array([(float(r[a]) + float(r[b])) / 2 for r in rows])
V = dict(R0=f("pq_R0_cv", "pq_R0_sd"), L1b=f("pq_L1b_cv", "pq_L1b_sd"), L1=f("pq_L1_cv", "pq_L1_sd"),
         R0_rerendered=f("pq_R0re_cv", "pq_R0re_sd"))
idx = np.random.default_rng(42).integers(0, len(rows), size=(10000, len(rows)))
ci = lambda v: dict(point=float(v.mean()), ci95=[float(np.quantile(v[idx].mean(1), q)) for q in (.025, .975)])
out = dict(n_tiles=len(rows), cross_pq={k: ci(v) for k, v in V.items()},
           gain_R0_to_L1b=ci(V["L1b"] - V["R0"]), gain_L1b_to_L1=ci(V["L1"] - V["L1b"]), gain_R0_to_L1=ci(V["L1"] - V["R0"]),
           per_set={s: {k: float(np.mean([v[i] for i, r in enumerate(rows) if r["set"] == s])) for k, v in V.items()} for s in ("crc33sec", "crc36ho")},
           R0_gate_all=all(r["gate_R0"] == "True" for r in rows))
a, b = out["cross_pq"]["L1b"]["ci95"], out["cross_pq"]["L1"]["ci95"]
out["overlap_L1b_L1"] = not (a[1] < b[0] or b[1] < a[0])
out["choice"] = "L1b" if out["overlap_L1b_L1"] else "L1"
out["share_of_gain_from_separation"] = out["gain_R0_to_L1b"]["point"] / out["gain_R0_to_L1"]["point"]
json.dump(out, open(f"{D}/l1b_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))
