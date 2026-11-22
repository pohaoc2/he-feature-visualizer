"""paperpq_official_20261122 (PREREG.md §3-4): build the four generation data roots and run the input gates.

  python pp_prep.py D
D = /gscratch/cheme/chiu/paperpq_official_20261122 (in/ from pp_fetch.slurm). cwd = a PixCell cb62fa7 copy on PYTHONPATH.
Roots (each one is a PixCell data root: exp_channels/<10 channels>, features/<t>_uni.npy, he/<t>.png; 100 tiles):
  P_run  paper inputs unchanged (orion-crc33.zip)                       -> CellViT-mask generators + Vanilla, runner pipeline
  P_off  paper channels transferred onto the official CellViT instances -> CellViT-mask generators + Vanilla, official
  R_run  shape-library (R0) channels transferred onto the 8-connected components of the paper cell_masks -> retrained, runner
  R_off  R0 channels transferred onto the official CellViT instances    -> retrained generators, official pipeline
Official instances: cellvit_official_20261122 full/cv/F2_P_gt/<t>.json filled with PixCell _cellvit_json_to_instance_mask;
mask PNG = (instance > 0) * 255 (the realref.py drawing of the paper cell_masks). Label transfer (PREREG §3.2): a new
object is matched when > 1/2 of its pixels lie on the old cell_masks foreground; type / state = the old channel covering
most of its overlap pixels (ties -> first in channel order, counted; no covering channel -> left empty, counted); the
whole object is drawn into the chosen channels; unmatched objects go into cell_masks only.
Writes D/roots/<root>/..., D/prep/{transfer_tiles.csv, transfer_summary.json, gates.json}. Exit 1 on any gate failure.
"""
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _label_binary_instances

D = Path(sys.argv[1])
CC = Path("/gscratch/cheme/chiu")
CVO = CC / "cellvit_official_20261122"
EVALPACK = CC / "hfv_eval_20261118/packs/evalpack_crc33sec.tar"
IDS_HFV = CC / "hfv_eval_20261118/packs/crc33sec_fig2_ids.txt"
TYPES = ["cell_type_healthy", "cell_type_cancer", "cell_type_immune"]
STATES = ["cell_state_prolif", "cell_state_nonprolif", "cell_state_dead"]
CELL = ["cell_masks"] + TYPES + STATES
TME = ["vasculature", "oxygen", "glucose"]
OUT = D / "roots"
PREP = D / "prep"
if OUT.exists() or PREP.exists():
    sys.exit(f"STOP {OUT} or {PREP} exists (new dirs only)")
PREP.mkdir(parents=True)
gates, fail = {}, []


def gate(name, ok, detail):
    gates[name] = dict(ok=bool(ok), detail=detail)
    print(f"[gate] {name}: {'ok' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        fail.append(name)


# ---- tile ids
ids = sorted(json.load(open(D / "in/cache.json.bak_n100"))["metric_tile_ids"])
ids_hfv = sorted(t.strip() for t in open(IDS_HFV) if t.strip())
gate("ids", len(ids) == 100 and ids == ids_hfv, dict(paper=len(ids), hfv=len(ids_hfv), same=ids == ids_hfv))
(PREP / "ids100.txt").write_text("".join(t + "\n" for t in ids))

# ---- sources: 100 tiles of the paper zip and of the shape-library evaluation pack, kept as bytes
want = set(ids)


def tid(name):
    s = Path(name).name
    return s[:-len("_uni.npy")] if s.endswith("_uni.npy") else Path(s).stem


src = {"P": {}, "R": {}}   # family -> relative path (exp_channels/<ch>/<file>, features/<f>, he/<f>) -> bytes
with zipfile.ZipFile(D / "in/orion-crc33.zip") as z:
    for n in z.namelist():
        parts = n.split("/")
        if n.endswith("/") or len(parts) < 3 or parts[1] not in ("exp_channels", "features", "he"):
            continue
        if parts[1] == "exp_channels" and parts[2] not in CELL + TME:
            continue
        if tid(n) in want:
            src["P"]["/".join(parts[1:])] = z.read(n)
with tarfile.open(EVALPACK) as t:
    for m in t:
        parts = m.name.split("/")
        if not m.isfile() or len(parts) < 3 or parts[1] not in ("exp_channels", "features", "he"):
            continue
        if parts[1] == "exp_channels" and parts[2] not in CELL + TME:
            continue
        if tid(m.name) in want:
            src["R"]["/".join(parts[1:])] = t.extractfile(m).read()


def find(fam, ch, t):
    k = [p for p in src[fam] if p.startswith(f"exp_channels/{ch}/") and Path(p).stem == t]
    return k


miss = {}
for fam in src:
    for t in ids:
        for ch in CELL + TME:
            if len(find(fam, ch, t)) != 1:
                miss.setdefault(fam, []).append(f"{ch}/{t}")
        for p in (f"features/{t}_uni.npy", f"he/{t}.png"):
            if p not in src[fam]:
                miss.setdefault(fam, []).append(p)
gate("sources_complete", not miss, {k: (len(v), v[:5]) for k, v in miss.items()})
if fail:
    json.dump(gates, open(PREP / "gates.json", "w"), indent=1)
    sys.exit("STOP " + ", ".join(fail))


def png(b):
    return np.asarray(Image.open(io.BytesIO(b)))


def cellch(fam, ch, t):
    return png(src[fam][find(fam, ch, t)[0]])


# ---- real H&E identical across the paper zip, the evaluation pack and the scored real tiles (F2_P_gt.zip)
he_bad = []
with zipfile.ZipFile(CVO / "zips/F2_P_gt.zip") as z:
    for t in ids:
        a, b, c = png(src["P"][f"he/{t}.png"]), png(src["R"][f"he/{t}.png"]), png(z.read(f"{t}.png"))
        if not (a.shape == b.shape == c.shape and (a == b).all() and (a == c).all()):
            he_bad.append(t)
gate("real_he_identical", not he_bad, dict(n_diff=len(he_bad), e=he_bad[:5]))

# ---- cell channels binary 0/255 256x256; old type / state channels inside the old mask
bad = []
for fam in src:
    for t in ids:
        for ch in CELL:
            a = cellch(fam, ch, t)
            if a.shape != (256, 256) or not set(np.unique(a)) <= {0, 255}:
                bad.append(f"{fam}:{ch}/{t}:{a.shape}:{np.unique(a)[:4]}")
gate("cell_channels_binary", not bad, dict(n=len(bad), e=bad[:5]))

# ---- official CellViT JSON present for all tiles
jdir = CVO / "full/cv/F2_P_gt"
jmiss = [t for t in ids if not (jdir / f"{t}.json").is_file()]
gate("official_json", not jmiss, dict(missing=len(jmiss)))

# ---- descriptive: runner JSON (Hyak rerun) drawn the PixCell way == paper cell_masks
rdir = CVO / "full_runner/cv/F2_P_gt"
same = sum(int((((_cellvit_json_to_instance_mask(rdir / f"{t}.json", image_hw=(256, 256)) > 0) * 255).astype(np.uint8)
                == cellch("P", "cell_masks", t)).all()) for t in ids)
gates["runner_redraw_eq_paper_cell_masks"] = dict(ok=True, detail=dict(identical=same, of=len(ids)), descriptive=True)
print(f"[desc] runner JSON redraw == paper cell_masks: {same}/{len(ids)}", flush=True)

# ---- descriptive: UNI and TME files identical between the two families
for kind in ("uni", *TME):
    n = 0
    for t in ids:
        if kind == "uni":
            a, b = (np.load(io.BytesIO(src[f][f"features/{t}_uni.npy"])) for f in "PR")
        else:
            pa, pr = find("P", kind, t)[0], find("R", kind, t)[0]
            a, b = ((np.load(io.BytesIO(src[f][p])) if p.endswith(".npy") else png(src[f][p]))
                    for f, p in (("P", pa), ("R", pr)))
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            if a.max() > 1.5 and b.max() <= 1.5:
                a = a / 255.0
            elif b.max() > 1.5 and a.max() <= 1.5:
                b = b / 255.0
        n += int(a.shape == b.shape and np.allclose(a, b, atol=1e-6))
    gates[f"family_{kind}_identical"] = dict(ok=True, detail=dict(identical=n, of=len(ids)), descriptive=True)
    print(f"[desc] {kind} identical P vs R: {n}/{len(ids)}", flush=True)


# ---- label transfer
def transfer(new_inst, fam, t):
    old = cellch(fam, "cell_masks", t) > 0
    ot = np.stack([cellch(fam, c, t) > 0 for c in TYPES])
    os_ = np.stack([cellch(fam, c, t) > 0 for c in STATES])
    nt, ns = np.zeros_like(ot), np.zeros_like(os_)
    st = dict(new=0, matched=0, unmatched=0, type_tie=0, state_tie=0, type_empty=0, state_empty=0)
    for l in np.unique(new_inst):
        if l == 0:
            continue
        o = new_inst == l
        st["new"] += 1
        ov = o & old
        if 2 * int(ov.sum()) <= int(o.sum()):
            st["unmatched"] += 1
            continue
        st["matched"] += 1
        for oldch, newch, k in ((ot, nt, "type"), (os_, ns, "state")):
            cnt = np.array([int((ov & c).sum()) for c in oldch])
            if cnt.max() == 0:
                st[f"{k}_empty"] += 1
                continue
            st[f"{k}_tie"] += int((cnt == cnt.max()).sum() > 1)
            newch[int(np.argmax(cnt))] |= o
    return nt, ns, st


def write(root, rel, data):
    p = OUT / root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def save_png(root, ch, t, arr):
    b = io.BytesIO()
    Image.fromarray((arr.astype(np.uint8) * 255) if arr.dtype == bool else arr, mode="L").save(b, format="PNG")
    write(root, f"exp_channels/{ch}/{t}.png", b.getvalue())


rows = []
chk = []
for t in ids:
    off = _cellvit_json_to_instance_mask(jdir / f"{t}.json", image_hw=(256, 256))
    run = _label_binary_instances(cellch("P", "cell_masks", t) > 0)
    for root, fam, inst in (("P_run", "P", None), ("P_off", "P", off), ("R_run", "R", run), ("R_off", "R", off)):
        for p, b in src[fam].items():   # family files (TME, UNI, H&E; cell channels overwritten below unless P_run)
            if Path(p).stem == t or p in (f"features/{t}_uni.npy",):
                if inst is not None and p.startswith("exp_channels/") and p.split("/")[1] in CELL:
                    continue
                write(root, p, b)
        if inst is None:
            continue
        mask = inst > 0
        nt, ns, st = transfer(inst, fam, t)
        save_png(root, "cell_masks", t, mask)
        for c, a in zip(TYPES, nt):
            save_png(root, c, t, a)
        for c, a in zip(STATES, ns):
            save_png(root, c, t, a)
        # PREREG §3.2 checks: channels inside the mask; one type / one state per matched object (empty excepted)
        inside = all(not (a & ~mask).any() for a in list(nt) + list(ns))
        overl = int((nt.sum(0) > 1).sum() + (ns.sum(0) > 1).sum())
        if not inside:
            chk.append(f"{root}/{t}: channel outside mask")
        rows.append(dict(root=root, tile=t, **st, multi_label_px=overl))

# multi_label_px > 0 is possible only where two new objects overlap; instance maps do not overlap -> must be 0
over = [r for r in rows if r["multi_label_px"] > 0]
gate("transfer_checks", not chk and not over, dict(outside=chk[:5], multi_label=len(over)))

import csv  # noqa: E402

with open(PREP / "transfer_tiles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
summ = {}
for root in ("P_off", "R_run", "R_off"):
    rr = [r for r in rows if r["root"] == root]
    s = {k: int(sum(r[k] for r in rr)) for k in ("new", "matched", "unmatched", "type_tie", "state_tie", "type_empty",
                                                   "state_empty")}
    s["unmatched_frac"] = round(s["unmatched"] / s["new"], 4)
    s["tiles_with_unmatched"] = sum(r["unmatched"] > 0 for r in rr)
    summ[root] = s
    print(f"[transfer] {root}: {s}", flush=True)
json.dump(summ, open(PREP / "transfer_summary.json", "w"), indent=1)

# P_off and R_off carry the identical official mask; P_run cell_masks == paper
eq = sum(int((np.asarray(Image.open(OUT / "P_off/exp_channels/cell_masks" / f"{t}.png")) ==
              np.asarray(Image.open(OUT / "R_off/exp_channels/cell_masks" / f"{t}.png"))).all()) for t in ids)
gate("official_mask_shared", eq == len(ids), dict(identical=eq))

# every root complete
for root in ("P_run", "P_off", "R_run", "R_off"):
    n = {ch: len(list((OUT / root / "exp_channels" / ch).glob("*"))) for ch in CELL + TME}
    n["features"] = len(list((OUT / root / "features").glob("*_uni.npy")))
    n["he"] = len(list((OUT / root / "he").glob("*.png")))
    gate(f"root_{root}_complete", all(v == len(ids) for v in n.values()), n)

# per-channel config readable by PixCell read_config
try:
    from diffusion.utils.misc import read_config

    cfg = read_config(str(D / "in/ck/a1_per_channel/config.py"))
    gate("perch_config", True, dict(active_channels=list(cfg.data.active_channels)))
except Exception as e:  # noqa: BLE001
    gate("perch_config", False, repr(e))

h = hashlib.sha256()
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        h.update(str(p.relative_to(OUT)).encode())
        h.update(p.read_bytes())
gates["roots_sha256"] = h.hexdigest()
json.dump(gates, open(PREP / "gates.json", "w"), indent=1)
if fail:
    sys.exit("STOP " + ", ".join(fail))
print("[prep] OK", flush=True)
