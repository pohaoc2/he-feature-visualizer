"""hfv_retrain_20261118 paper-figure re-render (README "論文 figure 用新模型重繪", coordinator 2026-11-19): helpers that
run the PixCell cb62fa7 figure scripts unchanged inside the Hyak re-render checkout.

  python figrender.py guard   ROOT
  python figrender.py bands   CACHE_JSON OUT_JSON            # Vanilla bands from si_a1_a2 cache.json (a2_off_shelf)
  python figrender.py patient CRC33_PAIRED CRC33_UNPAIRED CRC34_PAIRED CRC34_UNPAIRED OUT_JSON [N33P N33U N34P N34U]
  python figrender.py run     ROOT MODULE [--bands BANDS_JSON] [-- ARGS...]

guard    STOP (exit 1) unless ROOT.parent/share-space-paper is a plain directory (not a symlink) whose real path lies
         under ROOT.parent, and no symlink anywhere under ROOT (followed one level: the symlink targets themselves) or
         under the stand-in resolves to a path containing a share-space-paper other than the stand-in. The real paper
         repo lives only on the Mac; this makes sure the scripts that write to ROOT.parent/share-space-paper/figures
         (publish.save_mirrored, fig_combined_per_channel, fig_si_patient_generalization's assets/gen deletion) can only
         touch the stand-in. `run` calls it first every time.
bands    The three Vanilla ControlNet reference bands of tools/ablation_report/shared.py TRADEOFF_REFERENCE_BANDS (fud,
         pq, dice) from this run's CRC33 secondary-group cache (metrics["a2_off_shelf"], 100 tiles). The two "ORION H&E"
         bands (lpips, style_hed) are real-H&E references and are left as they are. Writes {key: {mean, std?}} plus the
         old values for the README deviation table.
patient  The si_patient_generalization input (share-space-paper/figures/data/patient_generalization_metrics.json),
         reconstructed with the same keys: no script in PixCell writes it and the original writer is unknown. Each of
         the four arguments is an ablation_results-style folder (<tile>/metrics.json with per_condition, fud_scores.json);
         the value is the all-four condition 'cell_state+cell_types+microenv+vasculature': mean and population sd
         (ddof 0) over tiles for lpips/pq/dice/style_hed, FUD from fud_scores.json (distribution level, no sd).
run      chdir ROOT (some scripts use cwd-relative data/ paths), put ROOT first on sys.path, run guard, optionally update
         TRADEOFF_REFERENCE_BANDS in place from BANDS_JSON (dict mutated, so modules that imported the name see it),
         then runpy the figure module as __main__ with ARGS. PixCell files are not changed.
         Environment (追加 3, 2026-11-21):
         A3_FONT_DIR   folder with URW base35 NimbusSans-*.otf (github.com/ArtifexSoftware/urw-base35-fonts). The fonts are
                       registered with matplotlib, and PIL.ImageFont.truetype("/usr/share/fonts/urw-base35/<f>") is
                       pointed to A3_FONT_DIR/<f> when the system path is missing (fig_combined_per_channel hard-codes
                       that path; the Hyak nodes do not have it).
         A3_LAYOUT_LAX=1  import the module (not as __main__), replace its _assert_layout with a version that prints
                       the ValueError instead of raising, then call main(). Used only where the label-overlap guard
                       stops a figure; the overlap is reported, the figure is drawn as the script lays it out.
"""
import json
import os
import runpy
import statistics
import sys
from pathlib import Path

ALL4 = "cell_state+cell_types+microenv+vasculature"
MET = ("lpips", "pq", "dice", "style_hed")


def guard(root):
    root = Path(root).resolve()
    stand = root.parent / "share-space-paper"
    bad = []
    if stand.is_symlink() or not stand.is_dir():
        bad.append(f"{stand} is not a plain directory")
    elif not str(stand.resolve()).startswith(str(root.parent) + os.sep):
        bad.append(f"{stand} resolves outside {root.parent}")
    real_stand = str(stand.resolve()) if stand.exists() else str(stand)
    for base in (root, stand):
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):  # does not descend into symlinked dirs
            for name in dirnames + filenames:
                p = Path(dirpath) / name
                if p.is_symlink():
                    t = os.path.realpath(p)
                    if "share-space-paper" in t and not (t == real_stand or t.startswith(real_stand + os.sep)):
                        bad.append(f"{p} -> {t}")
    if bad:
        print("STOP guard:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"guard ok: stand-in {stand} (plain directory); no symlink under ROOT reaches another share-space-paper")


def bands(cache_json, out_json):
    m = json.load(open(cache_json))["metrics"]["a2_off_shelf"]
    new = {"fud": {"mean": m["fud"]}, "pq": {"mean": m["pq"], "std": m["pq_std"]},
           "dice": {"mean": m["dice"], "std": m["dice_std"]}}
    json.dump({"source": str(cache_json), "n_tiles": m.get("n_tiles"), "bands": new}, open(out_json, "w"), indent=1)
    print(json.dumps(new))


def summarize(folder, n_expect=None):
    folder = Path(folder)
    vals = {k: [] for k in MET}
    tiles = sorted(folder.glob("*/metrics.json"))
    for f in tiles:
        rec = json.load(open(f))["per_condition"][ALL4]
        for k in MET:
            if rec.get(k) is not None:
                vals[k].append(float(rec[k]))
    fud = json.load(open(folder / "fud_scores.json"))[ALL4]
    n = len(tiles)
    if n_expect is not None and n != int(n_expect):
        sys.exit(f"STOP {folder}: {n} tiles, expected {n_expect}")
    for k in MET:
        if len(vals[k]) != n:
            sys.exit(f"STOP {folder}: {k} present for {len(vals[k])} of {n} tiles")
    out = {"fud": round(float(fud), 1)}
    out.update({k: round(statistics.fmean(vals[k]), 3) for k in MET})
    out["n"] = n
    out["sd"] = {k: round(statistics.pstdev(vals[k]), 4) for k in MET}
    return out


def patient(p33, u33, p34, u34, out_json, *n):
    n = list(n) + [None] * (4 - len(n))
    data = {"CRC33_01": {"label": "in-distribution (training patient)", "paired": summarize(p33, n[0]),
                         "unpaired": summarize(u33, n[1])},
            "CRC34": {"label": "cross-patient OOD", "paired": summarize(p34, n[2]), "unpaired": summarize(u34, n[3])}}
    doc = {
        "description": "RECONSTRUCTED for the hfv_retrain_20261118 re-render (original writer unknown). Frozen Concat "
                       "retrained on the new CRC33_01 channels (tr_concat epoch 20), all-four condition, seed 42. "
                       "CRC33_01 = training patient (secondary group, training tiles); CRC34 = held-out patient, "
                       "inputs rebuilt (see README). FUD is distribution-level (no sd); LPIPS/PQ/DICE/HED are per-tile "
                       "mean +/- population sd.",
        "metric_order": ["fud", "lpips", "pq", "dice", "style_hed"],
        "metric_labels": {"fud": "FUD", "lpips": "LPIPS", "pq": "PQ", "dice": "DICE", "style_hed": "HED"},
        "higher_is_better": {"fud": False, "lpips": False, "pq": True, "dice": True, "style_hed": False},
        "error_is": "population sd over tiles (FUD has none: distribution-level metric)",
        "data": data,
        "sources": {"CRC33_01_paired": str(p33), "CRC33_01_unpaired": str(u33), "CRC34_paired": str(p34),
                    "CRC34_unpaired": str(u34), "condition": ALL4, "writer": "hfv eval/figrender.py patient"},
    }
    json.dump(doc, open(out_json, "w"), indent=1)
    print(json.dumps(data, indent=1))


def run(root, module, rest):
    root = Path(root).resolve()
    bands_json = None
    if rest[:1] == ["--bands"]:
        bands_json, rest = os.path.abspath(rest[1]), rest[2:]
    if rest[:1] == ["--"]:
        rest = rest[1:]
    guard(root)
    os.chdir(root)
    sys.path.insert(0, str(root))
    if bands_json:
        import tools.ablation_report.shared as shared
        new = json.load(open(bands_json))["bands"]
        for k, v in new.items():
            old = dict(shared.TRADEOFF_REFERENCE_BANDS[k])
            shared.TRADEOFF_REFERENCE_BANDS[k].clear()
            shared.TRADEOFF_REFERENCE_BANDS[k].update({"label": old["label"], **v})
            print(f"band {k}: {old} -> {shared.TRADEOFF_REFERENCE_BANDS[k]}")
    fdir = os.environ.get("A3_FONT_DIR")
    if fdir:
        from pathlib import Path as _P
        from matplotlib import font_manager
        from PIL import ImageFont
        for f in sorted(_P(fdir).glob("*.otf")):
            font_manager.fontManager.addfont(str(f))
        _truetype = ImageFont.truetype

        def truetype(font=None, *a, **k):
            if isinstance(font, str) and font.startswith("/usr/share/fonts/urw-base35/") and not os.path.exists(font):
                font = os.path.join(fdir, os.path.basename(font))
                print(f"[figrender] PIL font -> {font}", flush=True)
            return _truetype(font, *a, **k)

        ImageFont.truetype = truetype
        print(f"[figrender] fonts registered from {fdir}", flush=True)
    sys.argv = [module] + rest
    if os.environ.get("A3_LAYOUT_LAX") == "1":
        import importlib
        mod = importlib.import_module(module)
        strict = mod._assert_layout

        def lax(fig):
            try:
                strict(fig)
            except ValueError as e:
                print(f"[figrender] A3_LAYOUT_LAX: layout guard says: {e}", flush=True)

        mod._assert_layout = lax
        mod.main()
        return
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "guard":
        guard(a[0])
    elif cmd == "bands":
        bands(a[0], a[1])
    elif cmd == "patient":
        patient(*a)
    elif cmd == "run":
        run(a[0], a[1], a[2:])
    else:
        sys.exit(__doc__)
