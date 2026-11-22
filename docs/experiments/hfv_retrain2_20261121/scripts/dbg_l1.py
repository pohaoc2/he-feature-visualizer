import csv, os, sys, numpy as np, tifffile, cv2
from scipy import ndimage
sys.path.insert(0, os.environ["HFV"]); sys.path.insert(0, os.environ["SEV_SRC"]); sys.path.insert(0, os.path.dirname(__file__))
from stages.lin_typing import nuclei as NU
from sim_exp_validation.morphology_renderer import separate_instances
import diag as D
tif, tcsv, tiles = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ref = {r["tile"]: r for r in csv.DictReader(open(tcsv))}
a = tifffile.imread(tif)
D.G["lab"] = a
for t in tiles:
    x0, y0 = map(int, t.split("_"))
    lab = a; H, W = lab.shape
    out = np.zeros((272, 272), np.int64)
    ya, yb = max(y0 - 8, 0), min(y0 + 264, H); xa, xb = max(x0 - 8, 0), min(x0 + 264, W)
    out[ya - (y0 - 8):yb - (y0 - 8), xa - (x0 - 8):xb - (x0 - 8)] = lab[ya:yb, xa:xb]
    _, loose = NU.erode_labels(out)
    L = loose[8:-8, 8:-8].astype(np.int64)
    s1 = D.separate(L); s2 = separate_instances(L)
    n1 = int(D.relabel(s1 > 0).max()); n2 = int(D.relabel(s2 > 0).max())
    print(t, "diag_csv", ref[t]["n_L1"], "n_lin_tile_csv", ref[t]["n_lin_tile"], "diag_sep", n1, "sev_sep", n2, "nlab", len(np.unique(L)) - 1,
          "pix_diff", int(((s1 > 0) != (s2 > 0)).sum()), flush=True)
