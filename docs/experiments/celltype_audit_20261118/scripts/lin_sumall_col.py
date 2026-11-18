# prereg_lin_reinit_20261106 §C5: one more slide column of Lin's published sumAllsample table (orion-crc
# matlab-sumAllsample-20220710.mat, MATLAB v7.3 table) added to lin_sumall.json.
# The table's variables live in the MATLAB object subsystem, so each existing json column is matched by value: the (1, 40)
# double arrays whose crc16 entry (row C16) and nanmedian equal the json's; every match of a column must agree on the new
# row. The rows are the 40-entry row-name array C01..C40 found in the file.
# Gate (exit 1 otherwise): every column lin_gate.py reads (NEED: the marker positive fractions mean_<m>p and the identity
# means) matched to one value. Columns whose matches disagree (0/1 clinical columns, mean_Solidity) keep only crc16 / median.
#   lin_sumall_col.py SUMALL.mat lin_sumall.json OUT.json C34 crc34
import json, sys
import h5py, numpy as np

IDENT = ["mean_Area", "mean_Hoechst", "mean_Pan_CK", "mean_CD45", "mean_Ki67", "mean_E_cadherin"]


def main(mat, src, out, row, key):
    f = h5py.File(mat, "r")
    d = json.load(open(src))
    arrs, names = [], None

    def visit(n, o):
        nonlocal names
        if not isinstance(o, h5py.Dataset) or o.size != 40:
            return
        if o.dtype == np.float64:
            arrs.append((n, o[()].ravel()))
        elif o.dtype == h5py.ref_dtype:
            s = ["".join(map(chr, f[r][()].ravel())) if f[r].dtype == np.uint16 else None for r in o[()].ravel()]
            if s == [f"C{i:02d}" for i in range(1, 41)]:
                names = s
    f.visititems(visit)
    assert names == d["slides"], (names, d["slides"])
    i16, inew = names.index("C16"), names.index(row)
    cols, bad = {}, []
    for c, v in d["cols"].items():
        m = [(n, a) for n, a in arrs if a[i16] == v["crc16"] and float(np.nanmedian(a)) == v["median"]]
        vals = {float(a[inew]) for _, a in m}
        if not m or len(vals) != 1:
            bad.append((c, len(m), sorted(vals)))
            cols[c] = dict(v)
            continue
        cols[c] = dict(v, **{key: vals.pop()})
    need = [c for c in d["cols"] if c.startswith("mean_") and c.endswith("p")] + IDENT
    ok = all(key in cols[c] for c in need)
    json.dump(dict(slides=names, cols=cols,
                   extract=dict(mat=mat, row=row, key=key, index=inew, n_cols=len(d["cols"]), need=need,
                                matched=len(cols) - len(bad), unmatched=bad, ok=ok)), open(out, "w"), indent=1)
    print(json.dumps(dict(ok=ok, matched=len(cols) - len(bad), of=len(d["cols"]), unmatched=[b[0] for b in bad],
                          need={c: cols[c].get(key) for c in need})))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:6])
