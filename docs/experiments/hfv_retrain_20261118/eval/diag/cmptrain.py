import tarfile, hashlib, sys
ids = [l.strip() for l in open("ids100.txt") if l.strip()]
CH = ["cell_masks","cell_type_healthy","cell_type_cancer","cell_type_immune","cell_state_prolif","cell_state_nonprolif","cell_state_dead"]
def md5s(tarpath, keyf):
    d = {}
    with tarfile.open(tarpath) as tf:
        for m in tf:
            k = keyf(m.name)
            if k: d[k] = hashlib.md5(tf.extractfile(m).read()).hexdigest()
    return d
idset = set(ids)
def tr(n):
    p = n.split("/")
    if len(p) == 4 and p[1] == "exp_channels" and p[2] in CH and p[3][:-4] in idset: return (p[2], p[3][:-4])
    if n.startswith("masklat/") and n.endswith("_mask_sd3_vae.npy") and n.split("/")[1][:-17] in idset: return ("masklat", n.split("/")[1][:-17])
def ev(n):
    p = n.split("/")
    if len(p) == 4 and p[1] == "exp_channels" and p[2] in CH and p[3][:-4] in idset: return (p[2], p[3][:-4])
    if len(p) == 3 and p[1] == "vae_features" and p[2].endswith("_mask_sd3_vae.npy") and p[2][:-17] in idset: return ("masklat", p[2][:-17])
a = md5s("train33/new_exp_channels_crc33_01.tar", tr); a.update(md5s("train33/masklat.tar", tr))
b = md5s("evalpack33/evalpack_crc33sec.tar", ev)
for c in CH + ["masklat"]:
    same = sum(a.get((c, t)) == b.get((c, t)) and a.get((c, t)) is not None for t in ids)
    print(f"{c:22s} train==eval {same}/{len(ids)}  (train has {sum((c,t) in a for t in ids)}, eval has {sum((c,t) in b for t in ids)})")
