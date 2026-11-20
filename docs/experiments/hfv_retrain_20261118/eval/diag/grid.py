import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
R = Path(sys.argv[1]); ids = [l.strip() for l in open(R / "ids100.txt") if l.strip()]
pick = sorted(np.random.default_rng(42).choice(sorted(ids), 4, replace=False).tolist())
def ov(he, mask, col):
    a = np.asarray(he.convert("RGB")).copy(); m = np.asarray(Image.open(mask)) > 0
    from scipy.ndimage import binary_erosion
    edge = m & ~binary_erosion(m, iterations=1); a[edge] = col; return Image.fromarray(a)
def cv(img, js, col=(0, 255, 0)):
    im = Image.open(img).convert("RGB"); d = ImageDraw.Draw(im)
    for c in json.load(open(js))["cells"]:
        p = [tuple(x) for x in c["contour"]]
        if len(p) > 2: d.polygon(p, outline=col)
    return im
cols = [("real H&E", lambda t: Image.open(R / f"new33pack/orion-crc33sec/he/{t}.png").convert("RGB")),
        ("real + old mask", lambda t: ov(Image.open(R / f"new33pack/orion-crc33sec/he/{t}.png"), R / f"old33zip/orion-crc33/exp_channels/cell_masks/{t}.png", (255, 255, 0))),
        ("real + new mask", lambda t: ov(Image.open(R / f"new33pack/orion-crc33sec/he/{t}.png"), R / f"new33pack/orion-crc33sec/exp_channels/cell_masks/{t}.png", (0, 255, 255))),
        ("Vanilla (old mask)", lambda t: Image.open(R / f"oldtiles/a2_off_shelf/{t}.png").convert("RGB")),
        ("Vanilla (new mask)", lambda t: Image.open(R / f"new33/fig2_crc33sec/tiles/a2_off_shelf/{t}.png").convert("RGB")),
        ("old Grouped", lambda t: Image.open(R / f"oldtiles/production/{t}.png").convert("RGB")),
        ("new Grouped", lambda t: Image.open(R / f"new33/fig2_crc33sec/tiles/production/{t}.png").convert("RGB")),
        ("new Grouped + new mask", lambda t: ov(Image.open(R / f"new33/fig2_crc33sec/tiles/production/{t}.png"), R / f"new33pack/orion-crc33sec/exp_channels/cell_masks/{t}.png", (0, 255, 255))),
        ("new Concat", lambda t: Image.open(R / f"new33/fig2_crc33sec/tiles/a1_concat/{t}.png").convert("RGB")),
        ("new Per-channel", lambda t: Image.open(R / f"new33/fig2_crc33sec/tiles/a1_per_channel/{t}.png").convert("RGB"))]
S, H = 256, 22
G = Image.new("RGB", (S * len(cols), H + S * len(pick)), "white"); d = ImageDraw.Draw(G)
try: f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except Exception: f = None
for j, (name, fn) in enumerate(cols):
    d.text((j * S + 4, 3), name, fill="black", font=f)
    for i, t in enumerate(pick):
        G.paste(fn(t).resize((S, S)), (j * S, H + i * S))
for i, t in enumerate(pick): d.text((4, H + i * S + 4), t, fill="white", font=f)
G.save(R / "grid_crc33_4tiles.png"); print(pick, G.size)
