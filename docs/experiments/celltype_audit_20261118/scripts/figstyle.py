"""Shared style of the report figures (user, 2026-11-11: like the storyline figure fig_sop_v2.py, Nature Communications
specs). Widths: 88 mm (one column) and 180 mm (two columns); text 5-8 pt in one sans-serif family; panel letters bold
lowercase; no titles that repeat the caption. Every report figure imports these constants instead of defining its own."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MM = 1 / 25.4
W1, W2 = 88 * MM, 180 * MM                       # one-column and two-column figure widths (in)
DPI = 300
FS_L, FS_T, FS, FS_K = 8.0, 7.0, 6.5, 5.8        # panel letter; panel heading; labels; ticks and notes (pt), as fig_sop_v2
LW_AX, LW, MS = .6, 1.0, 3.5                     # axes line; data line; marker size (pt)
INK, GREY, LIGHT = "#222222", "#7a7a7a", "#d9d9d9"
# One meaning per colour in every figure: the reference (real tissue / floor) is ink or grey; model arms below.
C = {"default": "#9a9a9a", "old": "#0b76a0", "turn": "#e97132", "new": "#4ea72e", "truth": "#222222",
     "shape": "#7570B3"}
FONT_DIR = Path.home() / "fonts" / "noto-sans-tc"


def _family():
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    for fam in ("Noto Sans TC", "Arial", "Helvetica", "DejaVu Sans"):
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            return fam
        except ValueError:
            continue
    return "DejaVu Sans"


FAMILY = _family()
plt.rcParams.update({
    "font.family": FAMILY, "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS_T, "xtick.labelsize": FS_K,
    "ytick.labelsize": FS_K, "legend.fontsize": FS_K, "legend.frameon": False, "axes.linewidth": LW_AX,
    "xtick.major.width": LW_AX, "ytick.major.width": LW_AX, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": LW, "lines.markersize": MS, "savefig.dpi": DPI, "figure.dpi": DPI, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": FAMILY, "mathtext.bf": f"{FAMILY}:bold"})


def letter(fig, x_in, y_in, s):
    """Bold lowercase panel letter with its top-left corner at (x_in, y_in) inches from the figure's top-left corner."""
    w, h = fig.get_size_inches()
    return fig.text(x_in / w, 1 - y_in / h, s, fontsize=FS_L, fontweight="bold", ha="left", va="top", color=INK)


def axes(fig, x_in, y_in, w_in, h_in):
    """Axes placed in inches from the figure's top-left corner, so panel sizes and gaps are exact."""
    w, h = fig.get_size_inches()
    return fig.add_axes([x_in / w, 1 - (y_in + h_in) / h, w_in / w, h_in / h])


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
