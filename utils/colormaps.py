"""Shared colour maps for H&E + multiplex visualisation tools."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# Hoechst 33342 fluorescence look: black background → electric blue → blue-white peak
HOECHST_CMAP = LinearSegmentedColormap.from_list(
    "hoechst33342",
    [
        (0.00, (0.00, 0.00, 0.00)),  # black background
        (0.35, (0.04, 0.10, 0.55)),  # deep blue
        (0.65, (0.10, 0.35, 0.90)),  # electric blue
        (0.85, (0.35, 0.65, 1.00)),  # bright blue-cyan
        (1.00, (0.80, 0.93, 1.00)),  # near-white peak
    ],
)

# O₂ / glucose proxy RGBA overlays use fixed palettes; colorbars use these gradients.
OXYGEN_PROXY_CMAP = LinearSegmentedColormap.from_list(
    "oxygen_proxy",
    [
        (0.0, (0.0, 0.0, 0.0)),  # hypoxic — black
        (1.0, (0.0, 1.0, 1.0)),  # oxygenated — cyan
    ],
)

GLUCOSE_PROXY_CMAP = LinearSegmentedColormap.from_list(
    "glucose_proxy",
    [
        (0.0, (0.0, 0.0, 0.0)),  # depleted — black
        (1.0, (1.0, 0.95, 0.12)),  # high — bright yellow
    ],
)

# ── Cell colormaps ─────────────────────────────────────────────────────────────

# RGBA (R, G, B, A) colour tuples shared by assign_cells, vis tools, and tests.
CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "cancer": (220, 50, 50, 200),
    "immune": (50, 100, 220, 200),
    "healthy": (50, 180, 50, 200),
    "other": (150, 150, 150, 120),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (230, 50, 180, 200),  # magenta
    "nonprolif": (240, 140, 30, 200),       # amber
    "dead": (110, 40, 160, 200),            # purple
    "other": (160, 160, 160, 120),
}

# Maps canonical color-dict key → subdirectory name used by assign_cells.py output.
# Used by vis_random_patches.py for exact directory matching (avoids substring ambiguity).
CELL_TYPE_DIR: dict[str, str] = {
    "cancer": "cell_type_cancer",
    "immune": "cell_type_immune",
    "healthy": "cell_type_healthy",
}

CELL_STATE_DIR: dict[str, str] = {
    "proliferative": "cell_state_prolif",
    "nonprolif": "cell_state_nonprolif",
    "dead": "cell_state_dead",
}
