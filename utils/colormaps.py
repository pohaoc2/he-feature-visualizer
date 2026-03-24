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
