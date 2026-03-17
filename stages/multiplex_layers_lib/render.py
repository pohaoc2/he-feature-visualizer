"""Rendering and map-construction helpers for multiplex layers."""

import matplotlib
import matplotlib.cm
import numpy as np
import scipy.ndimage

from utils.normalize import percentile_norm

from .pde import (
    build_consumption_map,
    build_vessel_source_map,
    compute_metabolic_demand_map,
    solve_steady_state_diffusion,
)


def apply_colormap(arr_01: np.ndarray, colormap: str) -> np.ndarray:
    """Apply a matplotlib named colormap to (H, W) float32 [0,1]."""
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = matplotlib.cm.get_cmap(colormap)  # type: ignore[attr-defined]
    rgba_float = cmap(arr_01)
    return (rgba_float * 255).clip(0, 255).astype(np.uint8)


def make_vasculature_overlay(
    cd31_mask: np.ndarray, color: tuple = (255, 60, 0, 200)
) -> np.ndarray:
    """Binary bool mask -> (H, W, 4) RGBA uint8."""
    h, w = cd31_mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[cd31_mask] = color
    return out


def _clamped_distance_map(
    mask: np.ndarray,
    mpp: float,
    max_dist_um: float,
) -> np.ndarray:
    """Euclidean distance from *mask* clamped at *max_dist_um*, in [0, 1].

    Implements the physically-grounded normalization from Zaidi et al. (2019)
    and Grimes et al. (2014): pixels beyond the diffusion limit are all set to
    1.0 (fully depleted), making values cross-patch comparable.
    """
    dist_px = scipy.ndimage.distance_transform_edt(~mask)
    max_dist_px = max_dist_um / mpp
    return np.clip(dist_px / max_dist_px, 0.0, 1.0).astype(np.float32)


def make_oxygen_map(
    cd31_mask: np.ndarray,
    mpp: float = 1.0,
    max_dist_um: float = 160.0,
) -> np.ndarray:
    """Oxygen proxy via physically clamped distance transform.

    Clamp at *max_dist_um* (default 160 µm, Grimes 2014 / Zaidi 2019).
    RdYlBu colormap: blue = near vessel (oxygenated), red = beyond clamp (hypoxic).
    """
    norm = _clamped_distance_map(cd31_mask, mpp, max_dist_um)
    inverted = (1.0 - norm).astype(np.float32)
    return apply_colormap(inverted, "RdYlBu")


def make_oxygen_map_pde(
    vessel_mask: np.ndarray,
    demand_map: np.ndarray,
    immune_map: np.ndarray | None = None,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    base_consumption: float = 0.1,
    demand_weight: float = 0.3,
    immune_weight: float = 0.1,
) -> np.ndarray:
    """Oxygen proxy via steady-state diffusion-consumption PDE.

    *immune_map* (e.g. normalised CD68) adds a macrophage consumption term
    following Kumar et al. (2024).
    """
    source_map = build_vessel_source_map(vessel_mask)
    consumption_map = build_consumption_map(
        demand_map,
        base_rate=base_consumption,
        demand_weight=demand_weight,
        immune_map=immune_map,
        immune_weight=immune_weight,
    )
    oxygen_density = solve_steady_state_diffusion(
        source_map=source_map,
        consumption_map=consumption_map,
        diffusion=diffusion,
        max_iters=max_iters,
        tol=tol,
    )
    return apply_colormap(oxygen_density, "RdYlBu")


def make_glucose_map(ki67: np.ndarray, pcna: np.ndarray | None = None) -> np.ndarray:
    """Metabolic demand proxy via max(norm(Ki67), norm(PCNA)).

    If *pcna* is None, Ki67 alone is used.
    """
    metabolic = compute_metabolic_demand_map(ki67, pcna)
    return apply_colormap(metabolic, "hot")


def make_glucose_map_distance(
    cd31_mask: np.ndarray,
    mpp: float = 1.0,
    max_dist_um: float = 450.0,
) -> np.ndarray:
    """Glucose proxy via physically clamped distance transform.

    Clamp at *max_dist_um* (default 450 µm, Grimes 2014).  Glucose diffuses
    further than O2 (higher plasma concentration compensates for lower D),
    so the supply zone is ~2.8× wider than the oxygen zone.
    Hot colormap: black/red = near vessel (glucose available), white = depleted.
    """
    norm = _clamped_distance_map(cd31_mask, mpp, max_dist_um)
    inverted = (1.0 - norm).astype(np.float32)
    return apply_colormap(inverted, "hot")


def make_glucose_map_pde(
    vessel_mask: np.ndarray,
    demand_map: np.ndarray,
    immune_map: np.ndarray | None = None,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    base_consumption: float = 0.1,
    demand_weight: float = 0.3,
    immune_weight: float = 0.1,
) -> np.ndarray:
    """Glucose proxy via steady-state diffusion-consumption PDE.

    *immune_map* (e.g. normalised CD68) adds a macrophage consumption term
    following Kumar et al. (2024).
    """
    source_map = build_vessel_source_map(vessel_mask)
    consumption_map = build_consumption_map(
        demand_map,
        base_rate=base_consumption,
        demand_weight=demand_weight,
        immune_map=immune_map,
        immune_weight=immune_weight,
    )
    glucose_density = solve_steady_state_diffusion(
        source_map=source_map,
        consumption_map=consumption_map,
        diffusion=diffusion,
        max_iters=max_iters,
        tol=tol,
    )
    return apply_colormap(glucose_density, "hot")
