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


def make_oxygen_map(cd31_mask: np.ndarray) -> np.ndarray:
    """Oxygen proxy via distance transform with RdYlBu colormap."""
    dist = scipy.ndimage.distance_transform_edt(~cd31_mask)
    norm = percentile_norm(dist.astype(np.float32))
    inverted = (1.0 - norm).astype(np.float32)
    return apply_colormap(inverted, "RdYlBu")


def make_oxygen_map_pde(
    vessel_mask: np.ndarray,
    demand_map: np.ndarray,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    base_consumption: float = 0.1,
    demand_weight: float = 0.3,
) -> np.ndarray:
    """Oxygen proxy via steady-state diffusion-consumption PDE."""
    source_map = build_vessel_source_map(vessel_mask)
    consumption_map = build_consumption_map(
        demand_map,
        base_rate=base_consumption,
        demand_weight=demand_weight,
    )
    oxygen_density = solve_steady_state_diffusion(
        source_map=source_map,
        consumption_map=consumption_map,
        diffusion=diffusion,
        max_iters=max_iters,
        tol=tol,
    )
    return apply_colormap(oxygen_density, "RdYlBu")


def make_glucose_map(ki67: np.ndarray, pcna: np.ndarray) -> np.ndarray:
    """Metabolic demand proxy via max(norm(Ki67), norm(PCNA))."""
    metabolic = compute_metabolic_demand_map(ki67, pcna)
    return apply_colormap(metabolic, "hot")


def make_glucose_map_pde(
    vessel_mask: np.ndarray,
    demand_map: np.ndarray,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    base_consumption: float = 0.1,
    demand_weight: float = 0.3,
) -> np.ndarray:
    """Glucose proxy via steady-state diffusion-consumption PDE."""
    source_map = build_vessel_source_map(vessel_mask)
    consumption_map = build_consumption_map(
        demand_map,
        base_rate=base_consumption,
        demand_weight=demand_weight,
    )
    glucose_density = solve_steady_state_diffusion(
        source_map=source_map,
        consumption_map=consumption_map,
        diffusion=diffusion,
        max_iters=max_iters,
        tol=tol,
    )
    return apply_colormap(glucose_density, "hot")
