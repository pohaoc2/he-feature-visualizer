"""PDE solver and demand/source map helpers for multiplex layers."""

import numpy as np

from utils.normalize import percentile_norm


def build_vessel_source_map(
    vessel_mask: np.ndarray,
    vessel_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Build a normalized vessel source map in [0, 1] for PDE models."""
    source = vessel_mask.astype(np.float32)
    if vessel_weight is not None:
        if vessel_weight.shape != vessel_mask.shape:
            raise ValueError(
                f"vessel_weight shape {vessel_weight.shape} must match "
                f"vessel_mask shape {vessel_mask.shape}"
            )
        source = source * np.clip(vessel_weight.astype(np.float32), 0.0, 1.0)
    return np.clip(source, 0.0, 1.0).astype(np.float32)


def build_consumption_map(
    demand_map: np.ndarray,
    base_rate: float = 0.1,
    demand_weight: float = 0.3,
    immune_map: np.ndarray | None = None,
    immune_weight: float = 0.1,
) -> np.ndarray:
    """Build non-negative spatial consumption map k(x) from demand in [0,1].

    k(x) = base_rate + demand_weight * Ki67_norm + immune_weight * CD68_norm

    Following Kumar et al. (2024), proliferative (Ki67) and immune (CD68)
    demands are additive independent terms.  *immune_map* is optional; omit
    it when no macrophage channel is available.
    """
    if base_rate < 0:
        raise ValueError("base_rate must be >= 0")
    if demand_weight < 0:
        raise ValueError("demand_weight must be >= 0")
    if immune_weight < 0:
        raise ValueError("immune_weight must be >= 0")
    demand_01 = np.clip(demand_map.astype(np.float32), 0.0, 1.0)
    k = (base_rate + demand_weight * demand_01).astype(np.float32)
    if immune_map is not None:
        if immune_map.shape != demand_map.shape:
            raise ValueError(
                f"immune_map shape {immune_map.shape} must match "
                f"demand_map shape {demand_map.shape}"
            )
        immune_01 = np.clip(immune_map.astype(np.float32), 0.0, 1.0)
        k = (k + immune_weight * immune_01).astype(np.float32)
    return k


def solve_steady_state_diffusion(
    source_map: np.ndarray,
    consumption_map: np.ndarray,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    relaxation: float = 1.0,
) -> np.ndarray:
    """Solve D*Laplacian(u) - k(x)u + s(x) = 0 on a 2D grid."""
    if diffusion <= 0:
        raise ValueError("diffusion must be > 0")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")
    if source_map.shape != consumption_map.shape:
        raise ValueError(
            f"source_map shape {source_map.shape} must match "
            f"consumption_map shape {consumption_map.shape}"
        )

    source = np.clip(source_map.astype(np.float32), 0.0, 1.0)
    consumption = np.clip(consumption_map.astype(np.float32), 0.0, None)
    u = np.zeros_like(source, dtype=np.float32)
    denom = (4.0 * diffusion + consumption).astype(np.float32)

    for _ in range(max_iters):
        padded = np.pad(u, 1, mode="edge")
        neighbor_sum = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )

        jacobi = (diffusion * neighbor_sum + source) / denom
        jacobi = np.clip(jacobi, 0.0, 1.0)
        if relaxation < 1.0:
            u_next = ((1.0 - relaxation) * u + relaxation * jacobi).astype(np.float32)
        else:
            u_next = jacobi.astype(np.float32)

        delta = float(np.max(np.abs(u_next - u)))
        u = u_next
        if delta < tol:
            break

    return u


def compute_metabolic_demand_map(
    ki67: np.ndarray, pcna: np.ndarray | None = None
) -> np.ndarray:
    """Compute normalized proliferative demand map in [0, 1].

    If *pcna* is None (channel absent), demand is derived from Ki67 alone.
    """
    ki67_norm = percentile_norm(ki67.astype(np.float32))
    if pcna is None:
        return ki67_norm
    pcna_norm = percentile_norm(pcna.astype(np.float32))
    return np.maximum(ki67_norm, pcna_norm).astype(np.float32)
