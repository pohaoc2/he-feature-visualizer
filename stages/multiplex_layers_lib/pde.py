"""PDE solver and demand/source map helpers for multiplex layers."""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from utils.normalize import percentile_norm

log = logging.getLogger(__name__)


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


def _pde_matvec(
    u_flat: np.ndarray,
    H: int,
    W: int,
    diffusion: float,
    k_map: np.ndarray,
) -> np.ndarray:
    """Matrix-free matvec for A = (4D + k)*I - D*Laplacian_disc with Neumann BCs."""
    u = u_flat.reshape(H, W)
    padded = np.pad(u, 1, mode="edge")
    neigh = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    return ((4.0 * diffusion + k_map) * u - diffusion * neigh).ravel()


def solve_steady_state_diffusion(
    source_map: np.ndarray,
    consumption_map: np.ndarray,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    relaxation: float = 1.0,
    method: str = "jacobi",
) -> np.ndarray:
    """Solve D*Laplacian(u) - k(x)u + s(x) = 0 on a 2D grid.

    Parameters
    ----------
    method:
        ``'jacobi'`` — classic point-iterative method; converges in O(L²)
        iterations where L = sqrt(D/k_min).  Suitable for small patches.

        ``'cg'`` — conjugate gradient (matrix-free, Neumann BCs).  Converges
        in O(L) iterations — ~L× faster than Jacobi.  Use for WSI-scale grids
        where L can exceed 100 coarse pixels and Jacobi would need >10 000
        iterations.  ``max_iters`` becomes the CG iteration budget (default
        500 is usually ample; auto-set in :func:`solve_wsi_pde_map`).
    """
    if diffusion <= 0:
        raise ValueError("diffusion must be > 0")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    if tol <= 0:
        raise ValueError("tol must be > 0")
    if source_map.shape != consumption_map.shape:
        raise ValueError(
            f"source_map shape {source_map.shape} must match "
            f"consumption_map shape {consumption_map.shape}"
        )

    source = np.clip(source_map.astype(np.float64), 0.0, 1.0)
    k_map = np.clip(consumption_map.astype(np.float64), 0.0, None)
    H, W = source.shape
    N = H * W

    if method == "cg":
        A = LinearOperator(
            (N, N),
            matvec=lambda v: _pde_matvec(v, H, W, diffusion, k_map),
            dtype=np.float64,
        )
        # Warm-start: exp(-dist/L) is the exact solution for uniform k.
        # For spatially varying k(x), CG only corrects the residual → very few iters.
        from scipy.ndimage import distance_transform_edt

        vessel_bool = source > 0
        if vessel_bool.any():
            dist = distance_transform_edt(~vessel_bool)
            L = (diffusion / float(k_map.mean())) ** 0.5
            x0 = np.exp(-dist / L).ravel()
        else:
            x0 = np.zeros(N, dtype=np.float64)
        u_sol, info = cg(A, source.ravel(), x0=x0, rtol=tol, maxiter=max_iters)
        if info != 0:
            log.warning("CG did not converge (info=%d, max_iters=%d)", info, max_iters)
        return np.clip(u_sol, 0.0, None).reshape(H, W).astype(np.float32)

    # --- Jacobi (default, backward-compatible) ---
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")
    source32 = source.astype(np.float32)
    k32 = k_map.astype(np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    denom = (4.0 * diffusion + k32).astype(np.float32)

    for _ in range(max_iters):
        padded = np.pad(u, 1, mode="edge")
        neighbor_sum = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        )
        jacobi = np.clip((diffusion * neighbor_sum + source32) / denom, 0.0, 1.0)
        if relaxation < 1.0:
            u_next = ((1.0 - relaxation) * u + relaxation * jacobi).astype(np.float32)
        else:
            u_next = jacobi
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
