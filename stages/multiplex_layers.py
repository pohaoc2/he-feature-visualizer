"""
multiplex_layers.py — Stage 4+5 of the histopathology analysis pipeline.

Stage 4 (spatial analysis) reads per-patch multiplex immunofluorescence arrays
produced by Stage 1 (patchify.py) and derives three biologically meaningful
proxy maps from the raw channel intensities:

    vasculature
        Binary segmentation of CD31+ vascular structures.  CD31 (PECAM-1) is an
        endothelial marker; the binarised mask delineates vessel lumens and
        capillaries in the tissue.  Rendered as red pixels on a transparent
        background for overlay on the H&E image.

    oxygen
        A spatial proxy for tissue oxygenation derived from the Euclidean
        distance transform of the CD31 mask.  Pixels at or near vessel walls
        are assumed to be well-oxygenated (blue in the RdYlBu colormap), while
        pixels far from vessels are assumed to be hypoxic (red).  This
        approximation follows the classical Krogh cylinder model of oxygen
        diffusion.

    glucose / metabolic demand
        A proxy for local metabolic activity based on the maximum of
        normalised Ki67 and PCNA intensities.  Ki67 marks cells in active
        proliferation (G1-M phases) and PCNA is a DNA-replication clamp loader
        also elevated during S-phase; their maximum highlights the most
        metabolically demanding regions.  Rendered with the 'hot' colormap
        (black → red → yellow/white as demand increases).

Stage 5 (visualisation) saves the three RGBA overlay PNGs to sub-directories
of the output folder so they can be served directly by server_patches.py and
composited on top of H&E tiles in viewer_patches.html.

Input
-----
processed/multiplex/{i}_{j}.npy   — (C, H, W) uint16 arrays from Stage 1

Output
------
{out}/vasculature/{i}_{j}.png     — RGBA: red vessels on transparent background
{out}/vasculature_mask/{i}_{j}.npy — bool: binary vessel mask for downstream analysis
{out}/oxygen/{i}_{j}.png          — RGBA: RdYlBu oxygenation map
{out}/glucose/{i}_{j}.png         — RGBA: hot metabolic-demand map
"""

import argparse
import json
import logging
import pathlib

import cv2
import matplotlib
import matplotlib.cm
import numpy as np
import scipy.ndimage
from PIL import Image

from utils.channels import resolve_channel_indices
from utils.normalize import percentile_norm

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_multiplex_patch(npy_path: str) -> np.ndarray:
    """Load and return (C, H, W) uint16 array from .npy file."""
    return np.load(npy_path)


def get_channel_index(channel_names: list[str], target: str) -> int:
    """Return 0-based index of target in channel_names (case-insensitive).

    Raises ValueError with a helpful message if not found.
    """
    target_lower = target.lower()
    for idx, name in enumerate(channel_names):
        if name.lower() == target_lower:
            return idx
    raise ValueError(
        f"Channel '{target}' not found in channel list {channel_names}. "
        f"Available (case-insensitive): {channel_names}"
    )


def get_first_matching_channel_index(
    channel_names: list[str], candidates: list[str]
) -> int | None:
    """Return index of the first channel name found in *candidates*.

    Matching is case-insensitive. Returns ``None`` if no candidate is present.
    """
    for name in candidates:
        try:
            return get_channel_index(channel_names, name)
        except ValueError:
            continue
    return None


def extract_channel(patch: np.ndarray, idx: int) -> np.ndarray:
    """Return patch[idx] as (H, W) uint16 array."""
    return patch[idx]


def binarize_otsu(arr: np.ndarray) -> np.ndarray:
    """Otsu threshold on (H, W) float32 [0,1] → bool mask.

    Implementation:
      scaled = (arr * 255).clip(0, 255).astype(np.uint8)
      _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      return binary.astype(bool)
    """
    scaled = (arr * 255).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def apply_colormap(arr_01: np.ndarray, colormap: str) -> np.ndarray:
    """Apply a matplotlib named colormap to (H, W) float32 [0,1].

    Returns (H, W, 4) RGBA uint8.

    Implementation:
      cmap = matplotlib.cm.get_cmap(colormap)
      rgba_float = cmap(arr_01)           # (H, W, 4) float64 in [0,1]
      return (rgba_float * 255).clip(0, 255).astype(np.uint8)
    """
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        cmap = matplotlib.cm.get_cmap(colormap)  # type: ignore[attr-defined]
    rgba_float = cmap(arr_01)  # (H, W, 4) float64 in [0, 1]
    return (rgba_float * 255).clip(0, 255).astype(np.uint8)


def make_vasculature_overlay(
    cd31_mask: np.ndarray, color: tuple = (255, 60, 0, 200)
) -> np.ndarray:
    """Binary bool mask → (H, W, 4) RGBA uint8.

    True pixels → color tuple; False → (0, 0, 0, 0).
    """
    h, w = cd31_mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[cd31_mask] = color
    return out


def refine_vasculature_with_sma(
    cd31_mask: np.ndarray,
    sma_mask: np.ndarray,
    adjacency_px: int = 2,
) -> np.ndarray:
    """Refine a CD31 vessel mask with nearby SMA support.

    SMA signal is only accepted if it lies within ``adjacency_px`` of CD31
    pixels, which avoids global ``CD31 OR SMA`` overcalling in stroma.
    """
    if adjacency_px < 0:
        raise ValueError("adjacency_px must be >= 0")

    if not np.any(sma_mask):
        return cd31_mask.copy()

    k = 2 * adjacency_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near_cd31 = cv2.dilate(cd31_mask.astype(np.uint8), kernel, iterations=1).astype(
        bool
    )
    return cd31_mask | (sma_mask & near_cd31)


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
) -> np.ndarray:
    """Build non-negative spatial consumption map k(x) from demand in [0,1]."""
    if base_rate < 0:
        raise ValueError("base_rate must be >= 0")
    if demand_weight < 0:
        raise ValueError("demand_weight must be >= 0")
    demand_01 = np.clip(demand_map.astype(np.float32), 0.0, 1.0)
    return (base_rate + demand_weight * demand_01).astype(np.float32)


def solve_steady_state_diffusion(
    source_map: np.ndarray,
    consumption_map: np.ndarray,
    diffusion: float = 1.0,
    max_iters: int = 500,
    tol: float = 1e-4,
    relaxation: float = 1.0,
) -> np.ndarray:
    """Solve D*Laplacian(u) - k(x)u + s(x) = 0 on a 2D grid.

    Uses Jacobi relaxation with edge padding (zero-flux/Neumann-style boundary).
    Returns a bounded proxy field in [0, 1].
    """
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


def compute_metabolic_demand_map(ki67: np.ndarray, pcna: np.ndarray) -> np.ndarray:
    """Compute normalized proliferative demand map in [0, 1]."""
    ki67_norm = percentile_norm(ki67.astype(np.float32))
    pcna_norm = percentile_norm(pcna.astype(np.float32))
    return np.maximum(ki67_norm, pcna_norm).astype(np.float32)


def make_oxygen_map(cd31_mask: np.ndarray) -> np.ndarray:
    """Oxygen proxy via distance transform.

    Steps:
      1. dist = scipy.ndimage.distance_transform_edt(~cd31_mask)   # 0 at vessel
      2. norm = percentile_norm(dist.astype(np.float32))           # 0=vessel, 1=far
      3. inverted = 1.0 - norm    # invert so 1=near vessel (oxygenated), 0=far (hypoxic)
      4. apply_colormap(inverted, 'RdYlBu')
         — RdYlBu: 0→red (hypoxic), 0.5→yellow, 1→blue (oxygenated)

    Returns (H, W, 4) RGBA uint8.
    """
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
    """Metabolic demand proxy.

    Steps:
      1. ki67_norm = percentile_norm(ki67.astype(np.float32))
      2. pcna_norm = percentile_norm(pcna.astype(np.float32))
      3. metabolic = np.maximum(ki67_norm, pcna_norm)
      4. apply_colormap(metabolic, 'hot')
         — hot: 0→black, 0.5→red, 1→yellow/white

    Returns (H, W, 4) RGBA uint8.
    """
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description=(
            "Stage 4+5: Generate vasculature, oxygen, and glucose overlay PNGs "
            "from per-patch multiplex .npy files."
        )
    )
    parser.add_argument(
        "--multiplex-dir",
        required=True,
        help="Directory containing {i}_{j}.npy multiplex patch files.",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Path to processed/index.json (patch grid manifest).",
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Channel metadata CSV with 'Channel ID' and 'Target Name' columns.",
    )
    parser.add_argument(
        "--out",
        default="processed/",
        help="Output directory (default: processed/).",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=[
            "Keratin",
            "NaKATPase",
            "CDX2",
            "CD45",
            "CD3",
            "CD4",
            "CD8a",
            "CD20",
            "CD45RO",
            "CD68",
            "CD163",
            "FOXP3",
            "PD1",
            "aSMA",
            "CD31",
            "Desmin",
            "Collagen",
            "Ki67",
            "PCNA",
            "Vimentin",
            "Ecadherin",
        ],
        metavar="NAME",
        help="Multiplex channel names present in the .npy files, in order "
        "(must match --channels passed to patchify.py). "
        "Default: full Stage 3 marker panel, 21 channels.",
    )
    parser.add_argument(
        "--oxygen-model",
        choices=["distance", "pde"],
        default="distance",
        help=(
            "Oxygen proxy model. 'distance' is the current CD31 distance-transform "
            "model; 'pde' uses a steady-state diffusion-consumption solver."
        ),
    )
    parser.add_argument(
        "--glucose-model",
        choices=["max", "pde"],
        default="max",
        help=(
            "Glucose proxy model. 'max' uses max(norm(Ki67), norm(PCNA)); "
            "'pde' uses a steady-state diffusion-consumption solver."
        ),
    )
    parser.add_argument(
        "--pde-max-iters",
        type=int,
        default=500,
        help="Maximum PDE solver iterations.",
    )
    parser.add_argument(
        "--pde-tol",
        type=float,
        default=1e-4,
        help="PDE convergence tolerance (max absolute update).",
    )
    parser.add_argument(
        "--pde-diffusion",
        type=float,
        default=1.0,
        help="Diffusion coefficient used by PDE models.",
    )
    parser.add_argument(
        "--oxygen-consumption-base",
        type=float,
        default=0.1,
        help="Base oxygen consumption term used in oxygen PDE mode.",
    )
    parser.add_argument(
        "--oxygen-consumption-demand-weight",
        type=float,
        default=0.3,
        help="Demand weight for oxygen PDE consumption map.",
    )
    parser.add_argument(
        "--glucose-consumption-base",
        type=float,
        default=0.1,
        help="Base glucose consumption term used in glucose PDE mode.",
    )
    parser.add_argument(
        "--glucose-consumption-demand-weight",
        type=float,
        default=0.3,
        help="Demand weight for glucose PDE consumption map.",
    )
    parser.add_argument(
        "--vasc-sma-refine",
        action="store_true",
        help="Refine CD31 vessel mask by adding only SMA pixels adjacent to CD31.",
    )
    parser.add_argument(
        "--sma-channel-candidates",
        nargs="+",
        default=["aSMA", "Aortic smooth muscle actin"],
        metavar="NAME",
        help=(
            "Ordered list of channel names to try for SMA refinement. "
            "Used only with --vasc-sma-refine."
        ),
    )
    parser.add_argument(
        "--sma-adjacency-px",
        type=int,
        default=2,
        help=(
            "Pixel radius used to gate SMA support near CD31. "
            "Used only with --vasc-sma-refine."
        ),
    )
    args = parser.parse_args()

    multiplex_dir = pathlib.Path(args.multiplex_dir)
    index_path = pathlib.Path(args.index)
    out_dir = pathlib.Path(args.out)

    # ------------------------------------------------------------------
    # 1. Validate channel names against metadata CSV
    # ------------------------------------------------------------------
    log.info("Validating channel names against metadata CSV: %s", args.metadata_csv)
    resolve_channel_indices(args.metadata_csv, args.channels)
    log.info("  All requested channels found: %s", args.channels)

    # ------------------------------------------------------------------
    # 2. Resolve channel positions within the .npy arrays
    # ------------------------------------------------------------------
    cd31_idx = get_channel_index(args.channels, "CD31")
    ki67_idx = get_channel_index(args.channels, "Ki67")
    pcna_idx = get_channel_index(args.channels, "PCNA")
    sma_idx: int | None = None
    if args.vasc_sma_refine:
        sma_idx = get_first_matching_channel_index(
            args.channels, args.sma_channel_candidates
        )
    log.info(
        "  Channel indices — CD31: %d, Ki67: %d, PCNA: %d",
        cd31_idx,
        ki67_idx,
        pcna_idx,
    )
    if args.vasc_sma_refine and sma_idx is None:
        log.warning(
            "SMA refinement requested, but none of %s were found in --channels. "
            "Proceeding with CD31-only vessel masks.",
            args.sma_channel_candidates,
        )
    elif sma_idx is not None:
        log.info("  SMA refinement channel index: %d", sma_idx)

    # ------------------------------------------------------------------
    # 3. Load index.json, prepare output directories
    # ------------------------------------------------------------------
    log.info("Loading patch index: %s", index_path)
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    vasc_dir = out_dir / "vasculature"
    vasc_mask_dir = out_dir / "vasculature_mask"
    oxygen_dir = out_dir / "oxygen"
    glucose_dir = out_dir / "glucose"
    for d in (vasc_dir, vasc_mask_dir, oxygen_dir, glucose_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4–6. Process each patch
    # ------------------------------------------------------------------
    processed = 0
    skipped = 0

    for patch_meta in patches:
        x0 = patch_meta["x0"]
        y0 = patch_meta["y0"]
        patch_id = f"{x0}_{y0}"

        npy_path = multiplex_dir / f"{patch_id}.npy"
        if not npy_path.exists():
            log.warning("Missing multiplex file: %s — skipping.", npy_path)
            skipped += 1
            continue

        # a. Load (C, H, W) uint16 array
        patch = load_multiplex_patch(str(npy_path))

        # b. Extract individual channels
        cd31_raw = extract_channel(patch, cd31_idx)
        ki67_raw = extract_channel(patch, ki67_idx)
        pcna_raw = extract_channel(patch, pcna_idx)
        sma_raw = extract_channel(patch, sma_idx) if sma_idx is not None else None

        # c. Vasculature overlay
        cd31_norm = percentile_norm(cd31_raw.astype(np.float32))
        cd31_mask = binarize_otsu(cd31_norm)
        if sma_raw is not None:
            sma_norm = percentile_norm(sma_raw.astype(np.float32))
            sma_mask = binarize_otsu(sma_norm)
            cd31_mask = refine_vasculature_with_sma(
                cd31_mask,
                sma_mask,
                adjacency_px=args.sma_adjacency_px,
            )
        vasc_rgba = make_vasculature_overlay(cd31_mask)
        demand_map = compute_metabolic_demand_map(ki67_raw, pcna_raw)

        # d. Oxygen map
        if args.oxygen_model == "pde":
            oxygen_rgba = make_oxygen_map_pde(
                vessel_mask=cd31_mask,
                demand_map=demand_map,
                diffusion=args.pde_diffusion,
                max_iters=args.pde_max_iters,
                tol=args.pde_tol,
                base_consumption=args.oxygen_consumption_base,
                demand_weight=args.oxygen_consumption_demand_weight,
            )
        else:
            oxygen_rgba = make_oxygen_map(cd31_mask)

        # e. Glucose / metabolic demand map
        if args.glucose_model == "pde":
            glucose_rgba = make_glucose_map_pde(
                vessel_mask=cd31_mask,
                demand_map=demand_map,
                diffusion=args.pde_diffusion,
                max_iters=args.pde_max_iters,
                tol=args.pde_tol,
                base_consumption=args.glucose_consumption_base,
                demand_weight=args.glucose_consumption_demand_weight,
            )
        else:
            glucose_rgba = make_glucose_map(ki67_raw, pcna_raw)

        # f. Save outputs
        Image.fromarray(vasc_rgba, "RGBA").save(vasc_dir / f"{patch_id}.png")
        np.save(
            vasc_mask_dir / f"{patch_id}.npy",
            cd31_mask.astype(bool),
            allow_pickle=False,
        )
        Image.fromarray(oxygen_rgba, "RGBA").save(oxygen_dir / f"{patch_id}.png")
        Image.fromarray(glucose_rgba, "RGBA").save(glucose_dir / f"{patch_id}.png")

        processed += 1
        if processed % 50 == 0:
            log.info(
                "  Progress: %d patches processed, %d skipped …",
                processed,
                skipped,
            )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("Done.")
    log.info("  Patches processed : %d", processed)
    log.info("  Patches skipped   : %d", skipped)
    log.info("  Vasculature PNGs  → %s", vasc_dir)
    log.info("  Vasculature masks → %s", vasc_mask_dir)
    log.info("  Oxygen PNGs       → %s", oxygen_dir)
    log.info("  Glucose PNGs      → %s", glucose_dir)


if __name__ == "__main__":
    main()
