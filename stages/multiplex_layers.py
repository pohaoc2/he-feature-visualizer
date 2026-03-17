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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.ndimage
from PIL import Image

from stages.multiplex_layers_lib.channels import (
    extract_channel,
    get_channel_index,
    get_first_matching_channel_index,
    load_multiplex_patch,
)
from stages.multiplex_layers_lib.masks import (
    _validate_odd_kernel_size,
    apply_vessel_mask_quality_fallback,
    binarize_otsu,
    cleanup_vasculature_mask,
    refine_vasculature_with_sma,
)
from stages.multiplex_layers_lib.pde import (
    build_consumption_map,
    build_vessel_source_map,
    compute_metabolic_demand_map,
    solve_steady_state_diffusion,
)
from stages.multiplex_layers_lib.render import (
    apply_colormap,
    make_glucose_map,
    make_glucose_map_distance,
    make_oxygen_map,
    make_vasculature_overlay,
)
from stages.multiplex_layers_lib.wsi_pde import (
    extract_patch_from_coarse,
    read_wsi_channel_stack,
    solve_wsi_pde_map,
)
from utils.channels import resolve_channel_indices
from utils.normalize import percentile_norm

# ---------------------------------------------------------------------------
# Re-exported helper functions
# ---------------------------------------------------------------------------

# Helper implementations now live in stages/multiplex_layers_lib/*.py.
# They are imported above and remain available from this module for
# backward compatibility with existing tests and scripts.
__all__ = [
    "load_multiplex_patch",
    "get_channel_index",
    "get_first_matching_channel_index",
    "extract_channel",
    "binarize_otsu",
    "apply_colormap",
    "make_vasculature_overlay",
    "refine_vasculature_with_sma",
    "_validate_odd_kernel_size",
    "cleanup_vasculature_mask",
    "apply_vessel_mask_quality_fallback",
    "build_vessel_source_map",
    "build_consumption_map",
    "solve_steady_state_diffusion",
    "compute_metabolic_demand_map",
    "make_oxygen_map",
    "make_glucose_map",
    "make_glucose_map_distance",
    "read_wsi_channel_stack",
    "solve_wsi_pde_map",
    "extract_patch_from_coarse",
    "main",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parallel worker (module-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

_ML_CTX: dict = {}


def _init_ml_worker(ctx: dict) -> None:
    global _ML_CTX
    _ML_CTX = ctx
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _ml_patch_worker(patch_meta: dict) -> dict:
    ctx = _ML_CTX
    _log = logging.getLogger(__name__)

    multiplex_dir: pathlib.Path = ctx["multiplex_dir"]
    mpp: float = ctx["mpp"]
    he_mpp: float = ctx["he_mpp"]
    patch_size: int = ctx["patch_size"]
    warp_matrix = ctx["warp_matrix"]
    cd31_idx: int = ctx["cd31_idx"]
    ki67_idx: int = ctx["ki67_idx"]
    pcna_idx = ctx["pcna_idx"]
    cd68_idx = ctx["cd68_idx"]
    sma_idx = ctx["sma_idx"]
    oxygen_model: str = ctx["oxygen_model"]
    glucose_model: str = ctx["glucose_model"]
    oxygen_krogh_um: float = ctx["oxygen_krogh_um"]
    glucose_krogh_um: float = ctx["glucose_krogh_um"]
    sma_adjacency_px: int = ctx["sma_adjacency_px"]
    vasc_open_kernel_size: int = ctx["vasc_open_kernel_size"]
    vasc_close_kernel_size: int = ctx["vasc_close_kernel_size"]
    vasc_min_area: int = ctx["vasc_min_area"]
    vasc_noisy_max_fraction: float = ctx["vasc_noisy_max_fraction"]
    wsi_o2_coarse = ctx["wsi_o2_coarse"]
    wsi_glc_coarse = ctx["wsi_glc_coarse"]
    wsi_pde_ds: int = ctx["wsi_pde_ds"]
    vasc_dir: pathlib.Path = ctx["vasc_dir"]
    vasc_mask_dir: pathlib.Path = ctx["vasc_mask_dir"]
    oxygen_dir: pathlib.Path = ctx["oxygen_dir"]
    glucose_dir: pathlib.Path = ctx["glucose_dir"]
    validate_ki67_distance: bool = ctx["validate_ki67_distance"]
    validate_bin_um: float = ctx["validate_bin_um"]
    val_n_bins: int = ctx["val_n_bins"]

    x0 = patch_meta["x0"]
    y0 = patch_meta["y0"]
    patch_id = f"{x0}_{y0}"

    npy_path = multiplex_dir / f"{patch_id}.npy"
    if not npy_path.exists():
        _log.warning("Missing multiplex file: %s — skipping.", npy_path)
        return {"missing": True, "val_sum_patch": None, "val_count_patch": None}

    patch = load_multiplex_patch(str(npy_path))

    cd31_raw = extract_channel(patch, cd31_idx)
    ki67_raw = extract_channel(patch, ki67_idx)
    pcna_raw = extract_channel(patch, pcna_idx) if pcna_idx is not None else None
    cd68_raw = extract_channel(patch, cd68_idx) if cd68_idx is not None else None
    sma_raw = extract_channel(patch, sma_idx) if sma_idx is not None else None

    cd31_norm = percentile_norm(cd31_raw.astype(np.float32))
    cd31_base_mask = binarize_otsu(cd31_norm)
    vessel_mask = cd31_base_mask.copy()
    if sma_raw is not None:
        sma_norm = percentile_norm(sma_raw.astype(np.float32))
        sma_mask = binarize_otsu(sma_norm)
        vessel_mask = refine_vasculature_with_sma(
            vessel_mask, sma_mask, adjacency_px=sma_adjacency_px,
        )
    vessel_mask = cleanup_vasculature_mask(
        vessel_mask,
        open_kernel_size=vasc_open_kernel_size,
        close_kernel_size=vasc_close_kernel_size,
        min_area=vasc_min_area,
    )
    vessel_mask, vessel_status = apply_vessel_mask_quality_fallback(
        candidate_mask=vessel_mask,
        cd31_fallback_mask=cd31_base_mask,
        noisy_max_fraction=vasc_noisy_max_fraction,
    )
    if vessel_status == "empty_fallback":
        _log.warning(
            "Patch %s: empty vessel mask after refinement/cleanup; "
            "falling back to CD31-only mask.", patch_id,
        )
    elif vessel_status == "noisy_fallback":
        _log.warning(
            "Patch %s: vessel mask coverage exceeded noisy threshold "
            "(>= %.3f); falling back to CD31-only mask.",
            patch_id, vasc_noisy_max_fraction,
        )
    elif vessel_status == "empty":
        _log.warning(
            "Patch %s: vessel mask is empty; proceeding with deterministic "
            "empty mask.", patch_id,
        )

    vasc_rgba = make_vasculature_overlay(vessel_mask)
    demand_map = compute_metabolic_demand_map(ki67_raw, pcna_raw)
    immune_map = (
        percentile_norm(cd68_raw.astype(np.float32)) if cd68_raw is not None else None
    )

    if oxygen_model == "wsi-pde":
        assert wsi_o2_coarse is not None
        o2_patch = extract_patch_from_coarse(
            wsi_o2_coarse, x0, y0, patch_size, warp_matrix, he_mpp, mpp, wsi_pde_ds,
        )
        oxygen_rgba = apply_colormap(o2_patch, "RdYlBu")
    else:
        oxygen_rgba = make_oxygen_map(vessel_mask, mpp=mpp, max_dist_um=oxygen_krogh_um)

    if glucose_model == "wsi-pde":
        assert wsi_glc_coarse is not None
        glc_patch = extract_patch_from_coarse(
            wsi_glc_coarse, x0, y0, patch_size, warp_matrix, he_mpp, mpp, wsi_pde_ds,
        )
        glucose_rgba = apply_colormap(glc_patch, "hot")
    elif glucose_model == "distance":
        glucose_rgba = make_glucose_map_distance(
            vessel_mask, mpp=mpp, max_dist_um=glucose_krogh_um,
        )
    else:
        glucose_rgba = make_glucose_map(ki67_raw, pcna_raw)

    val_sum_patch: np.ndarray | None = None
    val_count_patch: np.ndarray | None = None
    if validate_ki67_distance and val_n_bins > 0:
        dist_px = scipy.ndimage.distance_transform_edt(~vessel_mask)
        dist_um = dist_px * mpp
        ki67_norm_val = percentile_norm(ki67_raw.astype(np.float32))
        bin_idx = np.clip(
            (dist_um / validate_bin_um).astype(np.int64), 0, val_n_bins - 1,
        ).ravel()
        ki67_flat = ki67_norm_val.ravel().astype(np.float64)
        val_sum_patch = np.bincount(bin_idx, weights=ki67_flat, minlength=val_n_bins)
        val_count_patch = np.bincount(bin_idx, minlength=val_n_bins)

    Image.fromarray(vasc_rgba, "RGBA").save(vasc_dir / f"{patch_id}.png")
    np.save(vasc_mask_dir / f"{patch_id}.npy", vessel_mask.astype(bool), allow_pickle=False)
    Image.fromarray(oxygen_rgba, "RGBA").save(oxygen_dir / f"{patch_id}.png")
    Image.fromarray(glucose_rgba, "RGBA").save(glucose_dir / f"{patch_id}.png")

    return {"missing": False, "val_sum_patch": val_sum_patch, "val_count_patch": val_count_patch}


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
            "Hoechst",
            "AF1",
            "CD31",
            "CD45",
            "CD68",
            "Argo550",
            "CD4",
            "FOXP3",
            "CD8a",
            "CD45RO",
            "CD20",
            "PD-L1",
            "CD3e",
            "CD163",
            "E-cadherin",
            "PD-1",
            "Ki67",
            "Pan-CK",
            "SMA",
        ],
        metavar="NAME",
        help="Multiplex channel names present in the .npy files, in order "
        "(must match --channels passed to patchify.py). "
        "Default: 19-channel panel from data/markers.csv.",
    )
    parser.add_argument(
        "--multiplex-tiff",
        default=None,
        help=(
            "Path to the multiplex OME-TIFF (required when --oxygen-model wsi-pde "
            "or --glucose-model wsi-pde)."
        ),
    )
    parser.add_argument(
        "--wsi-pde-ds",
        type=int,
        default=4,
        help=(
            "Downsampling factor for WSI-scale PDE (default: 4). "
            "At ds=4 and mpp=0.325 the coarse grid spacing is 1.3 µm/px and "
            "the O2 decay length is ~123 coarse pixels, requiring ~15000 Jacobi "
            "iterations to converge."
        ),
    )
    parser.add_argument(
        "--wsi-pde-max-iters",
        type=int,
        default=20000,
        help="Maximum Jacobi iterations for the WSI-scale PDE solver (default: 20000).",
    )
    parser.add_argument(
        "--wsi-pde-tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for the WSI-scale PDE solver (default: 1e-4).",
    )
    parser.add_argument(
        "--oxygen-model",
        choices=["distance", "wsi-pde"],
        default="distance",
        help=(
            "Oxygen proxy model. 'distance' uses a physically clamped CD31 "
            "distance transform (Grimes 2014, Zaidi 2019); 'wsi-pde' runs a "
            "steady-state diffusion-consumption PDE on the full WSI at coarse "
            "resolution and extracts per-patch slices (Kumar 2024, physically "
            "correct for Krogh radii > patch size)."
        ),
    )
    parser.add_argument(
        "--glucose-model",
        choices=["max", "distance", "wsi-pde"],
        default="max",
        help=(
            "Glucose proxy model. 'max' uses max(norm(Ki67), norm(PCNA)); "
            "'distance' uses a physically clamped CD31 distance transform "
            "(default clamp 450 µm, Grimes 2014); "
            "'wsi-pde' runs WSI-scale PDE (requires --multiplex-tiff)."
        ),
    )
    parser.add_argument(
        "--cd68-consumption-weight",
        type=float,
        default=0.1,
        help=(
            "Weight for CD68 macrophage demand in PDE consumption map k(x) "
            "(Kumar et al. 2024). Only used with --oxygen-model wsi-pde or "
            "--glucose-model wsi-pde."
        ),
    )
    parser.add_argument(
        "--validate-ki67-distance",
        action="store_true",
        help=(
            "Accumulate global Ki67-vs-vessel-distance statistics across all "
            "patches and save to {out}/validation/ki67_vs_distance.csv "
            "(Zaidi et al. 2019 validation approach)."
        ),
    )
    parser.add_argument(
        "--validate-bin-um",
        type=float,
        default=10.0,
        help="Bin width in µm for Ki67-vs-distance curve (default: 10 µm).",
    )
    parser.add_argument(
        "--oxygen-krogh-um",
        type=float,
        default=160.0,
        help=(
            "Krogh cylinder O2 diffusion radius in µm. Used by both the "
            "'distance' model (distance clamp) and 'wsi-pde' model (D calibration). "
            "Default: 160 µm (Grimes 2014)."
        ),
    )
    parser.add_argument(
        "--glucose-krogh-um",
        type=float,
        default=450.0,
        help=(
            "Krogh cylinder glucose diffusion radius in µm. Used by both the "
            "'distance' model (distance clamp) and 'wsi-pde' model (D calibration). "
            "Default: 450 µm (Grimes 2014)."
        ),
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
        default=["SMA", "aSMA", "Aortic smooth muscle actin"],
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
    parser.add_argument(
        "--vasc-open-kernel-size",
        type=int,
        default=0,
        help=(
            "Optional odd kernel size for vessel-mask morphological opening "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--vasc-close-kernel-size",
        type=int,
        default=0,
        help=(
            "Optional odd kernel size for vessel-mask morphological closing "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--vasc-min-area",
        type=int,
        default=0,
        help=(
            "Optional minimum connected-component area for vessel masks "
            "(0 disables filtering)."
        ),
    )
    parser.add_argument(
        "--vasc-noisy-max-fraction",
        type=float,
        default=0.98,
        help=(
            "If vessel-mask coverage is >= this fraction, treat as noisy and "
            "fallback to CD31-only mask."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for patch processing (default: 4).",
    )
    args = parser.parse_args()

    # Validate vessel-mask cleanup options once at startup.
    _validate_odd_kernel_size("vasc_open_kernel_size", args.vasc_open_kernel_size)
    _validate_odd_kernel_size("vasc_close_kernel_size", args.vasc_close_kernel_size)
    if args.vasc_min_area < 0:
        parser.error("--vasc-min-area must be >= 0")
    if not (0.0 < args.vasc_noisy_max_fraction <= 1.0):
        parser.error("--vasc-noisy-max-fraction must be in (0, 1]")

    multiplex_dir = pathlib.Path(args.multiplex_dir)
    index_path = pathlib.Path(args.index)
    out_dir = pathlib.Path(args.out)

    needs_wsi_pde = args.oxygen_model == "wsi-pde" or args.glucose_model == "wsi-pde"
    if needs_wsi_pde and not args.multiplex_tiff:
        parser.error(
            "--multiplex-tiff is required when --oxygen-model wsi-pde or "
            "--glucose-model wsi-pde."
        )

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
    pcna_idx = get_first_matching_channel_index(args.channels, ["PCNA"])
    cd68_idx = get_first_matching_channel_index(args.channels, ["CD68"])
    sma_idx: int | None = None
    if args.vasc_sma_refine:
        sma_idx = get_first_matching_channel_index(
            args.channels, args.sma_channel_candidates
        )
    log.info(
        "  Channel indices — CD31: %d, Ki67: %d, PCNA: %s, CD68: %s",
        cd31_idx,
        ki67_idx,
        pcna_idx if pcna_idx is not None else "absent (Ki67-only demand)",
        cd68_idx if cd68_idx is not None else "absent (no immune consumption term)",
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
    # 3. Load index.json, read mx_mpp, prepare output directories
    # ------------------------------------------------------------------
    log.info("Loading patch index: %s", index_path)
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    mpp: float = index.get("mx_mpp") or index.get("he_mpp") or 1.0
    log.info("  Microns per pixel (mpp): %.4f", mpp)
    if mpp == 1.0 and "mx_mpp" not in index and "he_mpp" not in index:
        log.warning(
            "  'mx_mpp'/'he_mpp' not found in index.json — defaulting mpp=1.0. "
            "Distance-model clamps will be in pixels, not microns."
        )

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    he_mpp: float = index.get("he_mpp") or mpp
    warp_matrix: list = index.get("warp_matrix", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    patch_size: int = index.get("patch_size", 256)

    vasc_dir = out_dir / "vasculature"
    vasc_mask_dir = out_dir / "vasculature_mask"
    oxygen_dir = out_dir / "oxygen"
    glucose_dir = out_dir / "glucose"
    for d in (vasc_dir, vasc_mask_dir, oxygen_dir, glucose_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3b. WSI-scale PDE pre-computation (runs once before per-patch loop)
    # ------------------------------------------------------------------
    wsi_o2_coarse: np.ndarray | None = None
    wsi_glc_coarse: np.ndarray | None = None

    if needs_wsi_pde:
        # Read only the channels needed by the PDE solver to keep memory manageable.
        # On a large WSI, reading all 19 channels at even ds=4 can exceed 9 GB.
        _wsi_needed = [cd31_idx, ki67_idx]
        if cd68_idx is not None:
            _wsi_needed.append(cd68_idx)
        if sma_idx is not None:
            _wsi_needed.append(sma_idx)
        if pcna_idx is not None:
            _wsi_needed.append(pcna_idx)
        _wsi_needed_unique = list(dict.fromkeys(_wsi_needed))  # deduplicate, preserve order
        _remap = {orig: new for new, orig in enumerate(_wsi_needed_unique)}

        log.info(
            "Reading WSI channels %s at ds=%d from %s …",
            _wsi_needed_unique, args.wsi_pde_ds, args.multiplex_tiff,
        )
        channel_stack = read_wsi_channel_stack(
            args.multiplex_tiff, args.wsi_pde_ds, channel_indices=_wsi_needed_unique
        )
        log.info("  Channel stack shape: %s (%.0f MB)",
                 channel_stack.shape, channel_stack.nbytes / 1e6)

        # Remapped indices into the reduced channel_stack
        wsi_cd31_idx  = _remap[cd31_idx]
        wsi_ki67_idx  = _remap[ki67_idx]
        wsi_cd68_idx  = _remap.get(cd68_idx)
        wsi_sma_idx   = _remap.get(sma_idx)
        wsi_pcna_idx  = _remap.get(pcna_idx)

        if args.oxygen_model == "wsi-pde":
            mpp_coarse = mpp * args.wsi_pde_ds
            L_coarse = args.oxygen_krogh_um / mpp_coarse
            D_coarse = L_coarse**2 * args.oxygen_consumption_base
            log.info(
                "  O2 WSI-PDE: Krogh=%.0f µm, mpp_coarse=%.3f, L_coarse=%.1f px, "
                "D_coarse=%.1f, max_iters=%d",
                args.oxygen_krogh_um, mpp_coarse, L_coarse, D_coarse, args.wsi_pde_max_iters,
            )
            wsi_o2_coarse = solve_wsi_pde_map(
                channel_stack=channel_stack,
                cd31_idx=wsi_cd31_idx,
                ki67_idx=wsi_ki67_idx,
                mpp=mpp,
                ds=args.wsi_pde_ds,
                krogh_um=args.oxygen_krogh_um,
                k_base=args.oxygen_consumption_base,
                demand_weight=args.oxygen_consumption_demand_weight,
                immune_weight=args.cd68_consumption_weight,
                sma_idx=wsi_sma_idx,
                sma_adjacency_px=args.sma_adjacency_px,
                cd68_idx=wsi_cd68_idx,
                pcna_idx=wsi_pcna_idx,
                max_iters=args.wsi_pde_max_iters,
                tol=args.wsi_pde_tol,
            )
            log.info("  O2 WSI-PDE solved. Grid: %s, range [%.3f, %.3f]",
                     wsi_o2_coarse.shape, float(wsi_o2_coarse.min()), float(wsi_o2_coarse.max()))

        if args.glucose_model == "wsi-pde":
            mpp_coarse = mpp * args.wsi_pde_ds
            L_coarse = args.glucose_krogh_um / mpp_coarse
            D_coarse = L_coarse**2 * args.glucose_consumption_base
            log.info(
                "  Glucose WSI-PDE: Krogh=%.0f µm, mpp_coarse=%.3f, L_coarse=%.1f px, "
                "D_coarse=%.1f, max_iters=%d",
                args.glucose_krogh_um, mpp_coarse, L_coarse, D_coarse, args.wsi_pde_max_iters,
            )
            wsi_glc_coarse = solve_wsi_pde_map(
                channel_stack=channel_stack,
                cd31_idx=wsi_cd31_idx,
                ki67_idx=wsi_ki67_idx,
                mpp=mpp,
                ds=args.wsi_pde_ds,
                krogh_um=args.glucose_krogh_um,
                k_base=args.glucose_consumption_base,
                demand_weight=args.glucose_consumption_demand_weight,
                immune_weight=args.cd68_consumption_weight,
                sma_idx=wsi_sma_idx,
                sma_adjacency_px=args.sma_adjacency_px,
                cd68_idx=wsi_cd68_idx,
                pcna_idx=wsi_pcna_idx,
                max_iters=args.wsi_pde_max_iters,
                tol=args.wsi_pde_tol,
            )
            log.info("  Glucose WSI-PDE solved. Grid: %s, range [%.3f, %.3f]",
                     wsi_glc_coarse.shape, float(wsi_glc_coarse.min()), float(wsi_glc_coarse.max()))

    # Ki67-vs-distance validation accumulators (Zaidi et al. 2019)
    val_n_bins: int = 0
    val_sum: np.ndarray | None = None
    val_count: np.ndarray | None = None
    if args.validate_ki67_distance:
        if args.validate_bin_um <= 0:
            parser.error("--validate-bin-um must be > 0")
        val_n_bins = int(np.ceil(args.oxygen_krogh_um / args.validate_bin_um)) + 1
        val_sum = np.zeros(val_n_bins, dtype=np.float64)
        val_count = np.zeros(val_n_bins, dtype=np.int64)
        (out_dir / "validation").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4–6. Process each patch
    # ------------------------------------------------------------------
    processed = 0
    skipped = 0

    ctx = {
        "multiplex_dir": multiplex_dir,
        "mpp": mpp,
        "he_mpp": he_mpp,
        "patch_size": patch_size,
        "warp_matrix": warp_matrix,
        "cd31_idx": cd31_idx,
        "ki67_idx": ki67_idx,
        "pcna_idx": pcna_idx,
        "cd68_idx": cd68_idx,
        "sma_idx": sma_idx,
        "oxygen_model": args.oxygen_model,
        "glucose_model": args.glucose_model,
        "oxygen_krogh_um": args.oxygen_krogh_um,
        "glucose_krogh_um": args.glucose_krogh_um,
        "sma_adjacency_px": args.sma_adjacency_px,
        "vasc_open_kernel_size": args.vasc_open_kernel_size,
        "vasc_close_kernel_size": args.vasc_close_kernel_size,
        "vasc_min_area": args.vasc_min_area,
        "vasc_noisy_max_fraction": args.vasc_noisy_max_fraction,
        "wsi_o2_coarse": wsi_o2_coarse,
        "wsi_glc_coarse": wsi_glc_coarse,
        "wsi_pde_ds": args.wsi_pde_ds,
        "vasc_dir": vasc_dir,
        "vasc_mask_dir": vasc_mask_dir,
        "oxygen_dir": oxygen_dir,
        "glucose_dir": glucose_dir,
        "validate_ki67_distance": args.validate_ki67_distance,
        "validate_bin_um": args.validate_bin_um,
        "val_n_bins": val_n_bins,
    }
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_ml_worker,
        initargs=(ctx,),
    ) as executor:
        for result in executor.map(_ml_patch_worker, patches):
            if result["missing"]:
                skipped += 1
                continue
            if val_sum is not None and result["val_sum_patch"] is not None:
                val_sum += result["val_sum_patch"]
            if val_count is not None and result["val_count_patch"] is not None:
                val_count += result["val_count_patch"]
            processed += 1
            if processed % 50 == 0:
                log.info(
                    "  Progress: %d patches processed, %d skipped …",
                    processed,
                    skipped,
                )

    # ------------------------------------------------------------------
    # 7. Ki67-vs-distance validation CSV
    # ------------------------------------------------------------------
    if args.validate_ki67_distance and val_sum is not None and val_count is not None:
        bin_centers_um = (np.arange(val_n_bins) + 0.5) * args.validate_bin_um
        mean_ki67 = np.where(val_count > 0, val_sum / val_count, np.nan)
        val_df = pd.DataFrame(
            {
                "distance_um": bin_centers_um,
                "ki67_mean": mean_ki67,
                "pixel_count": val_count,
            }
        )
        val_csv = out_dir / "validation" / "ki67_vs_distance.csv"
        val_df.to_csv(val_csv, index=False)
        log.info("  Ki67-vs-distance CSV → %s", val_csv)

    # ------------------------------------------------------------------
    # 8. Summary
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
