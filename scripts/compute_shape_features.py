"""Compute per-cell morphology features from binary patch masks."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


SHAPE_COLUMNS = ["CellID", "PatchID", "area_px", "perimeter_px", "circularity"]


@dataclass(frozen=True)
class MaskRegion:
    """Minimal region representation for one connected component."""

    label: int
    area: float
    perimeter: float


def load_mask_regions(mask_path: Path) -> list:
    """Load a binary mask PNG and return connected-component regions."""
    labeled = _load_labeled_mask(mask_path)
    return _regions_from_labeled_mask(labeled)


def compute_circularity(region) -> float:
    """Return circularity = 4*pi*area/perimeter^2, guarding zero perimeter."""
    perimeter = float(region.perimeter)
    if perimeter == 0.0:
        return 0.0
    return 4.0 * math.pi * float(region.area) / (perimeter**2)


def match_cells_to_regions(
    cells_df: pd.DataFrame,
    labeled_mask: np.ndarray,
    regions_by_label: dict[int, object],
) -> pd.DataFrame:
    """Match cells to instance regions via centroid lookup in patch coordinates."""
    rows: list[dict[str, object]] = []
    height, width = labeled_mask.shape

    for _, cell in cells_df.iterrows():
        row_idx = int(round(float(cell["centroid_y_local"])))
        col_idx = int(round(float(cell["centroid_x_local"])))
        base = {"CellID": cell["CellID"], "PatchID": cell["PatchID"]}

        if not (0 <= row_idx < height and 0 <= col_idx < width):
            rows.append(
                {
                    **base,
                    "area_px": np.nan,
                    "perimeter_px": np.nan,
                    "circularity": np.nan,
                }
            )
            continue

        region_label = int(labeled_mask[row_idx, col_idx])
        region = regions_by_label.get(region_label)
        if region is None:
            rows.append(
                {
                    **base,
                    "area_px": np.nan,
                    "perimeter_px": np.nan,
                    "circularity": np.nan,
                }
            )
            continue

        rows.append(
            {
                **base,
                "area_px": float(region.area),
                "perimeter_px": float(region.perimeter),
                "circularity": compute_circularity(region),
            }
        )

    return pd.DataFrame(rows, columns=SHAPE_COLUMNS)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Compute per-cell shape features from binary patch masks."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing cell_assignments.csv and cell_masks/*.png",
    )
    return parser.parse_args()


def _load_labeled_mask(mask_path: Path) -> np.ndarray:
    """Load one patch mask and return a labeled instance mask."""
    image = np.asarray(Image.open(mask_path))
    if image.ndim == 3:
        image = image[..., 0]
    _, labeled = cv2.connectedComponents((image > 128).astype(np.uint8))
    return labeled.astype(np.int32)


def _regions_from_labeled_mask(labeled_mask: np.ndarray) -> list[MaskRegion]:
    """Compute area/perimeter summaries for each non-background label."""
    regions: list[MaskRegion] = []
    labels = np.unique(labeled_mask)
    for label_value in labels:
        if label_value == 0:
            continue
        mask = (labeled_mask == label_value).astype(np.uint8)
        area = float(mask.sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        regions.append(
            MaskRegion(label=int(label_value), area=area, perimeter=perimeter)
        )
    return regions


def _empty_shape_rows(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Return one unmatched shape row per input cell."""
    out = cells_df.loc[:, ["CellID", "PatchID"]].copy()
    out["area_px"] = np.nan
    out["perimeter_px"] = np.nan
    out["circularity"] = np.nan
    return out.loc[:, SHAPE_COLUMNS]


def compute_shape_features(data_dir: Path) -> pd.DataFrame:
    """Compute shape features for every row in cell_assignments.csv."""
    assignments_path = data_dir / "cell_assignments.csv"
    mask_dir = data_dir / "cell_masks"
    assignments = pd.read_csv(assignments_path)

    required = {"CellID", "PatchID", "centroid_x_local", "centroid_y_local"}
    missing = required.difference(assignments.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"cell_assignments.csv missing required columns: {missing_cols}")

    assignments = assignments.dropna(subset=["PatchID"]).copy()
    assignments["PatchID"] = assignments["PatchID"].astype(str)
    assignments_by_patch = {
        patch_id: patch_df.copy()
        for patch_id, patch_df in assignments.groupby("PatchID", sort=True)
    }

    features_by_patch: list[pd.DataFrame] = []
    print(f"Processing {len(assignments_by_patch)} patch masks...")

    missing_masks = 0
    for patch_id, patch_cells in assignments_by_patch.items():
        mask_path = mask_dir / f"{patch_id}.png"
        if not mask_path.exists():
            missing_masks += 1
            features_by_patch.append(_empty_shape_rows(patch_cells))
            continue

        labeled_mask = _load_labeled_mask(mask_path)
        regions_by_label = {
            region.label: region for region in _regions_from_labeled_mask(labeled_mask)
        }
        features_by_patch.append(
            match_cells_to_regions(patch_cells, labeled_mask, regions_by_label)
        )

    features_df = (
        pd.concat(features_by_patch, ignore_index=True)
        if features_by_patch
        else pd.DataFrame(columns=SHAPE_COLUMNS)
    )

    unmatched = int(features_df["circularity"].isna().sum())
    total = int(len(features_df))
    rate = (unmatched / total) if total else 0.0
    if missing_masks:
        print(f"WARNING: missing {missing_masks} patch mask files under {mask_dir}")
    if rate > 0.05:
        print(f"WARNING: {unmatched}/{total} cells ({rate:.1%}) unmatched to mask regions.")
    else:
        print(f"Matched {total - unmatched}/{total} cells ({1.0 - rate:.1%}).")

    out_path = data_dir / "cell_shape_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved {len(features_df)} records -> {out_path}")
    return features_df


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    compute_shape_features(Path(args.data_dir))


if __name__ == "__main__":
    main()
