from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class RepresentativePatch:
    group_id: str
    patch_id: str
    generated_he_path: Path


@dataclass(frozen=True)
class Stage2FigureAssets:
    patch_id: str
    group_id: str
    reference_he_path: Path
    cell_type_path: Path
    cell_state_path: Path
    vasculature_path: Path
    oxygen_path: Path
    glucose_path: Path
    generated_he_path: Path


def _iter_selected_patch_ids(selection_json: Path) -> list[tuple[str, str]]:
    try:
        with selection_json.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed selection manifest {selection_json}: {exc.msg}") from exc

    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError(f"Malformed selection manifest {selection_json}: groups must be a mapping")

    ordered: list[tuple[str, str]] = []
    for group_id, group in groups.items():
        if not isinstance(group, dict):
            raise ValueError(
                f"Malformed selection manifest {selection_json}: group {group_id!r} must be a mapping"
            )
        selections = group.get("selections", [])
        if not isinstance(selections, list):
            raise ValueError(
                f"Malformed selection manifest {selection_json}: group {group_id!r} selections must be a list"
            )
        for index, selection in enumerate(selections):
            if not isinstance(selection, dict) or "patch_id" not in selection:
                raise ValueError(
                    f"Malformed selection manifest {selection_json}: missing patch_id for group {group_id!r} selection {index}"
                )
            ordered.append((str(group_id), str(selection["patch_id"])))
    return ordered


def _generated_he_path(pixcell_root: Path, patch_id: str) -> Path:
    return pixcell_root / patch_id / "all" / "generated_he.png"


def _group_for_patch_id(selection_json: Path, patch_id: str) -> str | None:
    for group_id, selected_patch_id in _iter_selected_patch_ids(selection_json):
        if selected_patch_id == patch_id:
            return group_id
    return None


def pick_representative_patch(
    selection_json: Path,
    pixcell_root: Path,
    group_id: str | None = None,
) -> RepresentativePatch:
    checked_patch_ids: list[str] = []
    for gid, patch_id in _iter_selected_patch_ids(selection_json):
        if group_id is not None and gid != group_id:
            continue
        checked_patch_ids.append(patch_id)
        generated_he_path = _generated_he_path(pixcell_root, patch_id)
        if generated_he_path.exists():
            return RepresentativePatch(
                group_id=gid,
                patch_id=patch_id,
                generated_he_path=generated_he_path,
            )

    suffix = f" in group {group_id!r}" if group_id else ""
    raise FileNotFoundError(
        f"No selected patch{suffix} has PixCell generated output. Checked patch IDs: "
        + ", ".join(checked_patch_ids)
    )


def _require_existing(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required stage2 asset: {path}")


def resolve_stage2_assets(
    processed_dir: Path,
    selection_json: Path,
    pixcell_root: Path,
    patch_id: str | None = None,
    group_id: str | None = None,
) -> Stage2FigureAssets:
    match = (
        RepresentativePatch(
            group_id=_group_for_patch_id(selection_json, str(patch_id)) or "forced",
            patch_id=str(patch_id),
            generated_he_path=_generated_he_path(pixcell_root, str(patch_id)),
        )
        if patch_id is not None
        else pick_representative_patch(selection_json, pixcell_root, group_id=group_id)
    )

    assets = Stage2FigureAssets(
        patch_id=match.patch_id,
        group_id=match.group_id,
        reference_he_path=processed_dir / "he" / f"{match.patch_id}.png",
        cell_type_path=processed_dir / "cell_types/union" / f"{match.patch_id}.png",
        cell_state_path=processed_dir / "cell_states/union" / f"{match.patch_id}.png",
        vasculature_path=processed_dir / "vasculature" / f"{match.patch_id}.png",
        oxygen_path=processed_dir / "oxygen" / f"{match.patch_id}.png",
        glucose_path=processed_dir / "glucose" / f"{match.patch_id}.png",
        generated_he_path=match.generated_he_path,
    )

    _require_existing(assets.reference_he_path)
    _require_existing(assets.cell_type_path)
    _require_existing(assets.cell_state_path)
    _require_existing(assets.vasculature_path)
    _require_existing(assets.oxygen_path)
    _require_existing(assets.glucose_path)
    _require_existing(assets.generated_he_path)
    return assets