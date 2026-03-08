"""Shared helpers for interactive H&E + multiplex group visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import matplotlib
import numpy as np
import pandas as pd
import tifffile


@dataclass(frozen=True)
class GroupDefinition:
    """Display and marker config for one multiplex group."""

    id: str
    label: str
    markers: tuple[str, ...]
    colormap: str = "turbo"


GROUP_DEFINITIONS: dict[str, GroupDefinition] = {
    "immune": GroupDefinition(
        id="immune",
        label="Immune",
        markers=(
            "CD45",
            "CD4",
            "CD8a",
            "CD3",
            "CD20",
            "CD68",
            "CD163",
            "FOXP3",
            "CD45RO",
        ),
        colormap="tab20",
    ),
    "vasculature": GroupDefinition(
        id="vasculature",
        label="Vasculature",
        markers=("CD31",),
        colormap="Reds",
    ),
    "cancer": GroupDefinition(
        id="cancer",
        label="Cancer Cells",
        markers=(
            "Keratin",
            "PanCk",
            "PanCK",
            "Pan-cytokeratin",
            "Ecadherin",
            "E-cadherin",
            "CDX2",
            "CDX-2",
        ),
        colormap="plasma",
    ),
    "proliferative": GroupDefinition(
        id="proliferative",
        label="Proliferative Cells",
        markers=("Ki67", "Antigen Ki67", "Ki67_570", "Antigen Ki67 (2)", "PCNA"),
        colormap="YlGn",
    ),
}


@dataclass(frozen=True)
class MarkerChannel:
    """Resolved marker channel in multiplex image."""

    marker_name: str
    channel_index: int


@dataclass(frozen=True)
class GroupResolution:
    """Resolved channels for one group."""

    definition: GroupDefinition
    channels: tuple[MarkerChannel, ...]


@dataclass(frozen=True)
class GroupFrame:
    """Rendered data for one group at current visualization level."""

    group_id: str
    label: str
    image_rgb: np.ndarray
    used_markers: tuple[str, ...]
    missing_markers: tuple[str, ...]
    colormap: str
    marker_colors: tuple[tuple[str, tuple[int, int, int]], ...]


def normalize_marker_name(name: str) -> str:
    """Normalize marker names for robust matching across punctuation/spacing."""
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


def semantic_marker_key(name: str) -> str:
    """Map marker aliases to a semantic key for de-duplicated group rendering."""
    key = normalize_marker_name(name)
    proliferation_family = {
        "ki67",
        "ki67570",
        "antigenki67",
        "antigenki672",
        "pcna",
    }
    if key in proliferation_family:
        return "proliferation_marker"
    return key


def load_marker_channel_map(csv_path: str | Path) -> dict[str, MarkerChannel]:
    """Return marker mapping keyed by normalized marker name."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"Channel_Number", "Marker_Name"}
    if not required.issubset(df.columns):
        raise ValueError(
            "metadata CSV missing required columns: "
            f"expected {sorted(required)}, found {list(df.columns)}"
        )

    out: dict[str, MarkerChannel] = {}
    for _, row in df.iterrows():
        raw_marker = str(row["Marker_Name"]).strip()
        if not raw_marker:
            continue
        key = normalize_marker_name(raw_marker)
        if key in out:
            continue

        try:
            idx = int(row["Channel_Number"]) - 1
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Channel_Number value: {row['Channel_Number']}"
            ) from exc

        if idx < 0:
            raise ValueError(
                f"Channel_Number must be >= 1, found {row['Channel_Number']}"
            )

        out[key] = MarkerChannel(marker_name=raw_marker, channel_index=idx)

    return out


class OMEPyramidReader:
    """Read selected channels from a pyramidal OME-TIFF level lazily."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._tif = tifffile.TiffFile(str(self.path))
        self._series = self._tif.series[0]

    def close(self) -> None:
        self._tif.close()

    def level_count(self) -> int:
        return len(self._series.levels)

    def level_axes(self, level: int) -> str:
        return self._series.levels[level].axes.upper()

    def level_shape(self, level: int) -> tuple[int, int]:
        axes = self.level_axes(level)
        shape = self._series.levels[level].shape
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])

    def choose_level(self, min_level: int = 1, target_max_dim: int = 2400) -> int:
        n_levels = self.level_count()
        if n_levels <= 1:
            return 0

        start = min(max(min_level, 0), n_levels - 1)
        for level in range(start, n_levels):
            h, w = self.level_shape(level)
            if max(h, w) <= target_max_dim:
                return level
        return n_levels - 1

    def read_channel(self, level: int, channel_index: int) -> np.ndarray:
        """Read one channel in (H, W) order using on-disk dtype."""
        if level < 0 or level >= self.level_count():
            raise ValueError(
                f"invalid level={level}; range is [0, {self.level_count() - 1}]"
            )

        level_series = self._series.levels[level]
        axes = level_series.axes.upper()

        c_axis = axes.find("C")
        if c_axis >= 0:
            c_size = int(level_series.shape[c_axis])
            if channel_index < 0 or channel_index >= c_size:
                raise IndexError(
                    f"channel index {channel_index} out of range [0, {c_size - 1}]"
                )

        # Fast path for CYX pyramids (our WSI files): read only one channel page.
        if axes == "CYX" and len(level_series.pages) == int(level_series.shape[0]):
            arr = np.asarray(level_series.pages[int(channel_index)].asarray())
            return arr

        # Fallback for uncommon axis layouts: read full level then slice.
        full = np.asarray(level_series.asarray())
        if "C" not in axes:
            arr = full
        else:
            c_pos = axes.index("C")
            arr = np.take(full, indices=int(channel_index), axis=c_pos)

        active = [ax for ax in axes if ax in ("Y", "X")]
        if active != ["Y", "X"]:
            arr = arr.transpose([active.index("Y"), active.index("X")])

        return arr

    def read_channels(self, level: int, channel_indices: list[int]) -> np.ndarray:
        """Read multiple channels as (C, H, W)."""
        if not channel_indices:
            h, w = self.level_shape(level)
            return np.zeros((0, h, w), dtype=np.float32)
        chans = [self.read_channel(level, idx) for idx in channel_indices]
        return np.stack(chans, axis=0)

    def read_rgb(self, level: int) -> np.ndarray:
        """Read first 3 channels (or replicate channel 0) as uint8 RGB."""
        axes = self.level_axes(level)
        if "C" in axes:
            c_size = int(self._series.levels[level].shape[axes.index("C")])
        else:
            c_size = 1

        if c_size >= 3:
            chw = self.read_channels(level, [0, 1, 2])
        else:
            ch0 = self.read_channel(level, 0)
            chw = np.stack([ch0, ch0, ch0], axis=0)

        if chw.dtype == np.uint8:
            return np.moveaxis(chw, 0, -1)
        return np.moveaxis(_to_uint8(chw), 0, -1)


def resolve_group_channels(
    marker_map: Mapping[str, MarkerChannel],
    groups: Mapping[str, GroupDefinition] = GROUP_DEFINITIONS,
) -> dict[str, GroupResolution]:
    """Resolve available marker channels for each configured group."""
    out: dict[str, GroupResolution] = {}
    for group_id, definition in groups.items():
        found: list[MarkerChannel] = []
        seen_indices: set[int] = set()
        seen_semantic: set[str] = set()
        for marker in definition.markers:
            key = normalize_marker_name(marker)
            if key in marker_map:
                channel = marker_map[key]
                if channel.channel_index in seen_indices:
                    continue
                sem_key = semantic_marker_key(channel.marker_name)
                if sem_key in seen_semantic:
                    continue
                found.append(channel)
                seen_indices.add(channel.channel_index)
                seen_semantic.add(sem_key)
        out[group_id] = GroupResolution(definition=definition, channels=tuple(found))
    return out


def _to_unit_no_percentile(arr: np.ndarray) -> np.ndarray:
    """Scale by min-max (no percentile normalization)."""
    if arr.ndim == 3:
        out = np.zeros(arr.shape, dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i] = _to_unit_no_percentile(arr[i])
        return out

    arr_f = arr.astype(np.float32, copy=False)
    if arr_f.size == 0:
        return np.zeros(arr_f.shape, dtype=np.float32)

    min_value = float(np.nanmin(arr_f))
    max_value = float(np.nanmax(arr_f))
    denom = max(max_value - min_value, 1e-6)
    return np.clip((arr_f - min_value) / denom, 0.0, 1.0)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (_to_unit_no_percentile(arr) * 255.0).round().astype(np.uint8)


def enhance_unit_contrast(
    unit: np.ndarray,
    gamma: float = 0.65,
    gain: float = 1.35,
) -> np.ndarray:
    """Brightness/contrast boost in unit space without percentile statistics."""
    g = float(max(0.05, gamma))
    k = float(max(0.1, gain))
    boosted = np.power(np.clip(unit, 0.0, 1.0), g) * k
    return np.clip(boosted, 0.0, 1.0)


def combine_channels(chw: np.ndarray, mode: str = "max") -> np.ndarray:
    """Combine channels into one unit-scale map using max or sum."""
    if chw.ndim != 3:
        raise ValueError(f"expected CHW array, got shape {chw.shape}")
    if chw.shape[0] == 0:
        raise ValueError("cannot combine zero channels")

    unit = _to_unit_no_percentile(chw)
    mode_l = mode.strip().lower()
    if mode_l == "max":
        return np.max(unit, axis=0)
    if mode_l == "sum":
        return np.clip(np.sum(unit, axis=0), 0.0, 1.0)
    raise ValueError(f"unsupported combine mode '{mode}'. Expected: max, sum")


def tint_map(gray_map: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    """Colorize a unit-scale map into uint8 RGB using a fixed tint color."""
    color = np.asarray(color_rgb, dtype=np.float32) / 255.0
    rgb = gray_map[:, :, None] * color[None, None, :]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def apply_colormap(gray_map: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Apply a matplotlib colormap to a unit-scale map and return uint8 RGB."""
    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(np.clip(gray_map, 0.0, 1.0))
    return np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8)


def max_project_multicolor(
    chw: np.ndarray,
    marker_names: tuple[str, ...],
    colormap: str = "tab20",
    contrast_gamma: float = 0.65,
    contrast_gain: float = 1.35,
) -> tuple[np.ndarray, tuple[tuple[str, tuple[int, int, int]], ...]]:
    """Render multi-channel max-projection with one colormap-derived color per channel."""
    if chw.ndim != 3:
        raise ValueError(f"expected CHW array, got shape {chw.shape}")
    if chw.shape[0] <= 0:
        raise ValueError("cannot project zero channels")
    if len(marker_names) != chw.shape[0]:
        raise ValueError(
            f"marker_names length ({len(marker_names)}) must match channels ({chw.shape[0]})"
        )

    unit = _to_unit_no_percentile(chw)
    unit = enhance_unit_contrast(
        unit,
        gamma=contrast_gamma,
        gain=contrast_gain,
    )
    cmap = matplotlib.colormaps[colormap]
    if chw.shape[0] == 1:
        color_positions = np.array([0.85], dtype=np.float32)
    else:
        color_positions = np.linspace(0.05, 0.95, num=chw.shape[0], dtype=np.float32)

    colors = np.array(
        [cmap(float(pos))[:3] for pos in color_positions], dtype=np.float32
    )
    colored = unit[:, :, :, None] * colors[:, None, None, :]
    rgb = np.max(colored, axis=0)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    marker_colors: list[tuple[str, tuple[int, int, int]]] = []
    for name, color in zip(marker_names, colors):
        marker_colors.append((name, tuple((color * 255.0).astype(np.uint8).tolist())))

    return rgb_u8, tuple(marker_colors)


def alpha_blend(
    base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend two uint8 RGB images with the provided overlay alpha."""
    a = float(np.clip(alpha, 0.0, 1.0))
    blended = (
        base_rgb.astype(np.float32) * (1.0 - a) + overlay_rgb.astype(np.float32) * a
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


class GroupVisualizerCore:
    """Shared cached renderer used by both desktop and web interactive viewers."""

    def __init__(
        self,
        he_image: str | Path,
        multiplex_image: str | Path,
        metadata_csv: str | Path,
        he_level: int | None = None,
        multiplex_level: int | None = None,
        auto_target_max_dim: int = 2400,
        min_level: int = 1,
        default_colormap: str = "turbo",
        index_json: str | Path | None = None,
        preload_multiplex: bool = False,
        contrast_gamma: float = 0.65,
        contrast_gain: float = 1.35,
    ):
        self.he_reader = OMEPyramidReader(he_image)
        self.mx_reader = OMEPyramidReader(multiplex_image)
        self.default_colormap = default_colormap
        self.preload_multiplex = preload_multiplex
        self.contrast_gamma = contrast_gamma
        self.contrast_gain = contrast_gain

        self.he_level = (
            self.he_reader.choose_level(
                min_level=min_level, target_max_dim=auto_target_max_dim
            )
            if he_level is None
            else he_level
        )
        self.multiplex_level = (
            self.mx_reader.choose_level(
                min_level=min_level, target_max_dim=auto_target_max_dim
            )
            if multiplex_level is None
            else multiplex_level
        )

        marker_map = load_marker_channel_map(metadata_csv)
        self.group_resolutions = resolve_group_channels(marker_map)

        self._he_rgb = self.he_reader.read_rgb(self.he_level)
        self._group_rgb_cache: dict[tuple[str, str], GroupFrame] = {}
        self._mx_level_stack: np.ndarray | None = None
        self._mx_channel_render_cache: dict[int, np.ndarray] = {}
        self.registration_index_json: str | None = None
        self._warp_matrix_level: np.ndarray | None = self._load_scaled_warp_matrix(
            index_json
        )
        if self.preload_multiplex:
            self._preload_multiplex_level()

    def _load_scaled_warp_matrix(
        self,
        index_json: str | Path | None,
    ) -> np.ndarray | None:
        candidates: list[Path] = []
        if index_json is not None:
            candidates.append(Path(index_json))
        else:
            candidates.extend(
                [
                    Path("processed_wd/index.json"),
                    Path("proceeded_wd/index.json"),
                ]
            )

        selected: Path | None = None
        payload: dict | None = None
        for candidate in candidates:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    payload = json.load(f)
                selected = candidate
                break

        if selected is None or payload is None:
            return None

        if "warp_matrix" not in payload:
            return None

        m_full = np.asarray(payload["warp_matrix"], dtype=np.float64)
        if m_full.shape != (2, 3):
            raise ValueError(
                f"warp_matrix in {selected} must have shape (2,3), got {m_full.shape}"
            )

        he_h0, he_w0 = self.he_reader.level_shape(0)
        he_h, he_w = self.he_reader.level_shape(self.he_level)
        mx_h0, mx_w0 = self.mx_reader.level_shape(0)
        mx_h, mx_w = self.mx_reader.level_shape(self.multiplex_level)

        s_he_x = he_w0 / float(max(1, he_w))
        s_he_y = he_h0 / float(max(1, he_h))
        s_mx_x = mx_w0 / float(max(1, mx_w))
        s_mx_y = mx_h0 / float(max(1, mx_h))

        m_scaled = np.array(
            [
                [
                    m_full[0, 0] * (s_he_x / s_mx_x),
                    m_full[0, 1] * (s_he_y / s_mx_x),
                    m_full[0, 2] / s_mx_x,
                ],
                [
                    m_full[1, 0] * (s_he_x / s_mx_y),
                    m_full[1, 1] * (s_he_y / s_mx_y),
                    m_full[1, 2] / s_mx_y,
                ],
            ],
            dtype=np.float64,
        )

        self.registration_index_json = str(selected)
        return m_scaled

    def close(self) -> None:
        self.he_reader.close()
        self.mx_reader.close()

    @property
    def he_rgb(self) -> np.ndarray:
        return self._he_rgb

    def summary(self) -> dict:
        he_h, he_w = self.he_rgb.shape[:2]
        mx_h, mx_w = self.mx_reader.level_shape(self.multiplex_level)
        return {
            "he_level": self.he_level,
            "multiplex_level": self.multiplex_level,
            "he_shape": [he_h, he_w],
            "multiplex_shape": [mx_h, mx_w],
            "default_colormap": self.default_colormap,
            "registered": self._warp_matrix_level is not None,
            "registration_index_json": self.registration_index_json,
            "preload_multiplex": self.preload_multiplex,
            "contrast_gamma": self.contrast_gamma,
            "contrast_gain": self.contrast_gain,
            "groups": self.available_groups(),
        }

    def _preload_multiplex_level(self) -> None:
        """Preload all multiplex channels at selected level for faster group switching."""
        axes = self.mx_reader.level_axes(self.multiplex_level)
        level_series = self.mx_reader._series.levels[
            self.multiplex_level
        ]  # noqa: SLF001

        if "C" not in axes:
            arr = np.asarray(level_series.asarray())
            if arr.ndim == 2:
                self._mx_level_stack = arr[np.newaxis, :, :]
            else:
                self._mx_level_stack = arr
            return

        c_size = int(level_series.shape[axes.index("C")])
        if axes == "CYX" and len(level_series.pages) == c_size:
            self._mx_level_stack = np.stack(
                [np.asarray(level_series.pages[c].asarray()) for c in range(c_size)],
                axis=0,
            )
            return

        full = np.asarray(level_series.asarray())
        c_pos = axes.index("C")
        if c_pos != 0:
            perm = [c_pos, *[i for i in range(full.ndim) if i != c_pos]]
            full = full.transpose(perm)
        if full.ndim > 3:
            full = full.reshape(full.shape[0], full.shape[-2], full.shape[-1])
        self._mx_level_stack = full

    def _read_mx_channel_at_level(self, channel_index: int) -> np.ndarray:
        if self._mx_level_stack is not None:
            return self._mx_level_stack[channel_index]
        return self.mx_reader.read_channel(self.multiplex_level, channel_index)

    def _transform_mx_channel_to_he(self, channel_index: int) -> np.ndarray:
        """Read one multiplex channel and transform into H&E level space."""
        if channel_index in self._mx_channel_render_cache:
            return self._mx_channel_render_cache[channel_index]

        channel = self._read_mx_channel_at_level(channel_index)
        he_h, he_w = self.he_rgb.shape[:2]

        if self._warp_matrix_level is not None:
            transformed = cv2.warpAffine(
                channel.astype(np.float32),
                self._warp_matrix_level,
                (he_w, he_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        elif channel.shape != (he_h, he_w):
            transformed = cv2.resize(
                channel, (he_w, he_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            transformed = channel.astype(np.float32, copy=False)

        self._mx_channel_render_cache[channel_index] = transformed
        return transformed

    def available_groups(self) -> list[dict]:
        groups: list[dict] = [
            {
                "id": "he",
                "label": "H&E",
                "available": True,
                "used_markers": [],
                "missing_markers": [],
            }
        ]
        for group_id, resolution in self.group_resolutions.items():
            used = [x.marker_name for x in resolution.channels]
            missing = [
                marker
                for marker in resolution.definition.markers
                if normalize_marker_name(marker)
                not in {
                    normalize_marker_name(x.marker_name) for x in resolution.channels
                }
            ]
            groups.append(
                {
                    "id": group_id,
                    "label": resolution.definition.label,
                    "available": len(used) > 0,
                    "colormap": resolution.definition.colormap,
                    "used_markers": used,
                    "missing_markers": missing,
                }
            )
        return groups

    def render_group(self, group_id: str, combine_mode: str = "max") -> GroupFrame:
        group_key = group_id.strip().lower()
        mode_key = combine_mode.strip().lower()
        cache_key = (group_key, mode_key)
        if cache_key in self._group_rgb_cache:
            return self._group_rgb_cache[cache_key]

        if group_key == "he":
            frame = GroupFrame(
                group_id="he",
                label="H&E",
                image_rgb=self.he_rgb,
                used_markers=(),
                missing_markers=(),
                colormap="he_rgb",
                marker_colors=(),
            )
            self._group_rgb_cache[cache_key] = frame
            return frame

        if group_key not in self.group_resolutions:
            raise ValueError(
                f"unknown group '{group_id}'. Available: he, {', '.join(self.group_resolutions.keys())}"
            )

        resolution = self.group_resolutions[group_key]
        if not resolution.channels:
            raise ValueError(
                f"group '{group_key}' has no available markers in metadata CSV"
            )

        idxs = [channel.channel_index for channel in resolution.channels]
        mx_chw = np.stack(
            [self._transform_mx_channel_to_he(idx) for idx in idxs],
            axis=0,
        )
        used_marker_names = tuple(
            channel.marker_name for channel in resolution.channels
        )

        cmap_name = resolution.definition.colormap or self.default_colormap
        if mx_chw.shape[0] > 1:
            if mode_key != "max":
                raise ValueError(
                    "Multi-channel groups support max projection only. "
                    "Use combine_mode='max'."
                )
            group_rgb, marker_colors = max_project_multicolor(
                mx_chw,
                marker_names=used_marker_names,
                colormap=cmap_name,
                contrast_gamma=self.contrast_gamma,
                contrast_gain=self.contrast_gain,
            )
        else:
            unit = _to_unit_no_percentile(mx_chw[0])
            unit = enhance_unit_contrast(
                unit,
                gamma=self.contrast_gamma,
                gain=self.contrast_gain,
            )
            group_rgb = apply_colormap(unit, colormap=cmap_name)
            marker_colors = (
                (
                    used_marker_names[0],
                    tuple(
                        (np.array(matplotlib.colormaps[cmap_name](0.85)[:3]) * 255.0)
                        .astype(np.uint8)
                        .tolist()
                    ),
                ),
            )

        missing_marker_names = tuple(
            marker
            for marker in resolution.definition.markers
            if normalize_marker_name(marker)
            not in {normalize_marker_name(x) for x in used_marker_names}
        )

        frame = GroupFrame(
            group_id=group_key,
            label=resolution.definition.label,
            image_rgb=group_rgb,
            used_markers=used_marker_names,
            missing_markers=missing_marker_names,
            colormap=cmap_name,
            marker_colors=marker_colors,
        )
        self._group_rgb_cache[cache_key] = frame
        return frame

    def render_overlay(
        self,
        group_id: str,
        alpha: float = 0.5,
        combine_mode: str = "max",
    ) -> tuple[GroupFrame, np.ndarray]:
        frame = self.render_group(group_id, combine_mode=combine_mode)
        if frame.group_id == "he":
            return frame, self.he_rgb
        return frame, alpha_blend(self.he_rgb, frame.image_rgb, alpha=alpha)
