#!/usr/bin/env python3
"""Interactive desktop viewer for H&E + multiplex marker groups."""

from __future__ import annotations

import argparse
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from utils.group_visualizer import GroupVisualizerCore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Matplotlib viewer for H&E + multiplex groups."
    )
    parser.add_argument(
        "--he-image",
        default="data/WD-76845-096.ome.tif",
        help="Path to H&E OME-TIFF",
    )
    parser.add_argument(
        "--multiplex-image",
        default="data/WD-76845-097.ome.tif",
        help="Path to multiplex OME-TIFF",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/WD-76845-097-metadata.csv",
        help="Path to channel metadata CSV",
    )
    parser.add_argument(
        "--index-json",
        default=None,
        help=(
            "Path to registration index.json containing warp_matrix. "
            "If omitted, auto-detects processed_wd/index.json or proceeded_wd/index.json."
        ),
    )
    parser.add_argument("--he-level", type=int, default=None, help="H&E pyramid level")
    parser.add_argument(
        "--multiplex-level", type=int, default=None, help="Multiplex pyramid level"
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=1,
        help="Minimum pyramid level for auto selection (default: 1)",
    )
    parser.add_argument(
        "--auto-max-dim",
        type=int,
        default=1200,
        help="Target max side for auto level selection (default: 1200)",
    )
    parser.add_argument(
        "--default-colormap",
        default="turbo",
        help="Fallback matplotlib colormap for group rendering",
    )
    parser.add_argument(
        "--contrast-gamma",
        type=float,
        default=0.65,
        help="Gamma (<1 brightens multiplex channels, default: 0.65)",
    )
    parser.add_argument(
        "--contrast-gain",
        type=float,
        default=1.35,
        help="Gain multiplier for multiplex brightness (default: 1.35)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Multiplex panel transparency in [0, 1] (default: 1.0)",
    )
    parser.add_argument(
        "--group",
        default="immune",
        help="Initial group id (he, immune, vasculature, cancer, proliferative)",
    )
    parser.add_argument(
        "--combine-mode",
        choices=("max",),
        default="max",
        help="Channel combine mode (max projection).",
    )
    parser.add_argument(
        "--save-png",
        default=None,
        help="Write a static snapshot to PNG and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    core = GroupVisualizerCore(
        he_image=args.he_image,
        multiplex_image=args.multiplex_image,
        metadata_csv=args.metadata_csv,
        he_level=args.he_level,
        multiplex_level=args.multiplex_level,
        auto_target_max_dim=args.auto_max_dim,
        min_level=args.min_level,
        default_colormap=args.default_colormap,
        index_json=args.index_json,
        contrast_gamma=args.contrast_gamma,
        contrast_gain=args.contrast_gain,
    )

    try:
        summary = core.summary()
        groups = summary["groups"]
        group_ids = [group["id"] for group in groups]
        group_label_to_id = {
            f"{group['label']}{'' if group['available'] else ' (NA)'}": group["id"]
            for group in groups
        }
        group_labels = list(group_label_to_id.keys())

        initial_group = args.group.lower().strip()
        if initial_group not in group_ids:
            initial_group = "he"
        initial_mode = args.combine_mode.strip().lower()
        frame = core.render_group(initial_group, combine_mode=initial_mode)
        initial_alpha = float(max(0.05, min(1.0, args.alpha)))

        fig, (ax_he, ax_group) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#141414")
        plt.subplots_adjust(left=0.06, right=0.70, top=0.90, bottom=0.26, wspace=0.05)

        for ax in (ax_he, ax_group):
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])

        ax_he.imshow(core.he_rgb)
        ax_he.set_title("H&E", color="white")

        ax_group.imshow(core.he_rgb, zorder=1)
        im_group = ax_group.imshow(frame.image_rgb, zorder=2)
        im_group.set_alpha(initial_alpha)
        ax_group.set_title(f"{frame.label} on H&E", color="white")

        fig.suptitle(
            "H&E + Multiplex Group Viewer"
            f"  |  he_level={summary['he_level']}  mx_level={summary['multiplex_level']}"
            f"  |  registered={summary['registered']}",
            color="white",
        )

        ax_group_radio = plt.axes([0.06, 0.05, 0.23, 0.17], facecolor="#222")
        radio_group = RadioButtons(ax_group_radio, group_labels)
        ax_group_radio.set_title("Group", color="white", fontsize=10)
        for txt in radio_group.labels:
            txt.set_color("white")

        active_label = next(
            label for label, gid in group_label_to_id.items() if gid == initial_group
        )
        radio_group.set_active(group_labels.index(active_label))

        ax_mode_radio = plt.axes([0.31, 0.05, 0.12, 0.17], facecolor="#222")
        radio_mode = RadioButtons(ax_mode_radio, ["max"])
        ax_mode_radio.set_title("Combine", color="white", fontsize=10)
        for txt in radio_mode.labels:
            txt.set_color("white")
        radio_mode.set_active(0)

        ax_alpha = plt.axes([0.46, 0.10, 0.20, 0.05], facecolor="#222")
        slider_alpha = Slider(
            ax=ax_alpha,
            label="Transparency",
            valmin=0.05,
            valmax=1.0,
            valinit=initial_alpha,
            color="#4f9cff",
        )

        ax_info = plt.axes([0.72, 0.04, 0.25, 0.19], facecolor="#1f1f1f")
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        ax_info.set_title("Markers", color="white", fontsize=10)
        marker_text = ax_info.text(
            0.01,
            0.95,
            "",
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            wrap=True,
        )
        legend_artists = []

        state = {"group": initial_group, "mode": initial_mode, "alpha": initial_alpha}

        def zoom_axis(
            ax, x_center: float, y_center: float, scale_factor: float
        ) -> None:
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cur_w = cur_xlim[1] - cur_xlim[0]
            cur_h = cur_ylim[1] - cur_ylim[0]
            if cur_w == 0 or cur_h == 0:
                return
            rel_x = (x_center - cur_xlim[0]) / cur_w
            rel_y = (y_center - cur_ylim[0]) / cur_h
            new_w = cur_w * scale_factor
            new_h = cur_h * scale_factor
            new_x0 = x_center - rel_x * new_w
            new_x1 = x_center + (1.0 - rel_x) * new_w
            new_y0 = y_center - rel_y * new_h
            new_y1 = y_center + (1.0 - rel_y) * new_h
            ax.set_xlim(new_x0, new_x1)
            ax.set_ylim(new_y0, new_y1)

        def on_scroll(event) -> None:
            if event.inaxes not in (ax_he, ax_group):
                return
            if event.xdata is None or event.ydata is None:
                return
            step = 1.2
            scale = 1.0 / step if event.button == "up" else step
            zoom_axis(event.inaxes, event.xdata, event.ydata, scale)
            if event.inaxes is ax_he:
                ax_group.set_xlim(ax_he.get_xlim())
                ax_group.set_ylim(ax_he.get_ylim())
            else:
                ax_he.set_xlim(ax_group.get_xlim())
                ax_he.set_ylim(ax_group.get_ylim())
            fig.canvas.draw_idle()

        def redraw() -> None:
            try:
                selected_frame = core.render_group(
                    state["group"],
                    combine_mode=state["mode"],
                )
            except ValueError as exc:
                marker_text.set_text(str(exc))
                fig.canvas.draw_idle()
                return

            im_group.set_data(selected_frame.image_rgb)
            im_group.set_alpha(state["alpha"])
            ax_group.set_title(f"{selected_frame.label} on H&E", color="white")

            used = (
                ", ".join(selected_frame.used_markers)
                if selected_frame.used_markers
                else "None"
            )
            colors = "None"
            if selected_frame.marker_colors:
                colors = ", ".join(
                    f"{name}:{to_hex((rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0))}"
                    for name, rgb in selected_frame.marker_colors
                )
            marker_text.set_text(
                f"Colormap: {selected_frame.colormap}\n"
                f"Used: {used}\n"
                f"Colors: {colors}"
            )
            while legend_artists:
                artist = legend_artists.pop()
                artist.remove()
            if selected_frame.marker_colors:
                n = min(len(selected_frame.marker_colors), 12)
                step = 1.0 / (n + 1)
                y = 0.98 - step
                bg_h = max(0.18, min(0.90, (n + 1) * step + 0.03))
                bg = Rectangle(
                    (0.64, 0.98 - bg_h),
                    0.34,
                    bg_h,
                    transform=ax_group.transAxes,
                    facecolor=(0.05, 0.05, 0.05, 0.75),
                    edgecolor=(1.0, 1.0, 1.0, 0.35),
                    linewidth=0.5,
                    zorder=20,
                )
                ax_group.add_patch(bg)
                legend_artists.append(bg)
                title = ax_group.text(
                    0.66,
                    0.965,
                    "Color Legend",
                    transform=ax_group.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    zorder=21,
                )
                legend_artists.append(title)
                for marker_name, rgb in selected_frame.marker_colors[:n]:
                    color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                    patch = Rectangle(
                        (0.66, y - 0.020),
                        0.025,
                        0.025,
                        transform=ax_group.transAxes,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=21,
                    )
                    ax_group.add_patch(patch)
                    legend_artists.append(patch)
                    txt = ax_group.text(
                        0.69,
                        y,
                        marker_name[:20],
                        transform=ax_group.transAxes,
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="white",
                        zorder=21,
                    )
                    legend_artists.append(txt)
                    y -= step
            fig.canvas.draw_idle()

        def on_group_change(label: str) -> None:
            state["group"] = group_label_to_id[label]
            redraw()

        def on_mode_change(label: str) -> None:
            state["mode"] = label
            redraw()

        def on_alpha_change(_val: float) -> None:
            state["alpha"] = float(slider_alpha.val)
            redraw()

        radio_group.on_clicked(on_group_change)
        radio_mode.on_clicked(on_mode_change)
        slider_alpha.on_changed(on_alpha_change)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        redraw()

        if args.save_png:
            fig.savefig(args.save_png, dpi=150, facecolor=fig.get_facecolor())
            print(f"saved: {args.save_png}")
            return

        plt.show()
    finally:
        core.close()


if __name__ == "__main__":
    main()
