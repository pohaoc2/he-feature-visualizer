from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .assets import Stage2FigureAssets


BACKGROUND = (243, 241, 235)
PANEL = (255, 253, 249)
INK = (36, 33, 29)
MUTED = (109, 103, 95)
LINE = (89, 84, 79)
FROZEN_FILL = (247, 244, 239)
FROZEN_STRIPE = (198, 192, 183)
FROZEN_BORDER = (142, 136, 128)
TRAINABLE_FILL = (223, 244, 227)
TRAINABLE_BORDER = (45, 140, 79)
TRAINABLE_TEXT = (32, 95, 55)
LATENT_FILL = (238, 242, 249)
LATENT_BORDER = (138, 160, 195)
LOSS = (179, 59, 52)
TILE_BORDER = (214, 205, 194)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    lib_name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    dejavu_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/liberation2") / lib_name,
        Path("/usr/share/fonts/truetype/liberation") / lib_name,
        Path("/usr/share/fonts/truetype/dejavu") / dejavu_name,
        Path("/usr/local/share/fonts") / lib_name,
        Path("/usr/local/share/fonts") / dejavu_name,
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _fit_square(path: Path, size: int) -> Image.Image:
    return _fit_square_with_resample(path, size, Image.Resampling.LANCZOS)


def _fit_square_with_resample(
    path: Path,
    size: int,
    resample: Image.Resampling,
) -> Image.Image:
    image = _open_rgb(path)
    w, h = image.size
    if w != h:
        sq = min(w, h)
        image = image.crop((0, 0, sq, sq))
    if image.size == (size, size):
        return image
    return image.resize((size, size), resample)


def _compose_metabolic_tile(oxygen_path: Path, glucose_path: Path, size: int) -> Image.Image:
    half_width = size // 2
    oxygen = _fit_square(oxygen_path, size).resize(
        (half_width, size), Image.Resampling.NEAREST
    )
    glucose = _fit_square(glucose_path, size).resize(
        (size - half_width, size), Image.Resampling.NEAREST
    )
    tile = Image.new("RGB", (size, size), (255, 255, 255))
    tile.paste(oxygen, (0, 0))
    tile.paste(glucose, (half_width, 0))
    divider = ImageDraw.Draw(tile)
    divider.line((half_width, 0, half_width, size), fill=(255, 255, 255), width=2)
    return tile


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def _draw_frozen_fill(canvas: Image.Image, box: tuple[int, int, int, int], radius: int) -> None:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    patch = Image.new("RGB", (width, height), FROZEN_FILL)
    patch_draw = ImageDraw.Draw(patch)
    step = 6
    stripe_width = 1
    for start in range(-height, width + height, step):
        patch_draw.line(
            (start, 0, start + height, height),
            fill=FROZEN_STRIPE,
            width=stripe_width,
        )
    canvas.paste(patch, (x0, y0), _rounded_mask((width, height), radius))


def _draw_trainable_fill(canvas: Image.Image, box: tuple[int, int, int, int], radius: int) -> None:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    patch = Image.new("RGB", (width, height), TRAINABLE_FILL)
    canvas.paste(patch, (x0, y0), _rounded_mask((width, height), radius))


def _draw_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    *,
    scale_bar_color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=(255, 255, 255), outline=TILE_BORDER, width=2)
    inset = 4
    fitted = image.resize((x1 - x0 - inset * 2, y1 - y0 - inset * 2), Image.Resampling.NEAREST)
    canvas.paste(fitted, (x0 + inset, y0 + inset))

    band_height = max(10, (y1 - y0) // 6)
    band_box = (x0 + inset, y0 + inset, x1 - inset, y0 + inset + band_height)
    draw.rectangle(band_box, fill=(0, 0, 0))
    draw.text(
        ((band_box[0] + band_box[2]) // 2, (band_box[1] + band_box[3]) // 2),
        label,
        font=font,
        fill=(255, 255, 255),
        anchor="mm",
    )

    bar_margin = max(4, (x1 - x0) // 14)
    bar_length = max(10, (x1 - x0) // 5)
    bar_y = y1 - inset - bar_margin
    bar_x1 = x1 - inset - bar_margin
    bar_x0 = bar_x1 - bar_length
    draw.line((bar_x0, bar_y, bar_x1, bar_y), fill=scale_bar_color, width=2)


def _draw_plain_he_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    *,
    scale_bar_color: tuple[int, int, int],
    scale_bar_label: str = "",
    scale_bar_label_font: ImageFont.ImageFont | None = None,
) -> None:
    x0, y0, x1, y1 = box
    fitted = image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (x0, y0))
    draw.text(((x0 + x1) // 2, y0 - 6), label, font=font, fill=INK, anchor="mb")

    bar_margin = max(4, (x1 - x0) // 14)
    bar_length = max(10, (x1 - x0) // 5)
    bar_y = y1 - bar_margin
    bar_x1 = x1 - bar_margin
    bar_x0 = bar_x1 - bar_length
    draw.line((bar_x0, bar_y, bar_x1, bar_y), fill=scale_bar_color, width=2)
    if scale_bar_label and scale_bar_label_font is not None:
        draw.text(
            ((bar_x0 + bar_x1) // 2, bar_y + 4),
            scale_bar_label,
            font=scale_bar_label_font,
            fill=scale_bar_color,
            anchor="mt",
        )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str | None,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    if subtitle:
        sub_lines = subtitle.split("\n")
        if len(sub_lines) == 1:
            draw.text((cx, cy - 10), title, font=title_font, fill=fill, anchor="mm")
            draw.text((cx, cy + 10), subtitle, font=subtitle_font, fill=MUTED, anchor="mm")
        else:
            line_gap = 11
            total = 1 + len(sub_lines)
            start_y = cy - (total - 1) * line_gap // 2
            draw.text((cx, start_y), title, font=title_font, fill=fill, anchor="mm")
            for idx, line in enumerate(sub_lines):
                draw.text((cx, start_y + (idx + 1) * line_gap), line, font=subtitle_font, fill=MUTED, anchor="mm")
        return
    draw.text((cx, cy), title, font=title_font, fill=fill, anchor="mm")


def _draw_block(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str | None,
    *,
    frozen: bool,
    title_font: ImageFont.ImageFont,
    subtitle_font: ImageFont.ImageFont,
) -> None:
    radius = 10
    if frozen:
        _draw_frozen_fill(canvas, box, radius)
        outline = FROZEN_BORDER
        text_fill = INK
    else:
        _draw_trainable_fill(canvas, box, radius)
        outline = TRAINABLE_BORDER
        text_fill = TRAINABLE_TEXT
    draw.rounded_rectangle(box, radius=radius, outline=outline, width=2)
    _draw_centered_text(draw, box, title, subtitle, title_font, subtitle_font, text_fill)


def _draw_tme_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    fitted = image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (x0, y0))
    band_h = max(14, (y1 - y0) // 7)
    draw.rectangle((x0, y0, x1, y0 + band_h), fill=(20, 20, 20))
    draw.text(((x0 + x1) // 2, y0 + band_h // 2), label, font=font, fill=(255, 255, 255), anchor="mm")


def _draw_capsule(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    title_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=12, fill=LATENT_FILL, outline=LATENT_BORDER, width=2)
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    if len(lines) == 1:
        draw.text((cx, cy), lines[0], font=title_font, fill=(51, 67, 93), anchor="mm")
        return
    draw.text((cx, cy - 8), lines[0], font=title_font, fill=(51, 67, 93), anchor="mm")
    draw.text((cx, cy + 8), lines[1], font=title_font, fill=(51, 67, 93), anchor="mm")


def _draw_latent_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
) -> None:
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    if len(lines) == 1:
        draw.text((cx, cy), lines[0], font=font, fill=(51, 67, 93), anchor="mm")
        return
    draw.text((cx, cy - 7), lines[0], font=font, fill=(51, 67, 93), anchor="mm")
    draw.text((cx, cy + 7), lines[1], font=font, fill=(51, 67, 93), anchor="mm")


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return
    if abs(dx) >= abs(dy):
        direction = (1 if dx > 0 else -1, 0)
    else:
        direction = (0, 1 if dy > 0 else -1)

    size = 8
    if direction[0] != 0:
        sign = direction[0]
        points = [
            end,
            (end[0] - sign * size, end[1] - size // 2),
            (end[0] - sign * size, end[1] + size // 2),
        ]
    else:
        sign = direction[1]
        points = [
            end,
            (end[0] - size // 2, end[1] - sign * size),
            (end[0] + size // 2, end[1] - sign * size),
        ]
    draw.polygon(points, fill=color)


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    for idx in range(len(points) - 1):
        draw.line((points[idx], points[idx + 1]), fill=color, width=width)
    _draw_arrow_head(draw, points[-2], points[-1], color)


def _draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    dash: int = 8,
    gap: int = 6,
) -> None:
    if start[0] == end[0]:
        step = 1 if end[1] >= start[1] else -1
        for pos in range(start[1], end[1], step * (dash + gap)):
            seg_end = pos + step * dash
            if step > 0:
                seg_end = min(seg_end, end[1])
            else:
                seg_end = max(seg_end, end[1])
            draw.line((start[0], pos, end[0], seg_end), fill=color, width=width)
        return

    step = 1 if end[0] >= start[0] else -1
    for pos in range(start[0], end[0], step * (dash + gap)):
        seg_end = pos + step * dash
        if step > 0:
            seg_end = min(seg_end, end[0])
        else:
            seg_end = max(seg_end, end[0])
        draw.line((pos, start[1], seg_end, end[1]), fill=color, width=width)


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    for idx in range(len(points) - 1):
        _draw_dashed_segment(draw, points[idx], points[idx + 1], color, width)
    _draw_arrow_head(draw, points[-2], points[-1], color)


_N_TME = 5  # type, state, vas, O2, glu


def _compute_layout(
    tile_size: int,
    panel_gap: int,
    header_height: int,
) -> dict[str, object]:
    ref_tile_size = int(tile_size)
    tme_tile_size = ref_tile_size  # issues 2: TME same size as ref H&E
    cnn_width = max(38, int(round(tile_size * 0.62)))
    cnn_height = tme_tile_size
    attention_width = max(60, int(round(tile_size * 0.82)))
    attention_height = max(42, int(round(tile_size * 0.6)))
    denoiser_width = max(96, int(round(tile_size * 1.15)))
    denoiser_height = max(74, int(round(tile_size * 0.9)))
    vae_width = max(44, int(round(tile_size * 0.5)))
    vae_height = max(42, int(round(tile_size * 0.5)))
    latent_width = max(58, int(round(tile_size * 0.74)))
    latent_height = max(30, int(round(tile_size * 0.38)))
    margin = panel_gap
    text_gap = 18

    top_y = margin + header_height
    left_x = margin
    ref_box = (left_x, top_y, left_x + ref_tile_size, top_y + ref_tile_size)

    tme_y = ref_box[3] + 26
    tme_gap = max(10, panel_gap // 2)
    tme_boxes = [
        (
            left_x,
            tme_y + index * (tme_tile_size + tme_gap),
            left_x + tme_tile_size,
            tme_y + index * (tme_tile_size + tme_gap) + tme_tile_size,
        )
        for index in range(_N_TME)
    ]

    cnn_x = tme_boxes[0][2] + 22
    cnn_top = tme_y + 1
    cnn_gap = tme_gap
    cnn_boxes = []
    for index in range(_N_TME):
        y0 = cnn_top + index * (cnn_height + cnn_gap)
        cnn_boxes.append((cnn_x, y0, cnn_x + cnn_width, y0 + cnn_height))

    uni_box = (
        ref_box[2] + 18,
        ref_box[1] + ref_tile_size // 2 - 20,
        ref_box[2] + 18 + 60,
        ref_box[1] + ref_tile_size // 2 + 20,
    )

    # issue 5: +32 gap so bus_x != attention_box[0], arrow points into left side
    cnn_to_attn_gap = 32
    cnn_mid_y = (cnn_boxes[0][1] + cnn_boxes[-1][3]) // 2
    attention_box = (
        cnn_x + cnn_width + cnn_to_attn_gap,
        cnn_mid_y - attention_height // 2,
        cnn_x + cnn_width + cnn_to_attn_gap + attention_width,
        cnn_mid_y + attention_height // 2,
    )

    # denoiser vertically centered on attention block
    attn_mid_y = (attention_box[1] + attention_box[3]) // 2
    denoiser_box = (
        attention_box[2] + 18,
        attn_mid_y - denoiser_height // 2,
        attention_box[2] + 18 + denoiser_width,
        attn_mid_y + denoiser_height // 2,
    )
    noisy_box = (
        denoiser_box[0] + 16,
        denoiser_box[1] - 22,
        denoiser_box[0] + 16 + 88,
        denoiser_box[1] - 2,
    )
    latent_box = (
        denoiser_box[2] + 12,
        attn_mid_y - latent_height,
        denoiser_box[2] + 12 + latent_width,
        attn_mid_y,
    )
    vae_box = (
        latent_box[0] + max(4, (latent_width - vae_width) // 2),
        latent_box[3] + 18,
        latent_box[0] + max(4, (latent_width - vae_width) // 2) + vae_width,
        latent_box[3] + 18 + vae_height,
    )
    generated_box = (
        vae_box[0] - (ref_tile_size - vae_width) // 2,
        vae_box[3] + 20,
        vae_box[0] - (ref_tile_size - vae_width) // 2 + ref_tile_size,
        vae_box[3] + 20 + ref_tile_size,
    )

    width = max(generated_box[2], latent_box[2]) + margin

    # issue 4: loss arrow ends at attention right side midpoint
    loss_y = max(attention_box[3], generated_box[3]) + 20
    loss_points = [
        ((generated_box[0] + generated_box[2]) // 2, generated_box[3] + 22),
        ((generated_box[0] + generated_box[2]) // 2, loss_y),
        (attention_box[2], loss_y),
        (attention_box[2], attn_mid_y),
    ]

    height = max(
        loss_y + 28 + margin,
        tme_boxes[-1][3] + text_gap + margin,
        cnn_boxes[-1][3] + margin,
        attention_box[3] + margin,
    )

    return {
        "ref_box": ref_box,
        "tme_boxes": tme_boxes,
        "cnn_boxes": cnn_boxes,
        "uni_box": uni_box,
        "attention_box": attention_box,
        "denoiser_box": denoiser_box,
        "noisy_box": noisy_box,
        "latent_box": latent_box,
        "vae_box": vae_box,
        "generated_box": generated_box,
        "width": width,
        "height": height,
        "margin": margin,
        "loss_points": loss_points,
    }


def render_stage2_training_figure(
    assets: Stage2FigureAssets,
    tile_size: int = 88,
    panel_gap: int = 16,
    header_height: int = 34,
    scale_bar_label: str = "",
) -> Image.Image:
    layout = _compute_layout(tile_size, panel_gap, header_height)
    ref_box = layout["ref_box"]
    tme_boxes = layout["tme_boxes"]
    cnn_boxes = layout["cnn_boxes"]
    uni_box = layout["uni_box"]
    attention_box = layout["attention_box"]
    denoiser_box = layout["denoiser_box"]
    noisy_box = layout["noisy_box"]
    latent_box = layout["latent_box"]
    vae_box = layout["vae_box"]
    generated_box = layout["generated_box"]
    width = layout["width"]
    height = layout["height"]
    margin = layout["margin"]

    ref_tile_size = ref_box[2] - ref_box[0]
    tme_tile_size = tme_boxes[0][2] - tme_boxes[0][0]
    denoiser_height = denoiser_box[3] - denoiser_box[1]

    ref_tile = _fit_square_with_resample(
        assets.reference_he_path,
        ref_tile_size,
        Image.Resampling.LANCZOS,
    )
    generated_tile = _fit_square_with_resample(
        assets.generated_he_path,
        ref_tile_size,
        Image.Resampling.LANCZOS,
    )
    tme_tiles = [
        ("cell type", _fit_square(assets.cell_type_path, tme_tile_size)),
        ("cell state", _fit_square(assets.cell_state_path, tme_tile_size)),
        ("vasculature", _fit_square(assets.vasculature_path, tme_tile_size)),
        ("O2", _fit_square(assets.oxygen_path, tme_tile_size)),
        ("glucose", _fit_square(assets.glucose_path, tme_tile_size)),
    ]

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    panel_box = (margin // 2, margin // 2, width - margin // 2, height - margin // 2)
    panel = Image.new("RGB", (panel_box[2] - panel_box[0], panel_box[3] - panel_box[1]), PANEL)
    canvas.paste(panel, (panel_box[0], panel_box[1]))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(panel_box, radius=18, outline=(220, 214, 206), width=2)

    title_font = _load_font(max(22, int(tile_size * 0.28)), bold=True)
    block_font = _load_font(max(12, int(tile_size * 0.13)), bold=True)
    body_font = _load_font(max(9, int(tile_size * 0.10)))
    label_font = _load_font(max(12, int(tile_size * 0.13)), bold=True)
    tme_label_font = _load_font(max(10, int(tile_size * 0.11)), bold=True)
    latent_font = _load_font(max(11, int(tile_size * 0.13)), bold=True)
    scale_font = _load_font(max(9, int(tile_size * 0.10)))

    draw.text(
        (margin, margin + 6),
        "Stage 2 training",
        font=title_font,
        fill=INK,
    )

    _draw_plain_he_tile(
        canvas,
        draw,
        ref_tile,
        ref_box,
        "ref H&E",
        label_font,
        scale_bar_color=(255, 255, 255),
        scale_bar_label=scale_bar_label,
        scale_bar_label_font=scale_font,
    )
    draw.text((margin, ref_box[3] + 14), "Stage 1 TME examples", font=body_font, fill=MUTED)
    for (label, tile), box in zip(tme_tiles, tme_boxes):
        _draw_tme_tile(canvas, draw, tile, box, label, tme_label_font)

    _draw_block(
        canvas,
        draw,
        uni_box,
        "UNI",
        "frozen\nencoder",
        frozen=True,
        title_font=block_font,
        subtitle_font=body_font,
    )
    for index, box in enumerate(cnn_boxes, start=1):
        _draw_block(
            canvas,
            draw,
            box,
            f"CNN{index}",
            None,
            frozen=False,
            title_font=block_font,
            subtitle_font=body_font,
        )
    _draw_block(
        canvas,
        draw,
        attention_box,
        "Attention",
        "fusion",
        frozen=False,
        title_font=block_font,
        subtitle_font=body_font,
    )
    _draw_block(
        canvas,
        draw,
        denoiser_box,
        "PixCell denoiser",
        "ControlNet +\nbase transformer",
        frozen=True,
        title_font=block_font,
        subtitle_font=body_font,
    )
    _draw_latent_text(draw, noisy_box, ["noisy latent Z_t"], latent_font)
    _draw_latent_text(draw, latent_box, ["Denoised", "latent Z"], latent_font)
    _draw_block(
        canvas,
        draw,
        vae_box,
        "SD3.5",
        "VAE",
        frozen=True,
        title_font=block_font,
        subtitle_font=body_font,
    )
    _draw_plain_he_tile(
        canvas,
        draw,
        generated_tile,
        generated_box,
        "gen H&E",
        label_font,
        scale_bar_color=(255, 255, 255),
        scale_bar_label=scale_bar_label,
        scale_bar_label_font=scale_font,
    )

    _draw_polyline(
        draw,
        [
            (ref_box[2], (ref_box[1] + ref_box[3]) // 2),
            (uni_box[0], (uni_box[1] + uni_box[3]) // 2),
        ],
        LINE,
    )
    uni_arrow_y = (uni_box[1] + uni_box[3]) // 2
    draw.text(
        ((uni_box[2] + denoiser_box[0]) // 2, uni_arrow_y - 6),
        "UNI latent",
        font=body_font,
        fill=MUTED,
        anchor="mb",
    )
    _draw_polyline(
        draw,
        [
            (uni_box[2], uni_arrow_y),
            (denoiser_box[0], uni_arrow_y),
        ],
        LINE,
    )
    _draw_polyline(
        draw,
        [
            ((noisy_box[0] + noisy_box[2]) // 2, noisy_box[3]),
            ((noisy_box[0] + noisy_box[2]) // 2, denoiser_box[1]),
        ],
        LINE,
    )
    for tile_box, cnn_box in zip(tme_boxes, cnn_boxes):
        elbow_x = tile_box[2] + max(6, (cnn_box[0] - tile_box[2]) // 2)
        _draw_polyline(
            draw,
            [
                (tile_box[2], (tile_box[1] + tile_box[3]) // 2),
                (elbow_x, (tile_box[1] + tile_box[3]) // 2),
                (elbow_x, (cnn_box[1] + cnn_box[3]) // 2),
                (cnn_box[0], (cnn_box[1] + cnn_box[3]) // 2),
            ],
            TRAINABLE_BORDER,
        )

    # issue 5: bus at cnn right+16, horizontal segment to attention left
    bus_x = cnn_boxes[0][2] + 16
    top_bus_y = (cnn_boxes[0][1] + cnn_boxes[0][3]) // 2
    bottom_bus_y = (cnn_boxes[-1][1] + cnn_boxes[-1][3]) // 2
    for cnn_box in cnn_boxes:
        y_mid = (cnn_box[1] + cnn_box[3]) // 2
        draw.line((cnn_box[2], y_mid, bus_x, y_mid), fill=TRAINABLE_BORDER, width=2)
    draw.line((bus_x, top_bus_y, bus_x, bottom_bus_y), fill=TRAINABLE_BORDER, width=2)
    _draw_polyline(
        draw,
        [
            (bus_x, (top_bus_y + bottom_bus_y) // 2),
            (attention_box[0], (attention_box[1] + attention_box[3]) // 2),
        ],
        TRAINABLE_BORDER,
    )
    # conditioning latent: attention top → up → right into denoiser top quarter
    attn_cx = (attention_box[0] + attention_box[2]) // 2
    cond_y = denoiser_box[1] + max(6, denoiser_height // 4)
    _draw_polyline(
        draw,
        [
            (attn_cx, attention_box[1]),
            (attn_cx, cond_y),
            (denoiser_box[0], cond_y),
        ],
        TRAINABLE_BORDER,
    )
    draw.text(
        ((attention_box[2] + denoiser_box[0]) // 2, cond_y - 12),
        "conditioning latent",
        font=body_font,
        fill=TRAINABLE_BORDER,
        anchor="mb",
    )
    _denoiser_exit_y = (denoiser_box[1] + denoiser_box[3]) // 2
    _vae_cx = (vae_box[0] + vae_box[2]) // 2
    _draw_polyline(
        draw,
        [
            (denoiser_box[2], _denoiser_exit_y),
            (_vae_cx, _denoiser_exit_y),
            (_vae_cx, vae_box[1]),
        ],
        LINE,
    )
    _draw_polyline(
        draw,
        [
            ((vae_box[0] + vae_box[2]) // 2, vae_box[3]),
            ((vae_box[0] + vae_box[2]) // 2, generated_box[1]),
        ],
        LINE,
    )

    loss_points = layout["loss_points"]
    loss_y = loss_points[1][1]
    _draw_dashed_polyline(draw, loss_points, LOSS, width=2)
    draw.text(
        ((generated_box[0] + attention_box[2]) // 2, loss_y + 12),
        "loss updates CNNs + attention only",
        font=body_font,
        fill=LOSS,
        anchor="mm",
    )

    return canvas