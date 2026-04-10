#!/usr/bin/env python3
"""Render a PNG Open Graph cover for GitHub social preview.

The repository keeps the editable source design in `docs/assets/open-graph-cover.svg`.
This script generates a matching PNG asset using Pillow so the result can be
uploaded directly to GitHub repository settings on machines where SVG rendering
tools are inconsistent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 640
BG_TOP = (7, 11, 20)
BG_BOTTOM = (18, 26, 48)
FRAME = (42, 53, 80)
TEXT_PRIMARY = (240, 244, 255)
TEXT_SECONDARY = (158, 176, 216)
TEXT_MUTED = (92, 109, 148)
ACCENT_LEFT = (61, 139, 253)
ACCENT_RIGHT = (88, 214, 164)
PANEL_BG = (12, 16, 28)
PANEL_BORDER = (47, 61, 92)
PANEL_TITLE = (200, 212, 240)
PANEL_TEXT = (216, 227, 255)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_input = root / "docs" / "assets" / "open-graph-cover.svg"
    default_output = root / "docs" / "assets" / "open-graph-cover.png"

    parser = argparse.ArgumentParser(
        description="Render the Open Graph cover to a PNG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination PNG path.",
    )
    return parser.parse_args()


def _load_font(candidates: list[str], size: int, fallback: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/Library/Fonts"),
    ]
    for directory in font_dirs:
        for name in candidates:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
    if fallback:
        return ImageFont.load_default()
    raise SystemExit(f"Unable to locate a font from: {', '.join(candidates)}")


def _draw_vertical_gradient(draw: ImageDraw.ImageDraw) -> None:
    for y in range(HEIGHT):
        ratio = y / max(1, HEIGHT - 1)
        color = tuple(
            int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * ratio) for i in range(3)
        )
        draw.line((0, y, WIDTH, y), fill=color)


def _draw_grid(draw: ImageDraw.ImageDraw) -> None:
    grid = (168, 184, 224)
    alpha = 18
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(overlay)
    for y in (160, 320, 480):
        grid_draw.line((0, y, WIDTH, y), fill=(*grid, alpha), width=1)
    for x in (320, 640, 960):
        grid_draw.line((x, 0, x, HEIGHT), fill=(*grid, alpha), width=1)
    return overlay


def render_png(output_path: Path) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), BG_TOP)
    draw = ImageDraw.Draw(image)
    _draw_vertical_gradient(draw)
    image.alpha_composite(_draw_grid(draw))

    title_font = _load_font(["segoeuib.ttf", "arialbd.ttf"], 56)
    body_font = _load_font(["segoeui.ttf", "arial.ttf"], 28)
    meta_font = _load_font(["segoeuib.ttf", "arialbd.ttf"], 22)
    footer_font = _load_font(["segoeui.ttf", "arial.ttf"], 20, fallback=True)
    mono_font = _load_font(["consola.ttf", "CascadiaMono.ttf"], 18, fallback=True)
    mono_small = _load_font(["consola.ttf", "CascadiaMono.ttf"], 16, fallback=True)

    draw.rounded_rectangle((96, 96, 1184, 544), radius=28, outline=FRAME, width=2)
    draw.rounded_rectangle((96, 96, 102, 544), radius=2, fill=ACCENT_LEFT)

    draw.text((132, 140), "Memory Path Engine", font=title_font, fill=TEXT_PRIMARY)
    draw.text(
        (132, 214),
        "Replayable evidence paths for agent memory - not only flat top-k chunks.",
        font=body_font,
        fill=TEXT_SECONDARY,
    )
    draw.text(
        (132, 270),
        "Research prototype | Python | MIT",
        font=meta_font,
        fill=ACCENT_RIGHT,
    )

    draw.rounded_rectangle((132, 352, 1148, 484), radius=14, fill=PANEL_BG, outline=PANEL_BORDER, width=1)
    draw.text(
        (156, 382),
        "$ python -m memory_engine.demo --scenario runbook",
        font=mono_font,
        fill=TEXT_SECONDARY,
    )
    draw.text(
        (156, 420),
        "REPLAY PATH  ->  node id  |  score  |  via=edge_type",
        font=mono_font,
        fill=PANEL_TITLE,
    )
    draw.text(
        (156, 454),
        "Bundled runbook + contract demos | multiple retrieval baselines in one repo",
        font=mono_small,
        fill=PANEL_TEXT,
    )

    draw.text(
        (132, 586),
        "github.com/ly85206559/memory-path-engine",
        font=footer_font,
        fill=TEXT_MUTED,
    )
    draw.text(
        (1020, 586),
        "1280x640 | GitHub Social Preview",
        font=footer_font,
        fill=(74, 90, 122),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG")


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    render_png(output_path)
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
