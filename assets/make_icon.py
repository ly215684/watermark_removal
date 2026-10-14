# -*- coding: utf-8 -*-
"""Generate the application icon (multi-size .ico) with Pillow.

This runs at build time (see watermark_removal.spec), so no binary icon needs to
be committed. Pillow is already a build dependency. It draws a simple, crisp
"watermark eraser" glyph and exports a 16..256 px .ico.

Manual usage:  python assets/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
TOP = (20, 184, 166)      # teal
BOTTOM = (79, 70, 229)    # indigo


def _gradient(size: int, top, bottom):
    from PIL import Image

    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        col = (
            int(top[0] + (bottom[0] - top[0]) * t),
            int(top[1] + (bottom[1] - top[1]) * t),
            int(top[2] + (bottom[2] - top[2]) * t),
        )
        for x in range(size):
            px[x, y] = col
    return img


def _master(size: int = 256):
    from PIL import Image, ImageDraw

    grad = _gradient(size, TOP, BOTTOM)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(grad, (0, 0))

    # rounded-square clip
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=255)
    canvas.putalpha(mask)

    draw = ImageDraw.Draw(canvas)
    m = size * 0.18          # frame margin
    fw = size * 0.64         # frame width
    fh = size * 0.52         # frame height
    rect = [m, size * 0.16, m + fw, size * 0.16 + fh]

    # image frame (white border, translucent fill)
    draw.rounded_rectangle(rect, radius=size * 0.04,
                           fill=(255, 255, 255, 60), outline=(255, 255, 255, 235),
                           width=max(2, int(size * 0.02)))
    # "mountains" inside the frame to suggest a photo
    peak = [rect[0] + fw * 0.15, rect[3], rect[0] + fw * 0.42, rect[1] + fh * 0.35,
            rect[0] + fw * 0.62, rect[3]]
    draw.polygon(peak, fill=(255, 255, 255, 200))
    draw.ellipse([rect[0] + fw * 0.62, rect[1] + fh * 0.18,
                  rect[0] + fw * 0.82, rect[1] + fh * 0.42], fill=(255, 255, 255, 200))

    # diagonal watermark band (semi-transparent)
    band_h = size * 0.12
    draw.line([rect[0] - size * 0.02, rect[1] + fh * 0.7,
               rect[2] + size * 0.02, rect[1] + fh * 0.3],
              fill=(255, 90, 90, 150), width=int(band_h))
    # erased portion (gradient clear) on the right side of the band
    draw.line([rect[0] + fw * 0.55, rect[1] + fh * 0.5,
               rect[2] + size * 0.02, rect[1] + fh * 0.3],
              fill=(0, 0, 0, 0), width=int(band_h))

    # eraser block at the leading edge
    e = size * 0.16
    ex, ey = rect[0] + fw * 0.52, rect[1] + fh * 0.42
    draw.rounded_rectangle([ex, ey, ex + e, ey + e * 0.7], radius=size * 0.02,
                           fill=(250, 250, 250, 245), outline=(120, 120, 140, 255),
                           width=max(1, int(size * 0.01)))

    # sparkles
    for sx, sy in [(size * 0.78, size * 0.30), (size * 0.86, size * 0.44), (size * 0.72, size * 0.5)]:
        r = size * 0.03
        draw.polygon([(sx, sy - r), (sx + r * 0.3, sy), (sx, sy + r), (sx - r * 0.3, sy)],
                     fill=(255, 255, 255, 255))
        draw.polygon([(sx - r, sy), (sx, sy - r * 0.3), (sx + r, sy), (sx, sy + r * 0.3)],
                     fill=(255, 255, 255, 255))
    return canvas


def build_ico(dest: str | Path, png_preview: str | Path | None = None) -> Path:
    from PIL import Image

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    master = _master(256)
    if png_preview:
        master.save(png_preview)
    master.save(dest, sizes=ICO_SIZES)
    return dest


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = build_ico(here / "app.ico", png_preview=here / "app.png")
    print(f"Wrote {out}")
