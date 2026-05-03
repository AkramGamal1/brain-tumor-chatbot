"""Generate Phase 1 image fixtures. Run once: python eval/fixtures/_make_images.py"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


def _make_valid_grayscale(path: Path, size: tuple[int, int] = (224, 224)) -> None:
    img = Image.new("L", size, color=70)
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse((cx - 70, cy - 80, cx + 70, cy + 80), fill=140)
    draw.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=210)
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    img.save(path)


def _make_color_photo(path: Path, size: tuple[int, int] = (224, 224)) -> None:
    rng = random.Random(0)
    img = Image.new("RGB", size, color=(220, 90, 40))
    draw = ImageDraw.Draw(img)
    for _ in range(40):
        x = rng.randint(0, size[0] - 30)
        y = rng.randint(0, size[1] - 30)
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.ellipse((x, y, x + 30, y + 30), fill=c)
    img.save(path, quality=85)


def _make_extreme_aspect(path: Path) -> None:
    img = Image.new("L", (96, 480), color=120)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 50, 76, 430), fill=200)
    img.save(path)


def main() -> None:
    _make_valid_grayscale(OUT / "valid_grayscale.png", (224, 224))
    _make_valid_grayscale(OUT / "valid_grayscale_small.png", (96, 96))
    _make_color_photo(OUT / "color_photo.jpg")
    _make_extreme_aspect(OUT / "extreme_aspect.png")
    print(f"Wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
