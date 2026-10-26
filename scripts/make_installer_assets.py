"""Готовит картинки для установщика NSIS (164x314 и 150x57, 24-bit BMP)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "installer" / "assets"
ICON = ROOT / "assets" / "subash.ico"
VERSION = "0.2.0"

BG_TOP = (11, 18, 32)
BG_BOTTOM = (14, 28, 48)
ACCENT = (76, 194, 255)
MUTED = (138, 160, 189)
WHITE = (230, 237, 247)


def font(size: int, bold: bool = False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def gradient(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM))
        draw.line((0, y, width, y), fill=color)
    return img


def load_icon(size: int) -> Image.Image:
    icon = Image.open(ICON).convert("RGBA")
    icon = icon.resize((size, size), Image.LANCZOS)
    return icon


def centered(draw: ImageDraw.ImageDraw, text: str, fnt, width: int, y: int, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text(((width - (box[2] - box[0])) / 2 - box[0], y), text, font=fnt, fill=fill)


def make_side() -> None:
    width, height = 164, 314
    img = gradient(width, height)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(34, 50, 74))
    draw.rectangle((0, 0, 3, height), fill=ACCENT)

    icon = load_icon(104)
    img.paste(icon, ((width - 104) // 2, 30), icon)

    centered(draw, "SUBASH AI", font(18, True), width, 154, ACCENT)
    centered(draw, "мини-GPT на NumPy", font(10), width, 182, MUTED)
    draw.line((26, 210, width - 26, 210), fill=(35, 51, 75))
    centered(draw, "нейросеть с нуля", font(9), width, 224, MUTED)
    centered(draw, f"v{VERSION}", font(9, True), width, 284, WHITE)
    img.save(OUT / "side.bmp", "BMP")


def make_header() -> None:
    width, height = 150, 57
    img = gradient(width, height)
    draw = ImageDraw.Draw(img)
    icon = load_icon(38)
    img.paste(icon, (8, 9), icon)
    draw.text((54, 7), "SUBASH AI", font=font(15, True), fill=ACCENT)
    draw.text((55, 29), "мини-GPT на NumPy", font=font(9), fill=MUTED)
    draw.rectangle((0, height - 2, width, height - 1), fill=ACCENT)
    img.save(OUT / "header.bmp", "BMP")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_side()
    make_header()
    for name in ("side.bmp", "header.bmp"):
        img = Image.open(OUT / name)
        print(name, img.size, img.mode)


if __name__ == "__main__":
    main()
