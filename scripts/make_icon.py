"""Рисует иконку Subash AI (assets/subash.ico + assets/icon.png)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SIZE = 512


def font(size: int):
    for name in ("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (10, 10, SIZE - 10, SIZE - 10), radius=112,
        fill=(11, 18, 32, 255), outline=(76, 194, 255, 255), width=10,
    )

    # Лёгкие «нейронные» связи на фоне.
    nodes = [(128, 372), (196, 300), (196, 444), (300, 372)]
    for a in nodes[:3]:
        for b in nodes[3:]:
            draw.line((a, b), fill=(38, 78, 122, 255), width=5)
    for x, y in nodes:
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(76, 194, 255, 255))

    letter_font = font(300)
    text = "S"
    bbox = draw.textbbox((0, 0), text, font=letter_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text(
        ((SIZE - width) / 2 - bbox[0], (SIZE - height) / 2 - bbox[1] - 26),
        text, font=letter_font, fill=(230, 247, 255, 255),
    )
    draw.ellipse((SIZE - 108, 74, SIZE - 62, 120), fill=(122, 212, 255, 255))

    ASSETS.mkdir(exist_ok=True)
    png = ASSETS / "icon.png"
    img.save(png)
    ico = ASSETS / "subash.ico"
    img.save(ico, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                      (64, 64), (128, 128), (256, 256)])
    print(f"Сохранено: {png}, {ico}")


if __name__ == "__main__":
    main()
