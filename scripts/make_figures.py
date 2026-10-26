"""Рисует картинку «данные + границы классов» для README (только Pillow).

Запуск: python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subash import Tensor, load_checkpoint  # noqa: E402
from subash.data import make_spiral  # noqa: E402

EXTENT = 1.5
SIZE = 560
GRID = 280
PADDING = 24
LEGEND_H = 54

CLASS_COLORS = [(222, 74, 74), (58, 124, 210), (52, 168, 116)]
BG_COLORS = [(250, 222, 222), (220, 232, 248), (218, 242, 230)]


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_pixel(x: float, y: float) -> tuple[float, float]:
    px = PADDING + (x + EXTENT) / (2 * EXTENT) * SIZE
    py = PADDING + (EXTENT - y) / (2 * EXTENT) * SIZE
    return px, py


def main() -> None:
    model, _ = load_checkpoint(ROOT / "checkpoints" / "mlp_spiral.npz")
    model.eval()
    X, y = make_spiral(n_per_class=250, noise=0.05, seed=0)

    xs = np.linspace(-EXTENT, EXTENT, GRID)
    ys = np.linspace(EXTENT, -EXTENT, GRID)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    predictions = model(Tensor(points)).data.argmax(axis=1).reshape(GRID, GRID)

    palette = Image.new("RGB", (GRID, GRID))
    pixels = palette.load()
    for row in range(GRID):
        for col in range(GRID):
            pixels[col, row] = BG_COLORS[int(predictions[row, col])]
    palette = palette.resize((SIZE, SIZE), Image.NEAREST)

    canvas = Image.new("RGB", (SIZE + 2 * PADDING, SIZE + 2 * PADDING + LEGEND_H), "white")
    canvas.paste(palette, (PADDING, PADDING))
    draw = ImageDraw.Draw(canvas)

    for (x, yv), label in zip(X, y):
        px, py = to_pixel(float(x), float(yv))
        r = 3.6
        draw.ellipse((px - r, py - r, px + r, py + r), fill=CLASS_COLORS[int(label)],
                     outline="white", width=1)

    font = load_font(17)
    small = load_font(15)
    draw.text((PADDING, 6), "Subash AI: MLP на спирали, val accuracy 97%", fill=(30, 30, 30),
              font=font)
    legend_y = SIZE + PADDING + 12
    x_offset = PADDING
    for i, color in enumerate(CLASS_COLORS):
        draw.rectangle((x_offset, legend_y, x_offset + 14, legend_y + 14), fill=color)
        draw.text((x_offset + 20, legend_y - 1), f"класс {i}", fill=(40, 40, 40), font=small)
        x_offset += 110
    draw.text((PADDING, legend_y + 22), "фон — предсказание сети, точки — обучающие данные",
              fill=(90, 90, 90), font=small)

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    out = assets / "spiral.png"
    canvas.save(out)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
