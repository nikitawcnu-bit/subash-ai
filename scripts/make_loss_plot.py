"""Рисует кривую обучения мини-GPT по логу train_gpt.py (только Pillow).

Запуск: python scripts/make_loss_plot.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "checkpoints" / "train_gpt.log"
OUT = ROOT / "assets" / "gpt_loss.png"

W, H = 860, 500
LEFT, RIGHT, TOP, BOTTOM = 78, 24, 66, 78
TRAIN_COLOR = (52, 112, 200)
VAL_COLOR = (224, 122, 45)
GRID_COLOR = (232, 232, 232)
AXIS_COLOR = (90, 90, 90)


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_log(path: Path):
    train, val = [], []
    if not path.exists():
        return train, val
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        m = re.search(r"шаг\s+(\d+)\s+\|\s+loss\s+([\d.]+)", line)
        if m:
            train.append((int(m.group(1)), float(m.group(2))))
        m = re.search(r"val loss\s+([\d.]+)", line)
        if m:
            val.append((train[-1][0] if train else 0, float(m.group(1))))
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Кривая обучения по логу train_gpt.py")
    parser.add_argument("--log", default=str(LOG))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--title", default="Subash AI: обучение мини-GPT на сказках Пушкина")
    args = parser.parse_args()
    log, out = Path(args.log), Path(args.out)

    train, val = parse_log(log)
    if not train:
        raise SystemExit(f"Нет данных в {log}")

    max_step = max(train[-1][0], 1)
    max_loss = 5.0
    min_loss = 0.0

    def px(step: float) -> float:
        return LEFT + step / max_step * (W - LEFT - RIGHT)

    def py(loss: float) -> float:
        return TOP + (max_loss - loss) / (max_loss - min_loss) * (H - TOP - BOTTOM)

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(17)
    small = load_font(14)
    tiny = load_font(13)

    for value in range(int(min_loss), int(max_loss) + 1):
        y = py(value)
        draw.line((LEFT, y, W - RIGHT, y), fill=GRID_COLOR)
        draw.text((LEFT - 8, y - 7), str(value), fill=AXIS_COLOR, font=small, anchor="ra")

    tick_step = max(1, max_step // 6)
    tick_step = int(round(tick_step / 100.0) * 100) or tick_step
    for step in range(0, max_step + 1, tick_step):
        x = px(step)
        draw.line((x, TOP, x, H - BOTTOM), fill=GRID_COLOR)
        draw.text((x, H - BOTTOM + 8), str(step), fill=AXIS_COLOR, font=small, anchor="ma")

    draw.line((LEFT, H - BOTTOM, W - RIGHT, H - BOTTOM), fill=AXIS_COLOR, width=1)
    draw.line((LEFT, TOP, LEFT, H - BOTTOM), fill=AXIS_COLOR, width=1)

    if len(train) > 1:
        points = [(px(s), py(v)) for s, v in train]
        draw.line(points, fill=TRAIN_COLOR, width=2, joint="curve")
    for step, value in val:
        x, y = px(step), py(value)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=VAL_COLOR, outline="white")
    if len(val) > 1:
        draw.line([(px(s), py(v)) for s, v in val], fill=VAL_COLOR, width=2)

    draw.text((LEFT, 22), args.title,
              fill=(30, 30, 30), font=font)
    draw.text((LEFT - 8, TOP - 20), "loss", fill=AXIS_COLOR, font=tiny, anchor="ra")
    draw.text((W - RIGHT, H - BOTTOM + 26), "шаг", fill=AXIS_COLOR, font=tiny, anchor="ra")
    legend_y = H - 24
    draw.line((LEFT, legend_y, LEFT + 26, legend_y), fill=TRAIN_COLOR, width=3)
    draw.text((LEFT + 34, legend_y - 8), "train (сглаженный)", fill=(60, 60, 60), font=small)
    draw.ellipse((LEFT + 210, legend_y - 4, LEFT + 218, legend_y + 4), fill=VAL_COLOR)
    draw.text((LEFT + 226, legend_y - 8), "val", fill=(60, 60, 60), font=small)

    out.parent.mkdir(exist_ok=True)
    img.save(out)
    print(f"Сохранено: {out} ({len(train)} точек train, {len(val)} точек val)")


if __name__ == "__main__":
    main()
