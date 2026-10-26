"""Обучение код-модели Subash AI на корпусе Python-кода.

Запуск: python examples/train_code.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    command = [
        sys.executable, "-u", str(ROOT / "examples" / "train_gpt.py"),
        "--data", str(ROOT / "data" / "code.txt"),
        "--out", str(ROOT / "checkpoints" / "gpt_code.npz"),
        "--samples-file", str(ROOT / "checkpoints" / "samples_code.txt"),
        "--prompt", "def ",
        "--steps", "4000",
        "--batch-size", "12",
        "--block-size", "96",
        "--n-layer", "2",
        "--n-head", "4",
        "--n-embd", "160",
        "--dropout", "0.05",
        "--lr", "3e-3",
        "--warmup", "200",
        "--eval-every", "250",
        "--sample-every", "500",
        "--sample-tokens", "180",
        "--save-every", "500",
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
