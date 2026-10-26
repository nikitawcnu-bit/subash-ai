"""Генерация текста обученным мини-GPT.

Примеры:
    python examples/generate.py --prompt "Три девицы под окном" --tokens 250
    python examples/generate.py --prompt "Князь у синя моря" --temperature 0.7 --top-k 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import load_checkpoint  # noqa: E402
from subash.utils import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Subash AI: генерация текста")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "gpt_pushkin.npz"))
    parser.add_argument("--prompt", type=str, default="Три девицы под окном")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.ckpt)
    if not path.exists():
        raise SystemExit(f"Нет чекпойнта {path}. Сначала обучите модель: "
                         f"python examples/train_gpt.py")

    model, tokenizer = load_checkpoint(path)
    # сид ставим после загрузки: создание модели тоже тратит случайные числа
    if args.seed is not None:
        set_seed(args.seed)
    ids = tokenizer.encode_safe(args.prompt)
    if len(ids) == 0:
        raise SystemExit("В промпте нет символов из словаря модели")

    output = model.generate(ids, max_new_tokens=args.tokens,
                            temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p)
    print(tokenizer.decode(output[0]))


if __name__ == "__main__":
    main()
