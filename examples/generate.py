"""Генерация текста или кода обученным мини-GPT.

Примеры:
    python examples/generate.py --prompt "def " --model code
    python examples/generate.py --model pushkin --prompt "Три девицы под окном"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import load_checkpoint  # noqa: E402
from subash.utils import set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "code": ("gpt_code.npz", "def "),
    "pushkin": ("gpt_pushkin.npz", "Три девицы под окном"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Subash AI: генерация текста или кода")
    parser.add_argument("--model", choices=sorted(DEFAULTS), default="code")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="путь к чекпойнту (по умолчанию — из --model)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="начало фразы/кода (по умолчанию — из --model)")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    default_file, default_prompt = DEFAULTS[args.model]
    path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / default_file
    if not path.exists():
        raise SystemExit(f"Нет чекпойнта {path}. Сначала обучите модель: "
                         f"python examples/train_code.py")
    prompt = args.prompt if args.prompt is not None else default_prompt

    model, tokenizer = load_checkpoint(path)
    # сид ставим после загрузки: создание модели тоже тратит случайные числа
    if args.seed is not None:
        set_seed(args.seed)
    ids = tokenizer.encode_safe(prompt)
    if len(ids) == 0:
        raise SystemExit("В промпте нет символов из словаря модели")

    output = model.generate(ids, max_new_tokens=args.tokens,
                            temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p)
    print(tokenizer.decode(output[0]))


if __name__ == "__main__":
    main()
