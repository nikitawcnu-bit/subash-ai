"""Интерактивный REPL: продолжаем текст с обученным мини-GPT.

Запуск:
    python examples/chat.py
    python examples/chat.py --ckpt checkpoints/gpt_pushkin.npz --temperature 0.8

Команды:
    /temp 0.7    — температура
    /tokens 200  — длина ответа
    /help        — справка
    /quit        — выход
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Subash AI: интерактивная генерация")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "gpt_pushkin.npz"))
    parser.add_argument("--tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    path = Path(args.ckpt)
    if not path.exists():
        raise SystemExit(f"Нет чекпойнта {path}. Сначала: python examples/train_gpt.py")
    model, tokenizer = load_checkpoint(path)
    model.eval()
    print(f"Subash AI загружен: {model}")
    print("Введите начало фразы. /help — команды, /quit — выход.\n")

    while True:
        try:
            prompt = input("вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in ("/quit", "/exit", "/q"):
            break
        if prompt == "/help":
            print("  /temp 0.7   температура (0 = детерминированно)")
            print("  /tokens 200 длина продолжения")
            print("  /quit       выход")
            continue
        if prompt.startswith("/temp"):
            try:
                args.temperature = float(prompt.split()[1])
                print(f"температура = {args.temperature}")
            except (IndexError, ValueError):
                print("пример: /temp 0.7")
            continue
        if prompt.startswith("/tokens"):
            try:
                args.tokens = int(prompt.split()[1])
                print(f"длина = {args.tokens}")
            except (IndexError, ValueError):
                print("пример: /tokens 200")
            continue

        ids = tokenizer.encode_safe(prompt)
        if len(ids) == 0:
            print("В промпте нет знакомых модели символов, попробуйте другой текст.")
            continue
        out = model.generate(ids, max_new_tokens=args.tokens,
                             temperature=args.temperature, top_k=args.top_k)
        print("\n" + tokenizer.decode(out[0]) + "\n")


if __name__ == "__main__":
    main()
