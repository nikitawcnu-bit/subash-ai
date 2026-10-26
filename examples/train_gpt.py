"""Обучение мини-GPT на посимвольном корпусе (по умолчанию — сказки Пушкина).

Запуск:
    python examples/train_gpt.py
    python examples/train_gpt.py --steps 4000 --n-layer 3 --n-embd 160
    python examples/train_gpt.py --quick      # быстрый смоук-тест

Скрипт сохраняет чекпойнт с токенизатором в checkpoints/gpt_pushkin.npz
и примеры генерации в checkpoints/samples.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import data, load_checkpoint, optim, save_checkpoint  # noqa: E402
from subash.models import TinyGPT  # noqa: E402
from subash.utils import format_time, set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model, dataset, batch_size: int, block_size: int, rng) -> float:
    model.eval()
    losses = []
    for _ in range(4):
        x, y = dataset.get_batch("val", batch_size, block_size, rng)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def sample(model, tokenizer, prompt: str, tokens: int, temperature: float,
           top_k: int | None) -> str:
    ids = tokenizer.encode_safe(prompt)
    if len(ids) == 0:
        ids = tokenizer.encode_safe(" ")
    model.eval()
    out = model.generate(ids, max_new_tokens=tokens, temperature=temperature, top_k=top_k)
    model.train()
    return tokenizer.decode(out[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Subash AI: мини-GPT на сказках Пушкина")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "pushkin.txt"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-from", type=str, default=None,
                        help="продолжить обучение с чекпойнта")
    parser.add_argument("--quick", action="store_true", help="маленькая модель и 60 шагов")
    args = parser.parse_args()

    if args.quick:
        args.steps = 60
        args.n_layer, args.n_embd, args.n_head = 1, 32, 2
        args.batch_size, args.block_size = 8, 32

    set_seed(args.seed)
    dataset = data.TextDataset.from_file(args.data)
    tokenizer = dataset.tokenizer
    print(f"Корпус: {len(dataset.ids):,} символов, словарь {tokenizer.vocab_size}, "
          f"train {len(dataset.train_ids):,} / val {len(dataset.val_ids):,}")

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    print(f"Модель: {model}")
    print(f"Параметров: {model.num_parameters():,}")

    if args.init_from:
        old_model, _ = load_checkpoint(args.init_from)
        if old_model.config["args"] != model.config["args"]:
            raise SystemExit("Архитектура чекпойнта не совпадает с текущими флагами модели")
        old_params = dict(old_model.named_parameters())
        for name, param in model.named_parameters():
            param.data[...] = old_params[name].data
        print(f"Веса загружены из {args.init_from}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    prompt = "Три девицы под окном"
    samples: list[str] = []
    start = time.perf_counter()
    smooth_loss = None

    for step in range(args.steps):
        optimizer.lr = optim.cosine_lr(step, args.steps, args.lr, warmup=args.warmup,
                                       min_lr=args.lr * 0.05)
        x, y = dataset.get_batch("train", args.batch_size, args.block_size, rng)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = optim.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        smooth_loss = loss.item() if smooth_loss is None else 0.98 * smooth_loss + 0.02 * loss.item()
        if step % 20 == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - start
            speed = (step + 1) / max(elapsed, 1e-9)
            print(f"шаг {step:4d} | loss {smooth_loss:.4f} | grad {grad_norm:.2f} | "
                  f"lr {optimizer.lr:.2e} | {speed:.2f} шаг/с | {format_time(elapsed)}")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            val_loss = evaluate(model, dataset, args.batch_size, args.block_size, rng)
            print(f"           val loss {val_loss:.4f} | perplexity {np.exp(val_loss):.1f}")

        if (step + 1) % args.sample_every == 0 or step == args.steps - 1:
            text = sample(model, tokenizer, prompt, tokens=110, temperature=0.85, top_k=40)
            print("           ---\n" + "\n".join("           " + line for line in text.splitlines())
                  + "\n           ---")
            samples.append(f"=== шаг {step + 1} ===\n{text}\n")

    elapsed = time.perf_counter() - start
    val_loss = evaluate(model, dataset, args.batch_size, args.block_size, rng)
    print(f"\nГотово за {format_time(elapsed)}, val loss {val_loss:.4f}, "
          f"perplexity {np.exp(val_loss):.1f}")

    ckpt = save_checkpoint(model, ROOT / "checkpoints" / "gpt_pushkin.npz", tokenizer)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "checkpoints" / "samples.txt").write_text("\n".join(samples), encoding="utf-8")
    print(f"Чекпойнт: {ckpt}")


if __name__ == "__main__":
    main()
