"""Обучение MLP на задаче «спираль» + ASCII-карта решений.

Запуск:
    python examples/train_mlp.py
    python examples/train_mlp.py --steps 1200 --hidden 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import Tensor, data, nn, optim, save_checkpoint  # noqa: E402
from subash.models import MLP  # noqa: E402
from subash.utils import count_parameters, format_time, set_seed  # noqa: E402


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    logits = model(Tensor(X))
    return float((logits.data.argmax(axis=1) == y).mean())


def ascii_decision_map(model: MLP, resolution: int = 64, extent: float = 1.5) -> str:
    xs = np.linspace(-extent, extent, resolution)
    ys = np.linspace(extent, -extent, resolution // 2)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    predictions = model(Tensor(points)).data.argmax(axis=1).reshape(grid_y.shape)
    palette = ".+#"
    return "\n".join("".join(palette[p] for p in row) for row in predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Subash AI: MLP на спирали")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-map", action="store_true", help="не печатать ASCII-карту")
    args = parser.parse_args()

    set_seed(args.seed)
    X, y = data.make_spiral(n_per_class=args.samples_per_class, noise=0.05, seed=args.seed)
    X_train, y_train, X_val, y_val = data.train_val_split(X, y, val_fraction=0.2, seed=args.seed)

    model = MLP([2, args.hidden, args.hidden, 3], activation="relu")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)

    print(f"Модель: {model}  ({count_parameters(model):,} параметров)")
    print(f"Данные: {len(X_train)} train / {len(X_val)} val, 3 класса")

    start = time.perf_counter()
    batch_size = args.batch_size
    for step in range(1, args.steps + 1):
        model.train()
        batch_idx = rng.integers(0, len(X_train), size=batch_size)
        xb, yb = X_train[batch_idx], y_train[batch_idx]
        logits = model(Tensor(xb))
        loss = nn.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            val_acc = accuracy(model, X_val, y_val)
            print(f"шаг {step:4d} | loss {loss.item():.4f} | val acc {val_acc:.3f}")

    model.eval()
    train_acc = accuracy(model, X_train, y_train)
    val_acc = accuracy(model, X_val, y_val)
    elapsed = time.perf_counter() - start
    print(f"\nГотово за {format_time(elapsed)}: train acc {train_acc:.3f}, val acc {val_acc:.3f}")

    if not args.no_map:
        print("\nКарта решений ('.', '+', '#' — классы):")
        print(ascii_decision_map(model))

    path = save_checkpoint(model, Path("checkpoints") / "mlp_spiral.npz")
    print(f"\nЧекпойнт: {path}")


if __name__ == "__main__":
    main()
