"""Данные: char-level токенизатор, текстовые батчи и синтетические задачи."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "CharTokenizer",
    "TextDataset",
    "make_spiral",
    "make_moons",
    "make_xor",
    "train_val_split",
    "iterate_minibatches",
]


class CharTokenizer:
    """Посимвольная токенизация: словарь строится по тексту."""

    def __init__(self, text: str | None = None, chars: Sequence[str] | None = None):
        if chars is None:
            if text is None:
                raise ValueError("нужен text или chars")
            chars = sorted(set(text))
        self.chars = list(chars)
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("символы в словаре не должны повторяться")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            raise ValueError(f"символы отсутствуют в словаре: {unknown!r}")
        return np.array([self.stoi[ch] for ch in text], dtype=np.int64)

    def encode_safe(self, text: str) -> np.ndarray:
        """Как encode, но молча выбрасывает незнакомые символы (для промптов)."""
        return np.array([self.stoi[ch] for ch in text if ch in self.stoi], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in np.asarray(ids).reshape(-1))

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(chars=data["chars"])

    def __len__(self) -> int:
        return len(self.chars)

    def __repr__(self) -> str:
        sample = "".join(self.chars[:40])
        return f"CharTokenizer(vocab={self.vocab_size}, chars={sample!r}...)"


class TextDataset:
    """Текст, разбитый на train/val; выдаёт случайные батчи (x, y)."""

    def __init__(self, text: str, tokenizer: CharTokenizer | None = None,
                 val_fraction: float = 0.1):
        if len(text) < 100:
            raise ValueError("текст слишком короткий")
        self.tokenizer = tokenizer or CharTokenizer(text)
        self.ids = self.tokenizer.encode(text)
        split = int(len(self.ids) * (1.0 - val_fraction))
        self.train_ids = self.ids[:split]
        self.val_ids = self.ids[split:]

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "TextDataset":
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, **kwargs)

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  rng: np.random.Generator | None = None):
        data = self.train_ids if split == "train" else self.val_ids
        if len(data) <= block_size + 1:
            raise ValueError(
                f"{split}: нужно больше {block_size + 1} токенов, есть {len(data)}"
            )
        rng = rng or np.random.default_rng()
        starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[i : i + block_size] for i in starts])
        y = np.stack([data[i + 1 : i + block_size + 1] for i in starts])
        return x, y

    def __repr__(self) -> str:
        return (f"TextDataset(train={len(self.train_ids)}, val={len(self.val_ids)}, "
                f"vocab={self.tokenizer.vocab_size})")


# ---------------------------------------------------------------------- #
#  Синтетические задачи
# ---------------------------------------------------------------------- #
def make_spiral(n_per_class: int = 200, n_classes: int = 3, noise: float = 0.05,
                seed: int = 0):
    """Классическая задача «спираль» — нелинейная, для проверки MLP."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        t = np.linspace(0.15, 1.0, n_per_class)
        angle = t * 3.2 * np.pi + c * (2 * np.pi / n_classes)
        radius = t
        x = np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1)
        x += rng.normal(scale=noise, size=x.shape)
        xs.append(x)
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_xor(n: int = 400, noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    X = X + rng.normal(scale=noise, size=X.shape)
    return X.astype(np.float32), y


def make_moons(n: int = 400, noise: float = 0.08, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_out = n // 2
    theta = np.linspace(0, np.pi, n_out)
    outer = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inner = np.stack([1 - np.cos(theta), 0.5 - np.sin(theta)], axis=1)
    X = np.concatenate([outer, inner]) + rng.normal(scale=noise, size=(n_out * 2, 2))
    y = np.concatenate([np.zeros(n_out, dtype=np.int64), np.ones(n_out, dtype=np.int64)])
    order = rng.permutation(len(X))
    return X[order].astype(np.float32), y[order]


def train_val_split(X, y, val_fraction: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    split = int(len(X) * (1.0 - val_fraction))
    return X[:split], y[:split], X[split:], y[split:]


def iterate_minibatches(X, y, batch_size: int, shuffle: bool = True,
                        rng: np.random.Generator | None = None) -> Iterator[tuple]:
    rng = rng or np.random.default_rng()
    order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        idx = order[start : start + batch_size]
        yield X[idx], y[idx]
