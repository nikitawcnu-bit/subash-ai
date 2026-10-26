"""Слои, функции активации и потери Subash AI."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .tensor import Tensor, cross_entropy, softmax

__all__ = [
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "Dropout",
    "Sequential",
    "Activation",
    "ACTIVATIONS",
    "relu",
    "gelu",
    "tanh",
    "sigmoid",
    "mse_loss",
    "cross_entropy",
    "softmax",
]


class Module:
    """Базовый класс слоя/модели."""

    training: bool = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # -- параметры ------------------------------------------------------- #
    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        seen: set[int] = set()

        def walk(module: Module, prefix: str):
            for name, value in vars(module).items():
                if name.startswith("_"):
                    continue
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(value, Tensor) and value.requires_grad:
                    if id(value) not in seen:
                        seen.add(id(value))
                        yield full, value
                elif isinstance(value, Module):
                    yield from walk(value, full)
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if isinstance(item, Module):
                            yield from walk(item, f"{full}.{i}")

        yield from walk(self, "")

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for value in vars(self).values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Linear(Module):
    """y = x @ W + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 std: float | None = None, init: str = "he"):
        if std is None:
            if init == "he":
                std = float(np.sqrt(2.0 / in_features))
            elif init == "xavier":
                std = float(np.sqrt(2.0 / (in_features + out_features)))
            elif init == "normal":
                std = 0.02
            else:
                raise ValueError(f"неизвестная инициализация: {init}")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * np.float32(std),
            requires_grad=True,
        )
        self.bias = (
            Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear({self.in_features} -> {self.out_features})"


class Embedding(Module):
    """Таблица эмбеддингов: weight[ids]."""

    def __init__(self, num_embeddings: int, embedding_dim: int, std: float = 0.02):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            (np.random.randn(num_embeddings, embedding_dim) * std).astype(np.float32),
            requires_grad=True,
        )

    def forward(self, ids) -> Tensor:
        return self.weight[np.asarray(ids, dtype=np.int64)]

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings} x {self.embedding_dim})"


class LayerNorm(Module):
    """Нормализация по последней оси + обучаемые масштаб и сдвиг."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gain = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / (var + self.eps).sqrt()
        return normalized * self.gain + self.bias

    def __repr__(self) -> str:
        return f"LayerNorm({self.gain.size})"


class Dropout(Module):
    def __init__(self, p: float = 0.1):
        if not 0.0 <= p < 1.0:
            raise ValueError("p должен быть в [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (np.random.random(x.shape) >= self.p).astype(x.dtype)
        mask = Tensor(keep / (1.0 - self.p))
        return x * mask

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        body = "\n".join(f"  ({i}) {layer!r}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{body}\n)"


# ---------------------------------------------------------------------- #
#  Активации
# ---------------------------------------------------------------------- #
def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    # tanh-приближение из статьи "Gaussian Error Linear Units".
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + ((x + 0.044715 * x * x * x) * c).tanh())


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


ACTIVATIONS = {
    "relu": relu,
    "gelu": gelu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "linear": lambda x: x,
}


class Activation(Module):
    def __init__(self, name: str = "relu"):
        if name not in ACTIVATIONS:
            raise ValueError(f"неизвестная активация {name!r}, доступны: {sorted(ACTIVATIONS)}")
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        return ACTIVATIONS[self.name](x)

    def __repr__(self) -> str:
        return f"Activation({self.name})"


# ---------------------------------------------------------------------- #
#  Потери
# ---------------------------------------------------------------------- #
def mse_loss(prediction: Tensor, target) -> Tensor:
    diff = prediction - target
    return (diff * diff).mean()
