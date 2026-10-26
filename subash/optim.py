"""Оптимизаторы: SGD с моментом и AdamW. Всё вручную, на numpy."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .tensor import Tensor

__all__ = ["Optimizer", "SGD", "AdamW", "clip_grad_norm_", "cosine_lr"]


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("оптимизатору не передано ни одного параметра")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr}, params={len(self.params)})"


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.buffers = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            if self.momentum:
                buf *= self.momentum
                buf += grad
                grad = grad + self.momentum * buf if self.nesterov else buf
            p.data -= self.lr * grad


class AdamW(Optimizer):
    """Adam с развязанным weight decay (как в nanoGPT)."""

    def __init__(self, params, lr: float = 3e-3, betas: tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (grad * grad)
            m_hat = m / bias1
            v_hat = v / bias2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> float:
    """Обрезает общую L2-норму градиентов. Возвращает норму до обрезки."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total = math.sqrt(sum(float(np.sum(p.grad * p.grad)) for p in params))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for p in params:
            p.grad *= scale
    return total


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int = 0,
              min_lr: float = 0.0) -> float:
    """Линейный разогрев + косинусное затухание."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
