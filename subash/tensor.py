"""Subash AI: тензор с автоматическим дифференцированием.

Реализация в духе micrograd/autograd, но с numpy-массивами, broadcasting'ом
и batched matmul. Этого достаточно, чтобы обучать MLP и мини-GPT.

Пример:
    >>> from subash import Tensor
    >>> x = Tensor([2.0, 3.0], requires_grad=True)
    >>> loss = (x * x).sum()
    >>> loss.backward()
    >>> x.grad
    array([4., 6.])
"""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Tuple

import numpy as np

Array = np.ndarray
_grad_enabled = True


class _NoGrad(ContextDecorator):
    def __enter__(self):
        global _grad_enabled
        self._previous = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._previous
        return False


def no_grad(func=None):
    """Отключает построение графа: `with no_grad():` или декоратор `@no_grad`."""
    manager = _NoGrad()
    return manager if func is None else manager(func)


def grad_enabled() -> bool:
    return _grad_enabled


def _req_grad(*tensors) -> bool:
    return _grad_enabled and any(t.requires_grad for t in tensors)


def _normalize_axis(axis, ndim: int) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


def _unbroadcast(grad: Array, shape: Tuple[int, ...]) -> Array:
    """Свёртка градиента обратно к форме родителя при broadcasting."""
    if grad.shape == shape:
        return grad
    if grad.ndim > len(shape):
        grad = grad.sum(axis=tuple(range(grad.ndim - len(shape))))
    if grad.ndim != len(shape):  # страховка; при валидном broadcasting не срабатывает
        grad = grad.reshape(shape)
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad


def as_tensor(value) -> "Tensor":
    return value if isinstance(value, Tensor) else Tensor(value)


class Tensor:
    """N-мерный массив с градиентом и ссылками на родителей."""

    __slots__ = ("data", "grad", "requires_grad", "_prev", "_backward", "_op")
    __array_priority__ = 1000.0

    def __init__(self, data, requires_grad: bool = False, _prev: tuple = (),
                 _backward=None, op: str = ""):
        self.data = np.asarray(data)
        if self.data.dtype == object:
            raise TypeError("Tensor не поддерживает object-массивы")
        self.grad: Array | None = None
        self.requires_grad = bool(requires_grad)
        self._prev = _prev if requires_grad else ()
        self._backward = _backward
        self._op = op

    # ------------------------------------------------------------------ #
    #  Информация
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> Array:
        return self.data

    def item(self) -> float:
        return float(self.data)

    def tolist(self):
        return self.data.tolist()

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r}\n{np.array2string(self.data, precision=4)})"

    def __float__(self):
        return float(self.data)

    # ------------------------------------------------------------------ #
    #  Служебное
    # ------------------------------------------------------------------ #
    def _add_grad(self, grad) -> None:
        if not self.requires_grad:
            return
        grad = np.asarray(grad)
        if self.grad is None:
            self.grad = np.array(grad, copy=True)
        else:
            self.grad += grad

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad=None) -> None:
        """Обратное распространение от этого тензора (обычно скаляра)."""
        if not self.requires_grad:
            raise RuntimeError("backward() у тензора, который не требует градиента")
        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    f"backward() без градиента возможен только для скаляра, shape={self.shape}"
                )
            grad = np.ones_like(self.data)
        self._add_grad(grad)

        # Итеративный post-order DFS: без рекурсии, чтобы не ловить RecursionError.
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))

        for node in reversed(topo):
            if node._backward is not None:
                node._backward()

    # ------------------------------------------------------------------ #
    #  Арифметика
    # ------------------------------------------------------------------ #
    def __add__(self, other) -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data + other.data, _req_grad(self, other),
                     (self, other), op="add")
        if out.requires_grad:
            def backward():
                g = out.grad
                self._add_grad(_unbroadcast(g, self.shape))
                other._add_grad(_unbroadcast(g, other.shape))
            out._backward = backward
        return out

    def __radd__(self, other) -> "Tensor":
        return as_tensor(other).__add__(self)

    def __sub__(self, other) -> "Tensor":
        return self.__add__(-as_tensor(other))

    def __rsub__(self, other) -> "Tensor":
        return as_tensor(other).__add__(-self)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, _req_grad(self), (self,), op="neg")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(-out.grad)
        return out

    def __mul__(self, other) -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data * other.data, _req_grad(self, other),
                     (self, other), op="mul")
        if out.requires_grad:
            def backward():
                g = out.grad
                self._add_grad(_unbroadcast(g * other.data, self.shape))
                other._add_grad(_unbroadcast(g * self.data, other.shape))
            out._backward = backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return as_tensor(other).__mul__(self)

    def __truediv__(self, other) -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data / other.data, _req_grad(self, other),
                     (self, other), op="div")
        if out.requires_grad:
            def backward():
                g = out.grad
                self._add_grad(_unbroadcast(g / other.data, self.shape))
                other._add_grad(_unbroadcast(-g * self.data / (other.data ** 2), other.shape))
            out._backward = backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        return as_tensor(other).__truediv__(self)

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, _req_grad(self), (self,), op=f"pow{power}")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * power * self.data ** (power - 1))
        return out

    def __matmul__(self, other) -> "Tensor":
        other = as_tensor(other)
        out = Tensor(self.data @ other.data, _req_grad(self, other),
                     (self, other), op="matmul")
        if out.requires_grad:
            def backward():
                g = out.grad
                ga = g @ np.swapaxes(other.data, -1, -2)
                gb = np.swapaxes(self.data, -1, -2) @ g
                self._add_grad(_unbroadcast(ga, self.shape))
                other._add_grad(_unbroadcast(gb, other.shape))
            out._backward = backward
        return out

    # ------------------------------------------------------------------ #
    #  Форма
    # ------------------------------------------------------------------ #
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), _req_grad(self), (self,), op="reshape")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad.reshape(self.shape))
        return out

    def transpose(self, *axes) -> "Tensor":
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(axes), _req_grad(self), (self,), op="transpose")
        if out.requires_grad:
            inverse = tuple(np.argsort(axes))
            out._backward = lambda: self._add_grad(out.grad.transpose(inverse))
        return out

    def swapaxes(self, axis1: int, axis2: int) -> "Tensor":
        axes = list(range(self.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self.transpose(axes)

    def __getitem__(self, index) -> "Tensor":
        out = Tensor(self.data[index], _req_grad(self), (self,), op="getitem")
        if out.requires_grad:
            def backward():
                grad = np.zeros_like(self.data)
                np.add.at(grad, index, out.grad)
                self._add_grad(grad)
            out._backward = backward
        return out

    # ------------------------------------------------------------------ #
    #  Редукции
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        axes = _normalize_axis(axis, self.ndim)
        out = Tensor(self.data.sum(axis=axes, keepdims=keepdims), _req_grad(self),
                     (self,), op="sum")
        if out.requires_grad:
            def backward():
                g = out.grad
                if not keepdims:
                    g = np.expand_dims(g, axes)
                self._add_grad(np.broadcast_to(g, self.shape))
            out._backward = backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        axes = _normalize_axis(axis, self.ndim)
        count = 1
        for a in axes:
            count *= self.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / count)

    # ------------------------------------------------------------------ #
    #  Элементарные функции
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), _req_grad(self), (self,), op="exp")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * out.data)
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), _req_grad(self), (self,), op="log")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad / self.data)
        return out

    def sqrt(self) -> "Tensor":
        out = Tensor(np.sqrt(self.data), _req_grad(self), (self,), op="sqrt")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * 0.5 / out.data)
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), _req_grad(self), (self,), op="tanh")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * (1.0 - out.data ** 2))
        return out

    def sigmoid(self) -> "Tensor":
        out = Tensor(_sigmoid(self.data), _req_grad(self), (self,), op="sigmoid")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * out.data * (1.0 - out.data))
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), _req_grad(self), (self,), op="relu")
        if out.requires_grad:
            out._backward = lambda: self._add_grad(out.grad * (self.data > 0))
        return out

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


# ---------------------------------------------------------------------- #
#  Функции уровня тензоров
# ---------------------------------------------------------------------- #
def _sigmoid(x: Array) -> Array:
    # Численно устойчивый вариант без переполнения exp.
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Численно устойчивый softmax; максимум считается как константа."""
    m = x.data.max(axis=axis, keepdims=True)  # константа, градиент не нужен
    e = (x - m).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    m = x.data.max(axis=axis, keepdims=True)
    shifted = x - m
    return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits: Tensor, targets, reduction: str = "mean") -> Tensor:
    """Softmax + NLL одним узлом: стабильно и без лишних копий.

    logits: (..., V), targets: (...) целые индексы классов.
    """
    targets = np.asarray(targets, dtype=np.int64)
    vocab = logits.shape[-1]
    z = logits.data.reshape(-1, vocab)
    t = targets.reshape(-1)
    n = t.size

    z_max = z.max(axis=-1, keepdims=True)
    exp = np.exp(z - z_max)
    log_probs = (z - z_max) - np.log(exp.sum(axis=-1, keepdims=True))
    picked = log_probs[np.arange(n), t]
    loss_data = float(-picked.mean())

    out = Tensor(np.array(loss_data, dtype=logits.dtype), _req_grad(logits),
                 (logits,), op="cross_entropy")
    if out.requires_grad:
        def backward():
            g = np.exp(log_probs)
            g[np.arange(n), t] -= 1.0
            g *= out.grad / n
            logits._add_grad(g.reshape(logits.shape))
        out._backward = backward
    return out


def accuracy(logits: Tensor, targets) -> float:
    predictions = logits.data.reshape(-1, logits.shape[-1]).argmax(axis=-1)
    return float((predictions == np.asarray(targets).reshape(-1)).mean())
