"""Модели Subash AI: MLP и мини-GPT (decoder-only transformer)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import nn
from .tensor import Tensor, cross_entropy, no_grad, softmax

__all__ = ["MLP", "TinyGPT", "CausalSelfAttention", "Block"]


class MLP(nn.Module):
    """Полносвязная сеть: sizes = [in, hidden..., out]."""

    def __init__(self, sizes: Sequence[int], activation: str = "relu", dropout: float = 0.0,
                 init: str = "he"):
        if len(sizes) < 2:
            raise ValueError("нужно хотя бы два размера слоёв")
        self.sizes = list(sizes)
        self.activation = activation
        self.dropout = dropout
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], init=init))
            if i < len(sizes) - 2:
                layers.append(nn.Activation(activation))
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.config = {
            "type": "mlp",
            "args": {"sizes": self.sizes, "activation": activation, "dropout": dropout},
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        return f"MLP(sizes={self.sizes}, activation={self.activation!r})"


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention с причинной маской."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 std: float = 0.02):
        if n_embd % n_head != 0:
            raise ValueError("n_embd должен делиться на n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = nn.Linear(n_embd, n_embd, std=std)
        self.k = nn.Linear(n_embd, n_embd, std=std)
        self.v = nn.Linear(n_embd, n_embd, std=std)
        self.proj = nn.Linear(n_embd, n_embd, std=std)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = np.triu(np.ones((block_size, block_size), dtype=bool), k=1)
        self.mask = np.where(mask, -1e9, 0.0).astype(np.float32)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        q = self.q(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        att = att + Tensor(self.mask[:T, :T].reshape(1, 1, T, T))
        att = softmax(att, axis=-1)
        att = self.attn_dropout(att)

        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_dropout(self.proj(y))

    def __repr__(self) -> str:
        return f"CausalSelfAttention(heads={self.n_head}, dim={self.head_dim * self.n_head})"


class Block(nn.Module):
    """Pre-LN блок трансформера: attention + MLP с residual-связями."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 mlp_mult: int = 4):
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, mlp_mult * n_embd, std=0.02)
        self.fc2 = nn.Linear(mlp_mult * n_embd, n_embd, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        hidden = nn.gelu(self.fc1(self.ln2(x)))
        x = x + self.dropout(self.fc2(hidden))
        return x


class TiedHead(nn.Module):
    """Выходной слой, переиспользующий матрицу эмбеддингов (weight tying)."""

    def __init__(self, weight: Tensor):
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(1, 0)


class TinyGPT(nn.Module):
    """Маленький decoder-only трансформер для генерации текста по символам."""

    def __init__(self, vocab_size: int, block_size: int = 96, n_layer: int = 3,
                 n_head: int = 4, n_embd: int = 128, dropout: float = 0.0,
                 tie_weights: bool = True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd, std=0.02)
        self.pos_emb = nn.Embedding(block_size, n_embd, std=0.01)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        if tie_weights:
            self.head = TiedHead(self.tok_emb.weight)
        else:
            self.head = nn.Linear(n_embd, vocab_size, bias=False, std=0.02)
        self.config = {
            "type": "gpt",
            "args": {
                "vocab_size": vocab_size,
                "block_size": block_size,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "dropout": dropout,
                "tie_weights": tie_weights,
            },
        }

    def forward(self, idx, targets=None):
        idx = np.asarray(idx, dtype=np.int64)
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"длина {T} больше block_size={self.block_size}")
        positions = np.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    @no_grad
    def generate(self, idx, max_new_tokens: int = 200, temperature: float = 0.9,
                 top_k: int | None = None, top_p: float | None = None) -> np.ndarray:
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]
        if idx.size == 0:
            raise ValueError("пустой промпт")
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(Tensor(context))
            logits = logits.data[0, -1].astype(np.float64)

            if temperature <= 0:
                next_id = int(np.argmax(logits))
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, logits.size)
                    threshold = np.sort(logits)[-top_k]
                    logits = np.where(logits < threshold, -np.inf, logits)
                if top_p is not None and 0.0 < top_p < 1.0:
                    logits = _top_p_filter(logits, top_p)
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

            idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
        return idx

    def __repr__(self) -> str:
        return (f"TinyGPT(vocab={self.vocab_size}, block={self.block_size}, "
                f"layers={self.n_layer}, heads={self.n_head}, embd={self.n_embd})")


def _top_p_filter(logits: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(logits)[::-1]
    sorted_logits = logits[order]
    probs = np.exp(sorted_logits - sorted_logits.max())
    cumulative = np.cumsum(probs) / probs.sum()
    cutoff = int(np.searchsorted(cumulative, top_p) + 1)
    keep = order[:cutoff]
    filtered = np.full_like(logits, -np.inf)
    filtered[keep] = logits[keep]
    return filtered
