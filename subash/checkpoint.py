"""Сохранение и загрузка моделей в один .npz-файл."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import CharTokenizer
from .models import MLP, TinyGPT

__all__ = ["save_checkpoint", "load_checkpoint", "build_model"]


def build_model(config: dict):
    kind = config.get("type")
    args = config.get("args", {})
    if kind == "mlp":
        return MLP(**args)
    if kind == "gpt":
        return TinyGPT(**args)
    raise ValueError(f"неизвестный тип модели: {kind!r}")


def save_checkpoint(model, path: str | Path, tokenizer: CharTokenizer | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: p.data for name, p in model.named_parameters()}
    arrays["__config__"] = np.array(json.dumps(model.config, ensure_ascii=False))
    if tokenizer is not None:
        arrays["__tokenizer__"] = np.array(json.dumps(tokenizer.to_dict(), ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    return path


def load_checkpoint(path: str | Path):
    with np.load(Path(path), allow_pickle=False) as data:
        config = json.loads(str(data["__config__"]))
        model = build_model(config)
        for name, param in model.named_parameters():
            if name not in data:
                raise KeyError(f"в чекпойнте нет параметра {name!r}")
            param.data[...] = data[name]
        tokenizer = None
        if "__tokenizer__" in data:
            tokenizer = CharTokenizer.from_dict(json.loads(str(data["__tokenizer__"])))
        model.eval()
    return model, tokenizer
