"""Мелкие утилиты: сид, подсчёт параметров, форматирование времени."""

from __future__ import annotations

import random
import time

import numpy as np

__all__ = ["set_seed", "count_parameters", "format_time", "Timer"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model) -> int:
    return sum(p.size for p in model.parameters())


def format_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} с"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {seconds:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self) -> str:
        return format_time(self.elapsed)
