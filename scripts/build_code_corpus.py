"""Собирает корпус Python-кода: исходники проекта + стандартная библиотека.

Запуск: python scripts/build_code_corpus.py
Результат: data/code.txt (~1.5 МБ).
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "code.txt"
MAX_BYTES = 1_500_000

PROJECT_DIRS = ["subash", "examples", "tests", "scripts", "app"]
FORBIDDEN = ("пушкин", "князь у синя", "три девицы", "сказк")
SKIP_PARTS = {
    "test", "tests", "testing", "__pycache__", "site-packages", "idlelib",
    "tkinter", "turtledemo", "lib2to3", "ensurepip", "venv", "distutils",
    "pydoc_data", "lib-dynload", "doc", "docs",
}


def project_files() -> list[Path]:
    files = []
    for name in PROJECT_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            if path.is_file():
                files.append(path)
    return files


def stdlib_files() -> list[Path]:
    lib = Path(sysconfig.get_paths()["stdlib"])
    files = []
    for path in sorted(lib.rglob("*.py")):
        rel = path.relative_to(lib)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if not (500 <= path.stat().st_size <= 150_000):
            continue
        files.append(path)
    return files


def read_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    # В корпусе не должно быть сказок и пушкинских промптов — только код.
    lines = [
        line for line in text.splitlines()
        if not any(marker in line.lower() for marker in FORBIDDEN)
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    files = project_files()
    for path in stdlib_files():
        files.append(path)

    parts: list[str] = []
    size = 0
    used = 0
    for path in files:
        code = read_code(path)
        if size + len(code) > MAX_BYTES:
            continue
        parts.append(code)
        size += len(code) + 2
        used += 1

    corpus = "\n\n".join(parts) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(corpus, encoding="utf-8")
    print(f"Файлов: {used}, символов: {len(corpus):,}, словарь: {len(set(corpus))}")
    print(f"Сохранено: {OUT}")


if __name__ == "__main__":
    main()
