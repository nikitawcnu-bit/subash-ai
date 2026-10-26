"""Собирает корпус из сказок А. С. Пушкина (общественное достояние).

Скрипт нужен только для воспроизводимости: готовый корпус уже лежит в data/pushkin.txt.
Источник: ru.wikisource.org, шаблон {{PD-old}}.

Запуск:  python scripts/build_corpus.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

WORKS = [
    # (заголовок страницы, первая строка, последняя строка)
    ("Сказка о царе Салтане (Пушкин)", "Три девицы под окном", "И усы лишь обмочил."),
    ("Сказка о рыбаке и рыбке (Пушкин)", "Жил старик со своею старухой", "А пред нею разбитое корыто."),
    (
        "Сказка о мёртвой царевне и о семи богатырях (Пушкин)",
        "Царь с царицею простился",
        "Да усы лишь обмочил.",
    ),
    ("Сказка о золотом петушке (Пушкин)", "Негде, в тридевятом царстве", "Добрым молодцам урок."),
    ("Сказка о попе и о работнике его Балде (Пушкин)", "Жил-был поп", "за дешевизной"),
]

JUNK = re.compile(
    r"^(?:←|→|Материал из|Эта страница|Категория|Примечания|Источники|Литература|Ссылки)"
    r"|Викитек|Википеди|Викисклад|Викицитат|Лаборатория Фантастики|Нормативный контроль"
)

USER_AGENT = "SubashAI/0.1 (educational corpus builder; contact: local)"
CACHE = Path(__file__).resolve().parent / ".cache"


def normalize(line: str) -> str:
    """Нормализация для сравнения строк: без пунктуации, регистра и ё."""
    line = line.lower().replace("ё", "е").replace("«", "").replace("»", "")
    return re.sub(r"[^0-9a-zа-я]+", " ", line).strip()


def fetch_html(title: str) -> str:
    """Скачивает HTML страницы, кеширует на диск и переживает 429/5xx."""
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / (hashlib.sha1(title.encode("utf-8")).hexdigest()[:16] + ".html")
    if cached.exists() and cached.stat().st_size > 10_000:
        return cached.read_text(encoding="utf-8")

    url = "https://ru.wikisource.org/api/rest_v1/page/html/" + quote(title, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 5
    for attempt in range(5):
        try:
            html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8")
            cached.write_text(html, encoding="utf-8")
            return html
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt == 4:
                raise
            print(f"  HTTP {exc.code}, повтор через {delay} с...")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("не удалось скачать страницу")


def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", class_="mw-parser-output") or soup.body
    # Убираем служебные элементы. Картинки и их подписи не трогаем:
    # в подписях иногда оказываются строки стихотворения.
    for tag in root.select("style, script, sup, table, .mw-editsection"):
        tag.decompose()
    for br in root.find_all("br"):
        br.replace_with("\n")
    for tag in root.find_all(["p", "div", "li", "h2", "h3", "h4"]):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = root.get_text("", strip=False)

    lines = []
    for raw in text.split("\n"):
        line = raw.replace("\u00a0", " ")
        # Обрывки HTML, которые Викитека отдаёт как обычный текст.
        line = re.sub(r"<[^>\n]*>?", " ", line)
        line = re.sub(r'(?:class|style|id)="[^"\n]*"?', " ", line)
        line = line.replace(">", " ").replace("|", " ")
        line = re.sub(r"\s+", " ", line).strip(" \"'")
        if not line or not re.search(r"[А-Яа-яЁё]", line):
            continue
        lines.append(line)
    return lines


def extract(work) -> str:
    title, first, last = work
    lines = html_to_lines(fetch_html(title))

    # Первая строка может цитироваться в шапке страницы, поэтому берём
    # последнее совпадение; конец ищем первым после начала.
    starts = [i for i, line in enumerate(lines) if normalize(first) in normalize(line)]
    if not starts:
        raise RuntimeError(f"Не найдено начало {title!r}")
    start = starts[-1]
    end = next(
        (i for i in range(start, len(lines)) if normalize(last) in normalize(lines[i])),
        None,
    )
    if end is None or end <= start:
        raise RuntimeError(f"Не найден конец {title!r}: start={start}, end={end}")

    body = []
    for line in lines[start : end + 1]:
        if JUNK.search(line) or re.fullmatch(r"[\[\]\d\s.,—–-]+", line):
            continue
        if len(line) < 2:
            continue
        body.append(line)

    text = "\n".join(body)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for work in WORKS:
        text = extract(work)
        print(f"{work[0]}: {len(text)} символов, {text.count(chr(10)) + 1} строк")
        parts.append(text)
    corpus = "\n\n".join(parts) + "\n"
    path = out_dir / "pushkin.txt"
    path.write_text(corpus, encoding="utf-8")
    print(f"Итого: {len(corpus)} символов -> {path}")


if __name__ == "__main__":
    main()
