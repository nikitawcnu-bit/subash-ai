"""Subash AI — настольное приложение с мини-GPT на NumPy.

Запуск из исходников:
    python app/subash_app.py

Сборка .exe:
    powershell -ExecutionPolicy Bypass -File app/build_exe.ps1
"""

from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BG = "#0b1220"
PANEL = "#111c30"
ENTRY_BG = "#0f1a2e"
BORDER = "#24344d"
FG = "#e6edf7"
MUTED = "#8aa0bd"
ACCENT = "#4cc2ff"
ACCENT_HOVER = "#7ad4ff"
ACCENT_DARK = "#1f6feb"

DEFAULT_PROMPT = "Князь у синя моря ходит"
EXAMPLES = [
    "Три девицы под окном",
    "Свет мой, зеркальце! скажи",
    "Ветер по морю гуляет",
]


def resource_path(relative: str) -> Path:
    """Путь к ресурсу и в исходниках, и внутри PyInstaller-сборки."""
    base = Path(getattr(sys, "_MEIPASS", ROOT))
    return base / relative


def load_model():
    from subash import load_checkpoint

    path = resource_path("checkpoints/gpt_pushkin.npz")
    model, tokenizer = load_checkpoint(path)
    model.eval()
    return model, tokenizer


def run_selftest() -> int:
    """Проверка собранного exe без GUI: пишет результат в TEMP."""
    out_path = Path(tempfile.gettempdir()) / "subash_selftest.txt"
    try:
        model, tokenizer = load_model()
        ids = tokenizer.encode_safe(DEFAULT_PROMPT)
        result = model.generate(ids, max_new_tokens=24, temperature=0.6, top_k=20)
        out_path.write_text(tokenizer.decode(result[0]), encoding="utf-8")
        return 0
    except Exception:
        out_path.write_text("ERROR\n" + traceback.format_exc(), encoding="utf-8")
        return 1


class App:
    def __init__(self, root, model, tokenizer):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.model = model
        self.tokenizer = tokenizer

        root.title("Subash AI — нейросеть на NumPy")
        root.geometry("940x700")
        root.minsize(760, 560)
        root.configure(bg=BG)
        icon = resource_path("assets/subash.ico")
        if icon.exists():
            try:
                root.iconbitmap(str(icon))
            except tk.TclError:
                pass

        self.prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        self.temp_var = tk.DoubleVar(value=0.75)
        self.tokens_var = tk.IntVar(value=180)
        self.status_var = tk.StringVar(value="Загрузка модели…")
        self._busy = False
        self._results: queue.Queue = queue.Queue()

        self._build()
        self.prompt_entry.focus_set()
        self.status_var.set(
            f"Модель загружена · {model.num_parameters():,} параметров · "
            f"словарь {tokenizer.vocab_size} символов".replace(",", " ")
        )
        self.root.after(60, self._poll_results)

    # ------------------------------------------------------------------ #
    #  Интерфейс
    # ------------------------------------------------------------------ #
    def _label(self, parent, text, size=11, color=FG, bold=False, bg=None):
        tk = self.tk
        font = ("Segoe UI", size, "bold") if bold else ("Segoe UI", size)
        return tk.Label(parent, text=text, bg=bg or BG, fg=color, font=font)

    def _button(self, parent, text, command, kind="primary", width=None):
        tk = self.tk
        if kind == "primary":
            bg, fg, active = ACCENT, "#06121f", ACCENT_HOVER
            font = ("Segoe UI", 12, "bold")
            padx, pady = 24, 10
        else:
            bg, fg, active = PANEL, FG, "#1b2b45"
            font = ("Segoe UI", 10)
            padx, pady = 14, 7
        button = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=active, activeforeground=fg,
            disabledforeground="#6f839f", relief="flat", bd=0,
            font=font, padx=padx, pady=pady, cursor="hand2",
            highlightthickness=0,
        )
        if width:
            button.configure(width=width)
        if kind == "primary":
            button.bind("<Enter>", lambda _e: button.configure(bg=ACCENT_HOVER) if str(button["state"]) != "disabled" else None)
            button.bind("<Leave>", lambda _e: button.configure(bg=ACCENT) if str(button["state"]) != "disabled" else None)
        return button

    def _build(self):
        tk = self.tk
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        wrapper = tk.Frame(root, bg=BG, padx=28, pady=22)
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(5, weight=1)

        # Шапка
        header = tk.Frame(wrapper, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        tk.Label(header, text="SUBASH AI", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 26, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(header, text="мини-GPT на чистом NumPy · обучен на сказках Пушкина",
                 bg=BG, fg=MUTED, font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._label(header, "237 408 параметров · 96 эмбеддингов", size=10, color=MUTED).grid(
            row=0, column=1, rowspan=2, sticky="e")

        # Промпт
        prompt_box = tk.Frame(wrapper, bg=BG)
        prompt_box.grid(row=1, column=0, sticky="ew", pady=(22, 0))
        prompt_box.grid_columnconfigure(0, weight=1)
        self._label(prompt_box, "Начало фразы", size=10, color=MUTED).grid(row=0, column=0, sticky="w")
        self.prompt_entry = tk.Entry(
            prompt_box, textvariable=self.prompt_var, bg=ENTRY_BG, fg=FG,
            insertbackground=FG, relief="flat", bd=0, font=("Segoe UI", 14),
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self.prompt_entry.grid(row=1, column=0, sticky="ew", ipady=9, pady=(5, 0))
        self.prompt_entry.bind("<Return>", lambda _e: self.on_generate())

        examples = tk.Frame(prompt_box, bg=BG)
        examples.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self._label(examples, "Примеры:", size=9, color=MUTED).pack(side="left", padx=(2, 6))
        for text in EXAMPLES:
            link = tk.Label(examples, text=text, bg=BG, fg="#7dd3fc",
                            font=("Segoe UI", 9, "underline"), cursor="hand2")
            link.pack(side="left", padx=(0, 12))
            link.bind("<Button-1>", lambda _e, value=text: self.prompt_var.set(value))

        # Параметры
        options = tk.Frame(wrapper, bg=BG)
        options.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        options.grid_columnconfigure(2, weight=1)

        self._label(options, "Температура", size=10, color=MUTED).grid(row=0, column=0, sticky="w")
        self.temp_scale = tk.Scale(
            options, from_=0.1, to=1.3, resolution=0.05, orient="horizontal",
            variable=self.temp_var, showvalue=True, bg=BG, fg=FG,
            troughcolor="#1c2b44", highlightthickness=0, bd=0,
            sliderlength=18, length=210, font=("Segoe UI", 9),
            activebackground=ACCENT,
        )
        self.temp_scale.grid(row=1, column=0, sticky="w", padx=(0, 28))

        self._label(options, "Символов", size=10, color=MUTED).grid(row=0, column=1, sticky="w")
        self.tokens_spin = tk.Spinbox(
            options, from_=40, to=400, increment=20, textvariable=self.tokens_var,
            width=6, bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat",
            buttonbackground=PANEL, font=("Segoe UI", 11), justify="center",
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self.tokens_spin.grid(row=1, column=1, sticky="w")

        self.generate_button = self._button(options, "Продолжить текст", self.on_generate)
        self.generate_button.grid(row=1, column=2, sticky="e", padx=(28, 0))
        options.grid_columnconfigure(2, weight=1)

        # Вывод
        output_box = tk.Frame(wrapper, bg=BG)
        output_box.grid(row=5, column=0, sticky="nsew", pady=(20, 0))
        output_box.grid_columnconfigure(0, weight=1)
        output_box.grid_rowconfigure(1, weight=1)

        top = tk.Frame(output_box, bg=BG)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        top.grid_columnconfigure(0, weight=1)
        self._label(top, "Результат", size=10, color=MUTED).grid(row=0, column=0, sticky="w")
        self.copy_button = self._button(top, "Скопировать", self.on_copy, kind="secondary")
        self.copy_button.grid(row=0, column=1, padx=(0, 8))
        self.clear_button = self._button(top, "Очистить", self.on_clear, kind="secondary")
        self.clear_button.grid(row=0, column=2)

        text_frame = tk.Frame(output_box, bg=BORDER, padx=1, pady=1)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        self.output = tk.Text(
            text_frame, wrap="word", bg=PANEL, fg=FG, relief="flat", bd=0,
            font=("Segoe UI", 12), padx=16, pady=14, spacing1=2, spacing3=4,
            insertbackground=FG, selectbackground="#28456b", state="disabled",
        )
        self.output.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(text_frame, command=self.output.yview, bg=PANEL,
                                 troughcolor=PANEL, activebackground=BORDER, width=12)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.tag_configure("prompt", foreground="#7dd3fc")
        self.output.tag_configure("body", foreground=FG)

        # Статус
        footer = tk.Frame(wrapper, bg=BG)
        footer.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        footer.grid_columnconfigure(0, weight=1)
        tk.Label(footer, textvariable=self.status_var, bg=BG, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w").grid(row=0, column=0, sticky="w")
        credit = tk.Label(footer, text="github.com/nikitawcnu-bit/subash-ai", bg=BG,
                          fg="#5f7896", font=("Segoe UI", 9, "underline"), cursor="hand2")
        credit.grid(row=0, column=1, sticky="e")
        credit.bind("<Button-1>", lambda _e: self._open_github())

    # ------------------------------------------------------------------ #
    #  Действия
    # ------------------------------------------------------------------ #
    def _open_github(self):
        import webbrowser
        webbrowser.open("https://github.com/nikitawcnu-bit/subash-ai")

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.generate_button.configure(
            state="disabled" if busy else "normal",
            text="Генерирую…" if busy else "Продолжить текст",
            bg="#2a3d58" if busy else ACCENT,
        )

    def on_generate(self):
        if self._busy:
            return
        tk = self.tk
        prompt = self.prompt_var.get().strip()
        if not prompt:
            self.status_var.set("Введите начало фразы")
            return
        ids = self.tokenizer.encode_safe(prompt)
        if len(ids) == 0:
            self.status_var.set("В промпте нет символов, знакомых модели")
            return
        try:
            temperature = float(self.temp_var.get())
            tokens = int(self.tokens_var.get())
        except (tk.TclError, ValueError):
            temperature, tokens = 0.75, 180
        tokens = max(20, min(tokens, 400))

        self._show_text(prompt, "", "Генерирую продолжение…")
        self._set_busy(True)
        thread = threading.Thread(
            target=self._worker, args=(prompt, ids, temperature, tokens), daemon=True
        )
        thread.start()

    def _worker(self, prompt, ids, temperature, tokens):
        try:
            started = time.perf_counter()
            result = self.model.generate(
                ids, max_new_tokens=tokens, temperature=temperature, top_k=40
            )
            text = self.tokenizer.decode(result[0])
            elapsed = time.perf_counter() - started
            self._results.put(("done", prompt, text, elapsed))
        except Exception:
            self._results.put(("error", traceback.format_exc()))

    def _poll_results(self):
        try:
            while True:
                item = self._results.get_nowait()
                if item[0] == "done":
                    self._done(*item[1:])
                else:
                    self._failed(item[1])
        except queue.Empty:
            pass
        self.root.after(60, self._poll_results)

    def _done(self, prompt, text, elapsed):
        self._set_busy(False)
        self._show_text(prompt, text, f"Готово за {elapsed:.1f} с · {len(text)} символов")

    def _failed(self, error):
        from tkinter import messagebox

        self._set_busy(False)
        self.status_var.set("Ошибка генерации — подробности в окне сообщения")
        messagebox.showerror("Subash AI", error[-1200:])

    def _show_text(self, prompt, text, status):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", prompt, "prompt")
        if text.startswith(prompt):
            text = text[len(prompt):]
        self.output.insert("end", text, "body")
        self.output.configure(state="disabled")
        self.output.see("end")
        self.status_var.set(status)

    def on_clear(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.status_var.set("Очищено")

    def on_copy(self):
        text = self.output.get("1.0", "end").strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Скопировано в буфер обмена")


def main() -> None:
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    import tkinter as tk

    model, tokenizer = load_model()
    root = tk.Tk()
    App(root, model, tokenizer)
    root.mainloop()


if __name__ == "__main__":
    main()
