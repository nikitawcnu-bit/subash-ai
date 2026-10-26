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
SIDEBAR = "#0d1728"
CARD = "#111c30"
ENTRY = "#0b1526"
BORDER = "#22324a"
FG = "#e6edf7"
MUTED = "#8aa0bd"
ACCENT = "#4cc2ff"
ACCENT_HOVER = "#7ad4ff"
DIM = "#5f7896"

DEFAULT_PROMPT = "Князь у синя моря ходит"
EXAMPLES = [
    ("Князь у синя моря ходит", "Из «Сказки о царе Салтане»"),
    ("Три девицы под окном", "Начало сказки"),
    ("Свет мой, зеркальце! скажи", "Диалог с зеркальцем"),
    ("Ветер по морю гуляет", "Кораблик и волны"),
]


def resource_path(relative: str) -> Path:
    """Путь к ресурсу и в исходниках, и внутри PyInstaller-сборки."""
    base = Path(getattr(sys, "_MEIPASS", ROOT))
    return base / relative


def symbols_text(count: int) -> str:
    count = abs(int(count))
    if count % 10 == 1 and count % 100 != 11:
        return f"{count} символ"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return f"{count} символа"
    return f"{count} символов"


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
        import customtkinter as ctk
        import tkinter as tk
        from PIL import Image

        self.ctk = ctk
        self.tk = tk
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self._busy = False
        self._results: queue.Queue = queue.Queue()
        self._last_text = ""

        root.title("Subash AI — генератор текста")
        root.minsize(1000, 680)
        root.configure(fg_color=BG)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        icon = resource_path("assets/subash.ico")
        if icon.exists():
            try:
                root.iconbitmap(str(icon))
            except tk.TclError:
                pass

        self._logo = ctk.CTkImage(
            light_image=Image.open(icon), dark_image=Image.open(icon), size=(52, 52)
        ) if icon.exists() else None

        self._build_sidebar()
        self._build_main()
        self._center(1120, 760)
        self._dark_title_bar()
        self.prompt_entry.focus_set()
        self.root.bind("<Control-Return>", lambda _e: self.on_generate())
        self.root.after(60, self._poll_results)
        self.root.after(200, self._focus_prompt)
        self._set_status(
            f"Модель загружена · {model.num_parameters():,} параметров".replace(",", " ")
        )

    # ------------------------------------------------------------------ #
    #  Каркас
    # ------------------------------------------------------------------ #
    def _center(self, width: int, height: int):
        self.root.update_idletasks()
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2 - 20)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _dark_title_bar(self):
        """Тёмный заголовок окна в Windows 10/11 (если поддерживается)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            enabled = ctypes.c_int(1)
            for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(enabled), ctypes.sizeof(enabled)
                ) == 0:
                    break
            caption = ctypes.c_int(0x0020120B)  # COLORREF для #0b1220
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption)
            )
        except Exception:
            pass

    def _section(self, parent, text: str):
        ctk = self.ctk
        return ctk.CTkLabel(parent, text=text.upper(), text_color=DIM,
                            font=("Segoe UI", 10, "bold"), anchor="w")

    def _divider(self, parent, pady=(16, 16)):
        ctk = self.ctk
        line = ctk.CTkFrame(parent, height=1, fg_color=BORDER, corner_radius=0)
        line.pack(fill="x", padx=20, pady=pady)

    def _ghost_button(self, parent, text, command, height=32):
        ctk = self.ctk
        return ctk.CTkButton(
            parent, text=text, command=command, height=height,
            fg_color="transparent", hover_color="#16243a", text_color=FG,
            font=("Segoe UI", 12), anchor="w", corner_radius=8, border_width=0,
        )

    def _build_sidebar(self):
        ctk = self.ctk
        sidebar = ctk.CTkFrame(self.root, width=272, corner_radius=0, fg_color=SIDEBAR)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=20, pady=(24, 0))
        if self._logo is not None:
            ctk.CTkLabel(brand, text="", image=self._logo).pack(side="left")
        titles = ctk.CTkFrame(brand, fg_color="transparent")
        titles.pack(side="left", padx=(12, 0), fill="x", expand=True)
        ctk.CTkLabel(titles, text="SUBASH AI", text_color=ACCENT,
                     font=("Segoe UI", 20, "bold"), anchor="w").pack(fill="x")
        ctk.CTkLabel(titles, text="мини-GPT на NumPy", text_color=MUTED,
                     font=("Segoe UI", 11), anchor="w").pack(fill="x")

        self._divider(sidebar)

        self._section(sidebar, "Модель").pack(fill="x", padx=22, pady=(0, 8))
        card = ctk.CTkFrame(sidebar, fg_color=CARD, corner_radius=12)
        card.pack(fill="x", padx=16)
        rows = [
            ("Параметров", f"{self.model.num_parameters():,}".replace(",", " ")),
            ("Слоёв", "2"),
            ("Голов", "4"),
            ("Эмбеддингов", "96"),
            ("Словарь", f"{self.tokenizer.vocab_size} символов"),
        ]
        for i, (label, value) in enumerate(rows):
            ctk.CTkLabel(card, text=label, text_color=MUTED,
                         font=("Segoe UI", 11), anchor="w").grid(
                row=i, column=0, sticky="w", padx=(14, 8), pady=(7 if i == 0 else 2, 2))
            ctk.CTkLabel(card, text=value, text_color=FG,
                         font=("Segoe UI", 11, "bold"), anchor="e").grid(
                row=i, column=1, sticky="e", padx=(8, 14), pady=(7 if i == 0 else 2, 2))
        card.grid_columnconfigure(0, weight=1)

        self._section(sidebar, "Быстрые промпты").pack(fill="x", padx=22, pady=(20, 6))
        for prompt, hint in EXAMPLES:
            button = self._ghost_button(
                sidebar, f"›  {prompt}", lambda _=None, p=prompt: self._use_prompt(p)
            )
            button.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(sidebar, text=f"   {hint}", text_color=DIM,
                         font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=22, pady=(0, 4))

        bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
        bottom.pack(side="bottom", fill="x")
        self._divider(bottom, pady=(8, 8))
        self._ghost_button(bottom, "GitHub репозитория  ↗",
                           lambda: self._open("https://github.com/nikitawcnu-bit/subash-ai")).pack(
            fill="x", padx=14)
        self._ghost_button(bottom, "Скачать обновления  ↗",
                           lambda: self._open("https://github.com/nikitawcnu-bit/subash-ai/releases")).pack(
            fill="x", padx=14)
        ctk.CTkLabel(bottom, text="Сделано вместе с DeepSeek · v0.2.0",
                     text_color=DIM, font=("Segoe UI", 9)).pack(
            fill="x", padx=22, pady=(10, 18), anchor="w")

        spacer = ctk.CTkFrame(sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

    def _build_main(self):
        ctk = self.ctk
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=26, pady=22)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(main, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Генерация текста", text_color=FG,
                     font=("Segoe UI", 24, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Продолжение в стиле Александра Сергеевича Пушкина",
                     text_color=MUTED, font=("Segoe UI", 12), anchor="w").grid(
            row=1, column=0, sticky="w", pady=(1, 0))
        self.status_pill = ctk.CTkLabel(
            header, text="●  модель загружена", fg_color="#102b3f", text_color="#7dd3fc",
            corner_radius=11, height=30, width=180, font=("Segoe UI", 11),
        )
        self.status_pill.grid(row=0, column=1, rowspan=2, sticky="e")

        # Промпт
        prompt_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=16)
        prompt_card.grid(row=1, column=0, sticky="ew", pady=(20, 0))
        prompt_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(prompt_card, text="НАЧАЛО ФРАЗЫ", text_color=DIM,
                     font=("Segoe UI", 10, "bold"), anchor="w").grid(
            row=0, column=0, sticky="w", padx=20, pady=(16, 0))
        self.prompt_entry = ctk.CTkEntry(
            prompt_card, height=48, corner_radius=10, fg_color=ENTRY,
            border_color=BORDER, border_width=1, text_color=FG,
            placeholder_text="Введите начало фразы…", font=("Segoe UI", 15),
        )
        self.prompt_entry.grid(row=1, column=0, sticky="ew", padx=(20, 12), pady=(8, 18))
        self.prompt_entry.insert(0, DEFAULT_PROMPT)
        self.prompt_entry.bind("<Return>", lambda _e: self.on_generate())

        self.generate_button = ctk.CTkButton(
            prompt_card, text="Продолжить текст", command=self.on_generate,
            width=176, height=48, corner_radius=10, fg_color=ACCENT,
            hover_color=ACCENT_HOVER, text_color="#06121f",
            font=("Segoe UI", 13, "bold"),
        )
        self.generate_button.grid(row=1, column=1, sticky="e", padx=(0, 20), pady=(8, 18))

        # Параметры
        settings = ctk.CTkFrame(main, fg_color=CARD, corner_radius=16)
        settings.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        for column in range(3):
            settings.grid_columnconfigure(column, weight=1)

        self.temp_value = self.tk.StringVar(value="0.75")
        temp_box = self._setting_box(settings, 0, "ТЕМПЕРАТУРА", self.temp_value)
        self.temp_slider = ctk.CTkSlider(
            temp_box, from_=0.1, to=1.3, number_of_steps=24, height=16,
            fg_color=BORDER, progress_color=ACCENT, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, command=self._on_temp,
        )
        self.temp_slider.set(0.75)
        self.temp_slider.pack(fill="x", padx=16, pady=(4, 14))

        self.tokens_value = self.tk.StringVar(value="180")
        tokens_box = self._setting_box(settings, 1, "ДЛИНА, СИМВОЛОВ", self.tokens_value)
        self.tokens_slider = ctk.CTkSlider(
            tokens_box, from_=40, to=400, number_of_steps=18, height=16,
            fg_color=BORDER, progress_color=ACCENT, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, command=self._on_tokens,
        )
        self.tokens_slider.set(180)
        self.tokens_slider.pack(fill="x", padx=16, pady=(4, 14))

        topk_box = self._setting_box(settings, 2, "TOP-K", self.tk.StringVar(value="40"))
        ctk.CTkLabel(topk_box, text="Отсекаем хвост распределения:",
                     text_color=MUTED, font=("Segoe UI", 10), anchor="w").pack(
            fill="x", padx=16, pady=(8, 0))
        ctk.CTkLabel(topk_box, text="каждый шаг — 40 лучших символов",
                     text_color=FG, font=("Segoe UI", 11), anchor="w").pack(
            fill="x", padx=16, pady=(0, 16))

        # Результат
        output_card = ctk.CTkFrame(main, fg_color=CARD, corner_radius=16)
        output_card.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        output_card.grid_columnconfigure(0, weight=1)
        output_card.grid_rowconfigure(2, weight=1)

        out_header = ctk.CTkFrame(output_card, fg_color="transparent")
        out_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))
        out_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(out_header, text="РЕЗУЛЬТАТ", text_color=DIM,
                     font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.count_label = ctk.CTkLabel(out_header, text="0 символов", text_color=DIM,
                                        font=("Segoe UI", 10))
        self.count_label.grid(row=0, column=1, padx=(0, 10))
        self._small_button(out_header, "Копировать", self.on_copy).grid(row=0, column=2, padx=2)
        self._small_button(out_header, "Сохранить", self.on_save).grid(row=0, column=3, padx=2)
        self._small_button(out_header, "Очистить", self.on_clear).grid(row=0, column=4, padx=(2, 0))

        self.progress = ctk.CTkProgressBar(
            output_card, mode="indeterminate", height=3, corner_radius=0,
            fg_color=CARD, progress_color=ACCENT,
        )
        self.progress.grid(row=1, column=0, sticky="ew", padx=18)
        self.progress.grid_remove()

        self.output = ctk.CTkTextbox(
            output_card, fg_color=ENTRY, text_color=FG, corner_radius=10,
            border_width=0, border_spacing=14, wrap="word",
            font=("Georgia", 15), scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        self.output.grid(row=2, column=0, sticky="nsew", padx=16, pady=(2, 6))
        self.output.configure(state="disabled")
        self.placeholder = ctk.CTkLabel(
            output_card, text="Здесь появится продолжение…", text_color=DIM,
            font=("Georgia", 14),
        )
        self.placeholder.place(relx=0.5, rely=0.55, anchor="center")

        footer = ctk.CTkFrame(output_card, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 10))
        footer.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(footer, text="", text_color=MUTED,
                                         font=("Segoe UI", 10), anchor="w")
        self.status_label.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(footer, text="Ctrl+Enter — сгенерировать", text_color=DIM,
                     font=("Segoe UI", 10)).grid(row=0, column=1, sticky="e")

    def _setting_box(self, parent, column, title, variable):
        ctk = self.ctk
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=column, sticky="ew", padx=(4, 4), pady=8)
        head = ctk.CTkFrame(box, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkLabel(head, text=title, text_color=DIM,
                     font=("Segoe UI", 10, "bold"), anchor="w").pack(side="left")
        ctk.CTkLabel(head, textvariable=variable, text_color=ACCENT,
                     font=("Segoe UI", 12, "bold"), anchor="e").pack(side="right")
        return box

    def _small_button(self, parent, text, command):
        ctk = self.ctk
        return ctk.CTkButton(
            parent, text=text, command=command, width=92, height=28,
            fg_color="#16233a", hover_color="#1f3150", text_color=FG,
            corner_radius=8, font=("Segoe UI", 10),
        )

    # ------------------------------------------------------------------ #
    #  Действия
    # ------------------------------------------------------------------ #
    def _open(self, url: str):
        import webbrowser
        webbrowser.open(url)

    def _focus_prompt(self):
        try:
            self.root.focus_force()
        except Exception:
            pass
        self.prompt_entry.focus_set()

    def _use_prompt(self, prompt: str):
        self.prompt_entry.delete(0, "end")
        self.prompt_entry.insert(0, prompt)
        self.prompt_entry.focus_set()

    def _on_temp(self, value: float):
        self.temp_value.set(f"{value:.2f}")

    def _on_tokens(self, value: float):
        self.tokens_value.set(str(int(round(value / 20) * 20)))

    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool):
        self._busy = busy
        if busy:
            self.generate_button.configure(state="disabled", text="Генерирую…")
            self.status_pill.configure(text="●  генерирую…", fg_color="#3a2a12",
                                       text_color="#fbbf24")
            self.progress.grid()
            self.progress.start()
        else:
            self.generate_button.configure(state="normal", text="Продолжить текст")
            self.status_pill.configure(text="●  модель загружена", fg_color="#102b3f",
                                       text_color="#7dd3fc")
            self.progress.stop()
            self.progress.grid_remove()
            self.progress.set(0)

    def on_generate(self):
        if self._busy:
            return
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            self._set_status("Введите начало фразы")
            return
        ids = self.tokenizer.encode_safe(prompt)
        if len(ids) == 0:
            self._set_status("В промпте нет символов, знакомых модели")
            return
        temperature = float(self.temp_slider.get())
        tokens = int(round(self.tokens_slider.get() / 20) * 20)
        tokens = max(40, min(tokens, 400))

        self._set_busy(True)
        self._set_status("Модель придумывает продолжение…")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.placeholder.place_forget()
        self.count_label.configure(text="…")
        threading.Thread(
            target=self._worker, args=(prompt, ids, temperature, tokens), daemon=True
        ).start()

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
        self._show_text(text)
        self._set_status(f"Готово за {elapsed:.1f} с · {symbols_text(len(text))}")

    def _failed(self, error):
        from tkinter import messagebox

        self._set_busy(False)
        self._set_status("Ошибка генерации")
        messagebox.showerror("Subash AI", error[-1200:])

    def _show_text(self, text: str):
        self._last_text = text
        self.placeholder.place_forget()
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")
        self.output.see("end")
        self.count_label.configure(text=symbols_text(len(text)))

    def on_clear(self):
        self._last_text = ""
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        self.placeholder.place(relx=0.5, rely=0.55, anchor="center")
        self.count_label.configure(text="0 символов")
        self._set_status("Очищено")

    def on_copy(self):
        text = self.output.get("1.0", "end").strip()
        if not text:
            self._set_status("Нечего копировать")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("Скопировано в буфер обмена")

    def on_save(self):
        from tkinter import filedialog

        text = self.output.get("1.0", "end").strip()
        if not text:
            self._set_status("Нечего сохранять")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="subash.txt",
            filetypes=[("Текст", "*.txt"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")
        self._set_status(f"Сохранено: {Path(path).name}")


def main() -> None:
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    model, tokenizer = load_model()
    root = ctk.CTk()
    App(root, model, tokenizer)
    root.mainloop()


if __name__ == "__main__":
    main()
