# Subash AI

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![tests](https://github.com/nikitawcnu-bit/subash-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/nikitawcnu-bit/subash-ai/actions)
![NumPy](https://img.shields.io/badge/NumPy-only-orange)

**Своя нейросеть на Python с нуля** — без PyTorch, TensorFlow и JAX. Только NumPy,
чистая математика и понятный код: движок автоматического дифференцирования, слои,
оптимизаторы, MLP и мини-трансформер (GPT), который учится писать в стиле Пушкина.

> **Как создавался проект.** Subash AI сделан в паре с AI-ассистентом
> [DeepSeek](https://www.deepseek.com/): постановка задач и приёмка результата — за
> автором, код, обучение моделей и документация — за ассистентом. Ни одной строки
> PyTorch: только NumPy и математика.

```
Subash/
├── subash/            # сама библиотека (~1000 строк)
│   ├── tensor.py      # autograd: Tensor, операции, backward, softmax, cross-entropy
│   ├── nn.py          # Module, Linear, Embedding, LayerNorm, Dropout, активации
│   ├── models.py      # MLP и TinyGPT (multi-head causal attention)
│   ├── optim.py       # SGD с моментом, AdamW, clip_grad_norm, cosine_lr
│   ├── data.py        # char-level токенизатор, батчи, спираль/XOR/луны
│   ├── checkpoint.py  # сохранение/загрузка модели в один .npz
│   └── utils.py       # сид, подсчёт параметров, таймер
├── examples/          # обучение и генерация
├── tests/             # 44 теста, включая численную проверку градиентов
├── scripts/           # сборка корпуса из Викитеки
└── data/pushkin.txt   # корпус: 5 сказок А. С. Пушкина, 53 609 символов
```

## Быстрый старт

Нужен Python 3.10+ и NumPy.

```bash
pip install -r requirements.txt

# 1) обучить MLP на спирали (~7 секунд, точность ~97%)
python examples/train_mlp.py

# 2) обучить мини-GPT на сказках (~16 минут на CPU)
python examples/train_gpt.py

# 3) сгенерировать текст
python examples/generate.py --prompt "Три девицы под окном" --tokens 250

# 4) интерактивный режим
python examples/chat.py

# тесты
python -m unittest discover -s tests
```

## Результаты

**MLP на спирали** (2 → 128 → 128 → 3, 17 283 параметра, 3000 шагов, ~7 с на CPU):
точность на отложенной выборке **97%**. Точки — данные, фон — предсказание сети:

![Спираль: данные и границы классов](assets/spiral.png)

**Мини-GPT** (2 слоя, 4 головы, 96 эмбеддингов, 237 408 параметров, 1800 шагов,
~16 минут на CPU): финальный val loss **2.69** (perplexity 14.7), лучший val loss
**2.47** на 450-м шаге — дальше модель переобучается, потому что корпус маленький.

Пример генерации (temperature 0.6, top-k 20, промпт «Князь у синя моря ходит»):

```
Князь у синя моря ходит,
А видит опит вестает,
Судно весело бежит
Мимо острова Буяна,
В царст
```

Модель выучила ритм, пунктуацию, частые слова и целые куски («Судно весело бежит»,
«Мимо острова Буяна»), но на 53 тысячах символов смысл не держится. Это ожидаемо:
для связного текста нужен корпус в десятки мегабайт и модель побольше.

![Кривая обучения мини-GPT](assets/gpt_loss.png)

Больше примеров с разных шагов — в `checkpoints/samples.txt` (создаётся при обучении).

## Как это работает

### 1. Autograd

`Tensor` хранит массив NumPy, градиент и ссылки на родителей. Каждая операция
создаёт новый узел и функцию обратного распространения; `backward()` обходит граф
в обратном топологическом порядке.

```python
from subash import Tensor

x = Tensor([[1.0, 2.0]], requires_grad=True)
w = Tensor([[0.5], [-1.0]], requires_grad=True)

loss = ((x @ w) ** 2).sum()
loss.backward()

print(x.grad)   # [[-1.5  3. ]]
print(w.grad)   # [[-3.] [-6.]]
```

Поддерживаются broadcasting, batched `matmul`, срезы, `sum/mean`, `reshape`,
`transpose`, `exp/log/sqrt/tanh/sigmoid/relu`, `softmax` и fused `cross_entropy`
(softmax + NLL одним узлом, чтобы не терять точность). Режим `no_grad()`
отключает построение графа при генерации.

Корректность градиентов проверяется численно: в `tests/test_tensor.py` каждая
операция сравнивается с центральной конечной разностью (float64, точность 1e-5).

### 2. Слои и оптимизация

* `Linear` — инициализация He/Xavier, смещение;
* `Embedding` — таблица токенов, градиент собирается через `np.add.at`;
* `LayerNorm` — нормализация по последней оси;
* `Dropout` — переключается `model.train() / model.eval()`;
* `AdamW` — Adam с развязанным weight decay, `cosine_lr` с разогревом,
  обрезка нормы градиента.

### 3. Мини-GPT

Классический decoder-only трансформер:

```
токены → Embedding + позиции
       → [LayerNorm → Causal Self-Attention → +residual
          LayerNorm → MLP(GELU) → +residual] × N
       → LayerNorm → Linear → логиты по словарю
```

Причинная маска не даёт смотреть в будущее; веса выходного слоя связаны с
эмбеддингами (`weight tying`). Обучение — предсказание следующего символа,
loss — cross-entropy, генерация — сэмплирование с temperature / top-k / top-p.

## Обучение на своём тексте

Положите любой `.txt` в кодировке UTF-8 и запустите:

```bash
python examples/train_gpt.py --data my_text.txt --steps 3000 --n-embd 128 --n-layer 3

# продолжить обучение с сохранённых весов
python examples/train_gpt.py --init-from checkpoints/gpt_pushkin.npz --steps 800 --lr 1e-3
```

Словарь строится по символам автоматически и сохраняется вместе с весами.
Для корпуса побольше увеличьте `--n-embd`, `--n-layer` и `--steps`; на CPU это
линейно замедляет обучение.

Пересобрать корпус Пушкина (нужен `beautifulsoup4` и интернет):

```bash
python scripts/build_corpus.py
```

## Тесты

```bash
python -m unittest discover -s tests -v
```

44 теста: численная проверка всех операций autograd, broadcasting и batched
matmul, сборка градиентов в embedding, режимы dropout, AdamW на квадратичной
функции, обучение XOR, переобучение крошечного GPT, round-trip чекпойнта.

## Ограничения

* Только CPU и NumPy: модель маленькая, это учебный проект, а не замена PyTorch.
* Корпус — 53 тыс. символов, поэтому GPT частично запоминает текст и иногда «плывёт».
* Токенизация посимвольная; для больших корпусов нужен BPE.
* Нет GPU, mixed precision и распределённого обучения.

## Планы

* BPE-токенизатор и корпус побольше;
* веб-интерфейс для генерации;
* профилирование и ускорение matmul;
* сохранение лучшего чекпойнта по val loss.

## Репозиторий

Обученные веса (`checkpoints/mlp_spiral.npz` и `checkpoints/gpt_pushkin.npz`,
около 1 МБ) лежат в репозитории, поэтому после клонирования генерация работает
сразу, без 16 минут обучения. Логи и кеши исключены через `.gitignore`, а тесты
автоматически запускаются в GitHub Actions (`.github/workflows/tests.yml`).

## Данные и лицензия

Тексты — сказки А. С. Пушкина с [ru.wikisource.org](https://ru.wikisource.org),
общественное достояние (шаблон `PD-old`). Код — MIT, см. [LICENSE](LICENSE).
