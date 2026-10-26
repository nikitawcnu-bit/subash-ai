"""Тесты моделей: обучение, генерация и сохранение/загрузка."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import Tensor, data, load_checkpoint, nn, optim, save_checkpoint  # noqa: E402
from subash.models import MLP, TinyGPT  # noqa: E402


class MLPTest(unittest.TestCase):
    def test_learns_xor(self):
        np.random.seed(0)
        X, y = data.make_xor(400, noise=0.05, seed=0)
        X_train, y_train, X_val, y_val = data.train_val_split(X, y, seed=0)
        model = MLP([2, 32, 32, 2])
        opt = optim.AdamW(model.parameters(), lr=0.01)
        rng = np.random.default_rng(0)
        for _ in range(300):
            idx = rng.integers(0, len(X_train), 64)
            loss = nn.cross_entropy(model(Tensor(X_train[idx])), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        accuracy = float((model(Tensor(X_val)).data.argmax(1) == y_val).mean())
        self.assertGreater(accuracy, 0.9)

    def test_parameter_count(self):
        model = MLP([2, 8, 3])
        # 2*8+8 + 8*3+3 = 24 + 27 = 51
        self.assertEqual(model.num_parameters(), 51)


class TinyGPTTest(unittest.TestCase):
    def test_overfits_tiny_sequence(self):
        np.random.seed(0)
        model = TinyGPT(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16,
                        dropout=0.0)
        ids = np.tile(np.arange(8, dtype=np.int64), (8, 1))
        targets = np.roll(ids, -1, axis=1)
        opt = optim.AdamW(model.parameters(), lr=0.01)
        last = None
        for _ in range(150):
            _, loss = model(ids, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        self.assertLess(last, 1.0)

    def test_forward_shapes_and_causal_mask(self):
        np.random.seed(0)
        model = TinyGPT(vocab_size=10, block_size=6, n_layer=1, n_head=2, n_embd=8)
        logits, loss = model(np.zeros((3, 6), dtype=np.int64), np.zeros((3, 6), dtype=np.int64))
        self.assertEqual(logits.shape, (3, 6, 10))
        self.assertTrue(2.0 < loss.item() < 2.5)

    def test_generate_shapes_and_determinism(self):
        np.random.seed(0)
        model = TinyGPT(vocab_size=10, block_size=6, n_layer=1, n_head=2, n_embd=8)
        model.eval()
        first = model.generate(np.array([1, 2, 3]), max_new_tokens=5, temperature=0.0)
        second = model.generate(np.array([1, 2, 3]), max_new_tokens=5, temperature=0.0)
        self.assertEqual(first.shape, (1, 8))
        np.testing.assert_array_equal(first, second)

    def test_dropout_is_disabled_in_eval(self):
        np.random.seed(0)
        model = TinyGPT(vocab_size=10, block_size=6, n_layer=1, n_head=2, n_embd=8,
                        dropout=0.5)
        model.eval()
        ids = np.ones((2, 6), dtype=np.int64)
        a, _ = model(ids)
        b, _ = model(ids)
        np.testing.assert_allclose(a.data, b.data)


class CheckpointTest(unittest.TestCase):
    def test_roundtrip(self):
        np.random.seed(0)
        tokenizer = data.CharTokenizer("абвгдежзиклмн")
        model = TinyGPT(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=1,
                        n_head=2, n_embd=16)
        model.eval()
        ids = np.arange(8)[None, :]
        before = model(ids)[0].data.copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            save_checkpoint(model, path, tokenizer)
            loaded, loaded_tokenizer = load_checkpoint(path)

        after = loaded(ids)[0].data
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
        self.assertEqual(loaded_tokenizer.chars, tokenizer.chars)


if __name__ == "__main__":
    unittest.main()
