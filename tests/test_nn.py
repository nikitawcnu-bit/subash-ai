"""Тесты слоёв, оптимизаторов и данных."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import Tensor, data, nn, optim  # noqa: E402


class LayerTest(unittest.TestCase):
    def test_linear_shapes_and_grads(self):
        np.random.seed(0)
        layer = nn.Linear(3, 4)
        x = Tensor(np.random.randn(5, 3).astype(np.float32))
        y = layer(x)
        self.assertEqual(y.shape, (5, 4))
        loss = (y * y).sum()
        loss.backward()
        self.assertEqual(layer.weight.grad.shape, (3, 4))
        self.assertEqual(layer.bias.grad.shape, (4,))

    def test_linear_without_bias(self):
        layer = nn.Linear(2, 3, bias=False)
        self.assertIsNone(layer.bias)
        self.assertEqual(layer(Tensor(np.ones((4, 2), dtype=np.float32))).shape, (4, 3))

    def test_layernorm_normalizes(self):
        np.random.seed(0)
        layer = nn.LayerNorm(6)
        x = Tensor(np.random.randn(10, 6).astype(np.float32) * 3 + 5)
        y = layer(x).data
        np.testing.assert_allclose(y.mean(axis=-1), np.zeros(10), atol=1e-4)
        np.testing.assert_allclose(y.std(axis=-1), np.ones(10), atol=1e-3)

    def test_embedding_gather_and_grad(self):
        np.random.seed(0)
        emb = nn.Embedding(5, 3, std=0.1)
        ids = np.array([[0, 2], [2, 2]])
        out = emb(ids)
        self.assertEqual(out.shape, (2, 2, 3))
        np.testing.assert_allclose(out.data[0, 0], emb.weight.data[0])
        out.sum().backward()
        # id=2 встречается трижды, id=0 — один раз
        np.testing.assert_allclose(emb.weight.grad[2], [3.0, 3.0, 3.0])
        np.testing.assert_allclose(emb.weight.grad[0], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(emb.weight.grad[1], [0.0, 0.0, 0.0])

    def test_dropout_modes(self):
        np.random.seed(0)
        x = Tensor(np.ones((200, 200), dtype=np.float32))
        drop = nn.Dropout(0.5)
        drop.eval()
        np.testing.assert_allclose(drop(x).data, 1.0)
        drop.train()
        out = drop(x).data
        self.assertTrue(np.all((out == 0.0) | (np.isclose(out, 2.0))))
        self.assertAlmostEqual(float((out == 0.0).mean()), 0.5, delta=0.02)

    def test_sequential(self):
        net = nn.Sequential(nn.Linear(2, 4), nn.Activation("tanh"), nn.Linear(4, 1))
        self.assertEqual(net(Tensor(np.ones((3, 2), dtype=np.float32))).shape, (3, 1))
        self.assertEqual(len(net.parameters()), 4)

    def test_mse_loss(self):
        pred = Tensor(np.array([1.0, 2.0]))
        loss = nn.mse_loss(pred, np.array([1.0, 4.0]))
        self.assertAlmostEqual(loss.item(), 2.0)


class OptimizerTest(unittest.TestCase):
    def test_adamw_minimizes_quadratic(self):
        x = Tensor(np.array([5.0, -3.0], dtype=np.float32), requires_grad=True)
        opt = optim.AdamW([x], lr=0.1)
        for _ in range(500):
            loss = (x * x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(np.abs(x.data).max()), 0.01)

    def test_sgd_momentum(self):
        x = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
        opt = optim.SGD([x], lr=0.1, momentum=0.9)
        for _ in range(20):
            loss = (x * x).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(x.data[0]), 0.5)

    def test_clip_grad_norm(self):
        a = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
        a.grad = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        norm = optim.clip_grad_norm_([a], max_norm=1.0)
        self.assertAlmostEqual(norm, 5.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(a.grad)), 1.0, places=5)

    def test_cosine_lr(self):
        lr = optim.cosine_lr(0, 100, 1e-3, warmup=10)
        self.assertAlmostEqual(lr, 1e-4)
        self.assertAlmostEqual(optim.cosine_lr(50, 100, 1e-3, warmup=10, min_lr=0.0),
                               5.868e-4, places=6)
        self.assertAlmostEqual(optim.cosine_lr(100, 100, 1e-3, warmup=10), 0.0, places=6)


class DataTest(unittest.TestCase):
    def test_tokenizer_roundtrip(self):
        text = "Привет, Subash! 123"
        tok = data.CharTokenizer(text)
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)
        self.assertEqual(tok.vocab_size, len(set(text)))
        with self.assertRaises(ValueError):
            tok.encode("нет такого символа ☃")
        self.assertEqual(len(tok.encode_safe("ет ☃")), 3)

    def test_tokenizer_from_dict(self):
        tok = data.CharTokenizer("абв")
        restored = data.CharTokenizer.from_dict(tok.to_dict())
        self.assertEqual(tok.encode("абв").tolist(), restored.encode("абв").tolist())

    def test_text_dataset_batches(self):
        text = ("абвгдежзик" * 30)
        ds = data.TextDataset(text, val_fraction=0.2)
        rng = np.random.default_rng(0)
        x, y = ds.get_batch("train", batch_size=4, block_size=8, rng=rng)
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        for row in range(4):
            self.assertTrue(np.array_equal(x[row, 1:], y[row, :-1]))

    def test_spiral_and_split(self):
        X, y = data.make_spiral(50, 3, noise=0.02, seed=1)
        self.assertEqual(X.shape, (150, 2))
        self.assertEqual(set(np.unique(y)), {0, 1, 2})
        X_train, y_train, X_val, y_val = data.train_val_split(X, y, val_fraction=0.2, seed=0)
        self.assertEqual(len(X_train) + len(X_val), 150)
        self.assertEqual(len(y_train), len(X_train))

    def test_xor(self):
        X, y = data.make_xor(100, noise=0.0, seed=0)
        np.testing.assert_array_equal(((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int), y)


if __name__ == "__main__":
    unittest.main()
