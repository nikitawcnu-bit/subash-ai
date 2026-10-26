"""Проверки autograd: аналитические градиенты сверяются с численными."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash.tensor import Tensor, cross_entropy, no_grad, softmax  # noqa: E402


def numeric_grad(fn, arrays, eps=1e-6):
    grads = []
    for k, array in enumerate(arrays):
        grad = np.zeros_like(array, dtype=np.float64)
        for index in np.ndindex(*array.shape):
            plus = [a.copy() for a in arrays]
            minus = [a.copy() for a in arrays]
            plus[k][index] += eps
            minus[k][index] -= eps
            f_plus = float(fn(*[Tensor(a) for a in plus]).data)
            f_minus = float(fn(*[Tensor(a) for a in minus]).data)
            grad[index] = (f_plus - f_minus) / (2 * eps)
        grads.append(grad)
    return grads


class GradcheckTest(unittest.TestCase):
    def check(self, fn, *arrays, tol=1e-5):
        arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
        tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        out = fn(*tensors)
        self.assertEqual(out.size, 1, "функция должна возвращать скаляр")
        out.backward()
        analytic = [t.grad for t in tensors]
        numeric = numeric_grad(fn, arrays)
        for k, (a, n) in enumerate(zip(analytic, numeric)):
            self.assertIsNotNone(a, f"нет градиента для аргумента {k}")
            np.testing.assert_allclose(a, n, rtol=tol, atol=tol,
                                       err_msg=f"аргумент {k}")

    # --- поэлементные операции ---------------------------------------- #
    def test_add_broadcast(self):
        rng = np.random.default_rng(0)
        self.check(lambda a, b: (a + b).sum(),
                   rng.normal(size=(3, 4)), rng.normal(size=(4,)))

    def test_add_scalar(self):
        rng = np.random.default_rng(0)
        self.check(lambda a: (a + 2.5).sum(), rng.normal(size=(5,)))

    def test_sub_mul_div(self):
        rng = np.random.default_rng(1)
        self.check(lambda a, b: ((a - b) * a / (b + 3.0)).sum(),
                   rng.normal(size=(4, 3)), rng.normal(size=(3,)))

    def test_pow(self):
        rng = np.random.default_rng(2)
        self.check(lambda a: (a ** 3).sum(), rng.normal(size=(4,)))

    def test_exp_log_sqrt(self):
        rng = np.random.default_rng(3)
        self.check(lambda a: (a.exp() + (a * a + 0.5).log()).sum(), rng.normal(size=(4,)))
        self.check(lambda a: (a * a + 1.0).sqrt().sum(), rng.normal(size=(4,)))

    def test_tanh_sigmoid_relu(self):
        rng = np.random.default_rng(4)
        self.check(lambda a: a.tanh().sum(), rng.normal(size=(6,)))
        self.check(lambda a: a.sigmoid().sum(), rng.normal(size=(6,)))
        self.check(lambda a: a.relu().sum(), rng.normal(size=(6,)))

    def test_neg(self):
        rng = np.random.default_rng(5)
        self.check(lambda a: (-a).sum(), rng.normal(size=(3, 2)))

    # --- формы --------------------------------------------------------- #
    def test_matmul(self):
        rng = np.random.default_rng(6)
        self.check(lambda a, b: (a @ b).sum(),
                   rng.normal(size=(3, 4)), rng.normal(size=(4, 5)))
        self.check(lambda a, b: (a @ b).sum(),
                   rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 4, 5)))

    def test_sum_mean_axis(self):
        rng = np.random.default_rng(7)
        self.check(lambda a: (a.sum(axis=1) * 3.0).sum(), rng.normal(size=(4, 5)))
        self.check(lambda a: (a.mean(axis=0, keepdims=True) ** 2).sum(),
                   rng.normal(size=(4, 5)))
        self.check(lambda a: a.mean(), rng.normal(size=(3, 3)))

    def test_reshape_transpose(self):
        rng = np.random.default_rng(8)
        self.check(lambda a: (a.reshape(6, 4) * 2.0).sum(), rng.normal(size=(2, 3, 4)))
        self.check(lambda a: (a.transpose(0, 2, 1) * 2.0).sum(), rng.normal(size=(2, 3, 4)))
        self.check(lambda a: (a.T * 2.0).sum(), rng.normal(size=(3, 4)))

    def test_swapaxes_batched_matmul(self):
        rng = np.random.default_rng(9)
        self.check(lambda a, b: (a.swapaxes(-1, -2) @ b).sum(),
                   rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 5)))

    def test_getitem_slice_and_gather(self):
        rng = np.random.default_rng(10)
        self.check(lambda a: a[1:3].sum(), rng.normal(size=(4, 3)))
        ids = np.array([[0, 2], [3, 1]])

        def gather(a):
            return (a[ids] * 2.0).sum()

        self.check(gather, rng.normal(size=(4, 3)))

    # --- составные ----------------------------------------------------- #
    def test_diamond_accumulation(self):
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        loss = (x * x + x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, 2 * x.data + 1)

    def test_softmax_rows_sum_to_one(self):
        x = Tensor(np.random.default_rng(11).normal(size=(4, 7)))
        y = softmax(x, axis=-1)
        np.testing.assert_allclose(y.data.sum(axis=-1), np.ones(4), rtol=1e-6)

    def test_softmax_grad(self):
        rng = np.random.default_rng(12)

        def fn(a):
            return (softmax(a * 2.0, axis=-1) ** 2).sum()

        self.check(fn, rng.normal(size=(4, 5)))

    def test_cross_entropy_value_and_grad(self):
        rng = np.random.default_rng(13)
        logits = rng.normal(size=(4, 6))
        targets = np.array([0, 3, 5, 1])

        got = cross_entropy(Tensor(logits), targets).item()

        z = logits - logits.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        expected = -logp[np.arange(4), targets].mean()
        self.assertAlmostEqual(got, expected, places=6)

        def fn(a):
            return cross_entropy(a, targets)

        self.check(fn, logits, tol=1e-4)

    def test_cross_entropy_3d(self):
        rng = np.random.default_rng(14)
        logits = rng.normal(size=(2, 3, 5))
        targets = rng.integers(0, 5, size=(2, 3))

        def fn(a):
            return cross_entropy(a, targets)

        self.check(fn, logits, tol=1e-4)

    def test_layernorm_like_composite(self):
        rng = np.random.default_rng(15)

        def fn(a, g, b):
            x = (a - a.mean(axis=-1, keepdims=True)) / (a.sum(axis=-1, keepdims=True) * 0 + 2.0)
            return (x * g + b).sum()

        self.check(fn, rng.normal(size=(3, 4)), rng.normal(size=(4,)), rng.normal(size=(4,)))

    # --- режимы -------------------------------------------------------- #
    def test_no_grad(self):
        with no_grad():
            x = Tensor(np.ones(2), requires_grad=True)
            y = x * 2.0
        self.assertFalse(y.requires_grad)

    def test_requires_grad_propagation(self):
        a = Tensor(np.ones(2), requires_grad=True)
        b = Tensor(np.ones(2))
        self.assertTrue((a + b).requires_grad)
        self.assertFalse((b * 2.0).requires_grad)

    def test_backward_scalar_only(self):
        x = Tensor(np.ones((2, 2)), requires_grad=True)
        with self.assertRaises(RuntimeError):
            (x * 2).backward()


if __name__ == "__main__":
    unittest.main()
