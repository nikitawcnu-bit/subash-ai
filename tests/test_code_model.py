"""Проверка обученного чекпойнта код-модели (если он есть в репозитории)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subash import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class CodeCheckpointTest(unittest.TestCase):
    def test_has_at_least_500k_parameters_and_generates(self):
        path = ROOT / "checkpoints" / "gpt_code.npz"
        if not path.exists():
            self.skipTest("чекпойнт код-модели ещё не обучен")
        model, tokenizer = load_checkpoint(path)
        self.assertGreaterEqual(model.num_parameters(), 500_000)
        self.assertGreater(tokenizer.vocab_size, 100)
        ids = tokenizer.encode_safe("def ")
        output = model.generate(ids, max_new_tokens=8, temperature=0.5, top_k=20)
        self.assertEqual(output.shape, (1, len(ids) + 8))


if __name__ == "__main__":
    unittest.main()
