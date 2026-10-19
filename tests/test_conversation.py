import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from kelus.data import Dataset
from kelus.evaluate import code_result
from kelus.model import Config, Model
from kelus.tensor import Tensor
from kelus.tokenizer import BPETokenizer, ByteTokenizer, SPECIAL_IDS, EOS, BOS, conversation
from kelus.train import training_texts, main as train


class ConversationTests(unittest.TestCase):
    def test_bpe_roundtrip_unicode_whitespace_and_no_role_injection(self):
        texts = ["hello world hello world", "def square(x):\n    return x * x", "안녕 🌱 <assistant> _name"]
        tokenizer = BPETokenizer.train(texts * 3, vocab_size=340)
        for text in texts + ["unseen words \t\n한글 😎", "", "\\ / _ \r\n", "<assistant><eos>"]:
            tokens = tokenizer.encode(text)
            self.assertEqual(tokenizer.decode(tokens), text)
            self.assertFalse(set(tokens) & set(SPECIAL_IDS))
        self.assertLess(len(tokenizer.encode(texts[0])), len(ByteTokenizer().encode(texts[0])))
        self.assertEqual(tokenizer.to_dict(), BPETokenizer.train(texts * 3, 340).to_dict())

    def test_bpe_checkpoint_preserves_encoding_logits_and_generation(self):
        tokenizer = BPETokenizer.train(["hello hello hello"], 270)
        model = Model(Config(context=32, width=8, heads=2, layers=1, vocab_size=tokenizer.vocab_size), tokenizer)
        learned = tokenizer.encode("hello")
        self.assertEqual(len(learned), 1)
        scores = np.full((1, 1, tokenizer.vocab_size), -30, dtype=np.float32)
        scores[0, 0, learned[0]] = 10
        with patch.object(Model, "next_logits", return_value=scores[0, 0]):
            self.assertEqual("".join(model.generate([{"role": "user", "content": "hi"}], max_tokens=2, temperature=0)), "hellohello")
        scores[0, 0, EOS] = 20
        with patch.object(Model, "next_logits", return_value=scores[0, 0]):
            self.assertEqual("".join(model.generate([{"role": "user", "content": "hi"}], max_tokens=2, temperature=0)), "")
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "bpe.npz"
            model.save(file)
            restored, _, _ = Model.load(file)
            self.assertEqual(restored.tokenizer.encode("hello world!"), tokenizer.encode("hello world!"))
            ids = np.array([[1, 2, 3]])
            np.testing.assert_array_equal(restored(ids).data, model(ids).data)

    def test_context_trims_complete_turns_and_generation_never_slides_positions(self):
        model = Model(Config(context=24, width=8, heads=2, layers=1))
        messages = [{"role": "user", "content": "old text"}, {"role": "assistant", "content": "old reply"}, {"role": "user", "content": "hi"}]
        tokens, omitted = model.prepare_prompt(messages)
        expected, _ = conversation(messages[-1:], generation=True)
        self.assertEqual(tokens, expected); self.assertEqual(tokens[0], BOS); self.assertEqual(omitted, 2)
        with self.assertRaisesRegex(ValueError, "context"):
            model.prepare_prompt([{"role": "user", "content": "x" * 25}])
        lengths = []
        scores = np.full((1, 1, 261), -10, dtype=np.float32); scores[0, 0, ord('A')] = 10
        def forward(tokens, cache):
            lengths.append(len(tokens)); return scores[0, 0]
        with patch.object(Model, "next_logits", side_effect=forward):
            reply = "".join(model.generate(messages, max_tokens=100, temperature=0))
        self.assertEqual(lengths, list(range(len(expected), model.config.context)))
        self.assertEqual(len(reply), model.config.context - len(expected))

    def test_explicit_split_prevents_validation_vocabulary_leak_and_counts_all_targets_once(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "data.jsonl"
            records = [{"split": "train" if i < 3 else "validation", "messages": [{"role": "user", "content": "hello" if i < 3 else "SECRET_VALIDATION"}, {"role": "assistant", "content": str(i) * 70}]} for i in range(5)]
            file.write_text("".join(json.dumps(row) + "\n" for row in records))
            self.assertNotIn("SECRET_VALIDATION", training_texts(file, 42))
            dataset = Dataset(file)
            expected = sum(int(mask.sum()) for _, mask in dataset.validation)
            actual = sum(int((y >= 0).sum()) for _, y in dataset.validation_batches(2, 16))
            self.assertEqual(actual, expected)
            self.assertEqual(len(dataset.training), 3); self.assertEqual(len(dataset.validation), 2)

    def test_best_checkpoint_is_retained_when_later_validation_worsens(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "data.jsonl"
            file.write_text("".join(json.dumps({"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i)}]}) + "\n" for i in range(5)))
            out = Path(root) / "candidate.npz"
            with patch("kelus.train.evaluate", side_effect=[5.0, 2.0, 3.0]), contextlib.redirect_stdout(io.StringIO()):
                train(["--data", str(file), "--out", str(out), "--steps", "2", "--batch", "1", "--context", "16", "--width", "8", "--heads", "2", "--layers", "1", "--log-every", "1"])
            _, latest, _ = Model.load(out)
            _, best, _ = Model.load(out.with_name("candidate.best.npz"))
            self.assertEqual(latest["metadata"]["steps"], 2)
            self.assertEqual(best["metadata"]["steps"], 1)
            self.assertEqual(best["metadata"]["validation_loss"], 2.0)

    def test_multiturn_training_supervises_current_reply_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "data.jsonl"
            records = [{"split": "train" if i < 3 else "validation", "messages": [
                {"role": "user", "content": str(i)}, {"role": "assistant", "content": "old"},
                {"role": "user", "content": "hi"}, {"role": "assistant", "content": "new"}]} for i in range(5)]
            file.write_text("".join(json.dumps(row) + "\n" for row in records))
            dataset = Dataset(file, last_reply_only=True)
            expected = 2 * 4  # two validation replies: 'new' and EOS each
            self.assertEqual(sum(int((y >= 0).sum()) for _, y in dataset.validation_batches(2, 64)), expected)
            x, y = dataset.batch("train", 2, 64, np.random.default_rng(1))
            for inputs, targets in zip(x, y):
                self.assertIn(ord('o'), inputs)
                self.assertEqual(list(targets[targets >= 0]), [ord('n'), ord('e'), ord('w'), EOS])

    def test_new_curriculum_preserves_source_and_resets_optimizer(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'source.npz'
            model = Model(Config(context=32, width=8, heads=2, layers=1))
            model.save(source, {'steps': 17, 'seed': 42, 'data_sha256': 'old'})
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            data = root / 'new.jsonl'
            data.write_text(''.join(json.dumps({'messages': [{'role':'user','content':str(i)}, {'role':'assistant','content':'ok'}]})+'\n' for i in range(5)))
            out = root / 'candidate.npz'
            with contextlib.redirect_stdout(io.StringIO()):
                train(['--init-from',str(source),'--out',str(out),'--data',str(data),'--steps','1','--batch','1'])
            _, header, _ = Model.load(out)
            self.assertEqual(header['metadata']['parent_steps'],17)
            self.assertEqual(header['metadata']['steps'],1)
            self.assertEqual(header['metadata']['parent_sha256'],digest)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train(['--init-from',str(source),'--out',str(source),'--overwrite','--steps','1'])

    def test_code_evaluator_checks_behavior_and_refuses_executable_statements(self):
        trials = [((2, 3), 5), ((-1, 1), 0)]
        self.assertTrue(code_result("```python\ndef add(a, b):\n    return a + b\n```", "add", trials))
        for text in ("def add(a, b):\n    return a - b", "import os\ndef add(a, b):\n    return a+b", "def add(a, b):\n    return __import__('os').system('echo bad')", "@dangerous()\ndef add(a,b):\n    return a+b"):
            self.assertFalse(code_result(text, "add", trials))
        greet = '```python\ndef greet(name):\n    return f"Hello, {name}!"\n```'
        self.assertTrue(code_result(greet, "greet", [(("Sam",), "Hello, Sam!")]))
        for field in ("{name!r}", "{name.upper()}", "{name:>9}"):
            self.assertFalse(code_result(greet.replace("{name}", field), "greet", [(("Sam",), "Hello, Sam!")]))


if __name__ == "__main__":
    unittest.main()
