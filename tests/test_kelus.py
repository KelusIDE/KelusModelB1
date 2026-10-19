import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import http.client
from http.server import ThreadingHTTPServer
import threading
import contextlib
import io
import hashlib
import numpy as np
from kelus.tensor import Tensor, cross_entropy, embedding, no_grad
from kelus.model import Config, Model
from kelus.optim import AdamW
from kelus.data import Dataset
from kelus.tokenizer import encode, decode, conversation, ASSISTANT, EOS
from kelus.server import validate_chat, make_handler


class TensorTests(unittest.TestCase):
    def check_gradient(self, value, function, tolerance=1e-5):
        tensor = Tensor(np.array(value, dtype=np.float64), True)
        loss = function(tensor)
        loss.backward()
        for index in np.ndindex(tensor.data.shape):
            before = tensor.data[index]
            tensor.data[index] = before + 1e-5
            plus = float(function(tensor).data)
            tensor.data[index] = before - 1e-5
            minus = float(function(tensor).data)
            tensor.data[index] = before
            self.assertAlmostEqual(float(tensor.grad[index]), (plus - minus) / 2e-5, delta=tolerance)

    def test_broadcasting_reductions_and_repeated_use(self):
        self.check_gradient([[0.2, 0.4], [0.6, 0.8]], lambda x: ((x - x.mean(-1, True)) ** 2).mean())
        self.check_gradient([[0.2], [0.5]], lambda x: (x * np.array([[1., 2., 3.]]) + x).sum())

    def test_batched_matmul_transpose_reshape_and_softmax(self):
        rng = np.random.default_rng(3)
        matrix = Tensor(rng.normal(size=(2, 3, 2)))
        target = Tensor(rng.normal(size=(2, 2, 2)))
        self.check_gradient(rng.normal(size=(2, 2, 3)), lambda x: (((x @ matrix).transpose(0, 2, 1).softmax() * target).reshape(2, 4)).sum())
        self.check_gradient(rng.normal(size=(3, 2)), lambda x: ((Tensor(rng.normal(size=(1, 1, 1))) * 0 + Tensor(np.ones((2, 2, 3)))) @ x).sum())

    def test_gelu_and_normalization(self):
        self.check_gradient([[-0.4, 0.3, 0.8]], lambda x: (0.5 * x * (1 + (0.7978845608 * (x + 0.044715 * x ** 3)).tanh())).sum())
        self.check_gradient([[0.1, 0.3, 0.8]], lambda x: ((x - x.mean(-1, True)) * (((x - x.mean(-1, True)) ** 2).mean(-1, True) + 1e-5) ** -0.5 * np.array([[1., 2., 3.]])).sum(), tolerance=1e-4)

    def test_embedding_accumulates_repeated_indices(self):
        weight = Tensor(np.zeros((3, 2)), True)
        embedding(weight, np.array([1, 1, 2])).sum().backward()
        np.testing.assert_array_equal(weight.grad, [[0, 0], [2, 2], [1, 1]])

    def test_cross_entropy_gradient_and_mask(self):
        labels = np.array([[1, -1]])
        self.check_gradient([[[0.1, 0.5, -0.2], [0.7, -0.1, 0.3]]], lambda x: cross_entropy(x, labels))
        x = Tensor(np.array([[[1000., -1000.]]]), True)
        loss = cross_entropy(x, np.array([[1]]))
        self.assertTrue(np.isfinite(loss.data)); loss.backward()
        np.testing.assert_array_equal(x.grad, [[[1, -1]]])

    def test_no_grad_releases_tracking_and_second_backward_is_rejected(self):
        x = Tensor(np.ones((2, 2)), True)
        with no_grad():
            y = x @ x
        self.assertFalse(y.requires_grad); self.assertFalse(y._parents)
        loss = x.sum(); loss.backward()
        with self.assertRaises(RuntimeError):
            loss.backward()


class ModelTests(unittest.TestCase):
    def model(self):
        return Model(Config(context=16, width=8, heads=2, layers=1, seed=7))

    def test_causal_mask_prevents_future_token_leakage(self):
        model = self.model()
        with no_grad():
            before = model(np.array([[1, 2, 3, 4]])).data
            after = model(np.array([[1, 2, 99, 98]])).data
        np.testing.assert_allclose(before[:, :2], after[:, :2], atol=1e-7)

    def test_end_to_end_parameter_gradient_matches_finite_difference(self):
        model = self.model()
        for parameter in model.parameters.values():
            parameter.data = parameter.data.astype(np.float64)
        x, y = np.array([[1, 2, 3]]), np.array([[2, 3, 4]])
        cross_entropy(model(x), y).backward()
        for name, index in (("token", (1, 0)), ("block0.q", (0, 1)), ("block0.up", (1, 1)), ("final.gain", (0,)), ("output", (0, 2))):
            parameter = model.parameters[name]
            original = parameter.data[index]
            with no_grad():
                parameter.data[index] = original + 1e-5
                plus = float(cross_entropy(model(x), y).data)
                parameter.data[index] = original - 1e-5
                minus = float(cross_entropy(model(x), y).data)
                parameter.data[index] = original
            self.assertAlmostEqual(float(parameter.grad[index]), (plus - minus) / 2e-5, delta=2e-5, msg=name)

    def test_training_lowers_loss_and_updates_every_parameter(self):
        model = self.model()
        optimizer = AdamW(model.parameters, learning_rate=0.02, weight_decay=0)
        x, y = np.array([[1, 2, 3, 4]]), np.array([[2, 3, 4, 5]])
        initial = float(cross_entropy(model(x), y).data)
        for _ in range(40):
            optimizer.zero_grad(); loss = cross_entropy(model(x), y); loss.backward()
            self.assertTrue(all(parameter.grad is not None and np.isfinite(parameter.grad).all() for parameter in model.parameters.values()))
            optimizer.step()
        with no_grad():
            final = float(cross_entropy(model(x), y).data)
        self.assertLess(final, initial * 0.6)

    def test_checkpoint_and_optimizer_resume_are_exact(self):
        model = self.model(); optimizer = AdamW(model.parameters)
        x = np.array([[1, 2, 3]])
        cross_entropy(model(x), np.array([[2, 3, 4]])).backward(); optimizer.step()
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "model.npz"
            model.save(file, {"steps": 1}, optimizer)
            restored, header, states = Model.load(file)
            new_optimizer = AdamW(restored.parameters); new_optimizer.restore(header["optimizer"], states)
            with no_grad():
                np.testing.assert_array_equal(model(x).data, restored(x).data)
            self.assertEqual(new_optimizer.step_count, 1)
            for name in model.parameters:
                np.testing.assert_array_equal(optimizer.m[name], new_optimizer.m[name])
                np.testing.assert_array_equal(optimizer.v[name], new_optimizer.v[name])

    def test_generation_returns_text_and_learned_eos_score_is_preserved(self):
        model = self.model()
        chunks = list(model.generate([{"role": "user", "content": "hi"}], max_tokens=4, temperature=0))
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertLessEqual(len("".join(chunks).encode("utf-8")), 12)
        with self.assertRaises(ValueError):
            list(model.generate([{"role": "user", "content": "hi"}], temperature=-1))
        scores = np.full((1, 1, 261), -20, dtype=np.float32)
        scores[0, 0, ord('A')] = -2
        scores[0, 0, EOS] = -100
        with patch.object(Model, "next_logits", return_value=scores[0, 0]):
            self.assertEqual("".join(model.generate([{"role": "user", "content": "hi"}], max_tokens=3, temperature=0)), "AAA")
        scores[0, 0, EOS] = 2
        with patch.object(Model, "next_logits", return_value=scores[0, 0]):
            self.assertEqual("".join(model.generate([{"role": "user", "content": "hi"}], max_tokens=3, temperature=0)), "")

    def test_local_http_chat_stream_and_origin_checks(self):
        model = self.model()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model, {"config": {"context": 16}, "metadata": {"steps": 0}}, "test.npz"))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/api/info")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["name"], "Kelus Model B1")
            payload = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "temperature": 0})
            connection.request("POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse(); self.assertEqual(response.status, 200)
            events = [json.loads(line) for line in response.read().decode().splitlines()]
            self.assertTrue(events[-1]["done"])
            connection.request("POST", "/api/chat", body=payload, headers={"Content-Type": "application/json", "Origin": "https://example.com"})
            response = connection.getresponse(); self.assertEqual(response.status, 403); response.read()
            connection.request("GET", "/../README.md")
            response = connection.getresponse(); self.assertEqual(response.status, 404); response.read()
        finally:
            connection.close(); server.shutdown(); server.server_close(); thread.join(timeout=2)


class DataTests(unittest.TestCase):
    def test_training_command_resume_preserves_seed_and_protects_existing_weights(self):
        from kelus.train import main as train
        with tempfile.TemporaryDirectory() as root:
            data = Path(root) / 'examples.jsonl'
            data.write_text(''.join(json.dumps({'messages': [{'role': 'user', 'content': str(i)}, {'role': 'assistant', 'content': str(i + 1)}]}) + '\n' for i in range(5)))
            checkpoint = Path(root) / 'test.npz'
            args = ['--data', str(data), '--out', str(checkpoint), '--steps', '2', '--batch', '1', '--context', '16', '--width', '8', '--heads', '2', '--layers', '1', '--seed', '9', '--log-every', '1']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                train(args)
                initial_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                with self.assertRaises(SystemExit):
                    train(args)
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), initial_hash)
                train(['--data', str(data), '--out', str(checkpoint), '--resume', str(checkpoint), '--steps', '1', '--batch', '1'])
                _, header, _ = Model.load(checkpoint)
                self.assertEqual(header['metadata']['seed'], 9)
                self.assertEqual(header['metadata']['steps'], 3)
                self.assertIn('rng_state', header['metadata'])
                with self.assertRaises(SystemExit):
                    train(['--data', str(data), '--out', str(checkpoint), '--resume', str(checkpoint), '--steps', '1', '--seed', '42'])

    def test_utf8_and_literal_role_strings_round_trip(self):
        text = "hi 안녕 🌱 <assistant>"
        self.assertEqual(decode(encode(text)), text)
        self.assertTrue(all(token < 256 for token in encode(text)))

    def test_supervision_covers_only_assistant_tokens_and_eos(self):
        tokens, mask = conversation([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "ok"}])
        self.assertEqual([token for token, supervised in zip(tokens, mask) if supervised], [ord('o'), ord('k'), EOS])
        tokens, _ = conversation([{"role": "user", "content": "hi"}], generation=True)
        self.assertEqual(tokens[-1], ASSISTANT)
        with self.assertRaises(ValueError):
            conversation([{"role": "assistant", "content": "wrong"}])

    def test_dataset_split_and_batches_have_valid_targets(self):
        dataset = Dataset(Path(__file__).resolve().parent.parent / "data" / "starter.jsonl")
        train = {tuple(example[0]) for example in dataset.training}
        validation = {tuple(example[0]) for example in dataset.validation}
        self.assertFalse(train & validation)
        x, y = dataset.batch("train", 8, 16, np.random.default_rng(42))
        self.assertEqual(x.shape, (8, 16)); self.assertTrue(np.all((y >= 0).sum(axis=1) > 0))

    def test_server_rejects_malformed_and_oversized_generation_requests(self):
        valid = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(validate_chat(valid)[1], 240)
        for payload in ([], {"messages": []}, {**valid, "max_tokens": 9999}, {**valid, "temperature": float('nan')}, {**valid, "max_tokens": True}):
            with self.assertRaises(ValueError):
                validate_chat(payload)


if __name__ == "__main__":
    unittest.main()
