import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from kelus.model import Config, Model
from kelus.tensor import Tensor, cross_entropy
from kelus.tokenizer import BPETokenizer, conversation, from_dict, EOS


def randomized(config):
    model = Model(config)
    rng = np.random.default_rng(0)
    for parameter in model.parameters.values():
        parameter.data[:] = rng.normal(0, 0.3, parameter.data.shape)
    return model


class ArchitectureTests(unittest.TestCase):
    configs = (Config(context=24, width=16, heads=2, layers=2),
               Config(context=24, width=16, heads=2, layers=2, positions="rope", tie_embeddings=True, dropout=0.2))

    def test_cached_decoding_matches_full_forward(self):
        for config in self.configs:
            model = randomized(config)
            ids = [int(i) for i in np.random.default_rng(1).integers(0, 261, 20)]
            full = model(np.array([ids])).data[0]
            cache = {}
            np.testing.assert_allclose(model.next_logits(ids[:6], cache), full[5], atol=1e-5)
            for end in range(7, 21):
                np.testing.assert_allclose(model.next_logits(ids[:end], cache), full[end - 1], atol=1e-5)
            np.testing.assert_allclose(model.next_logits(ids, {}), full[-1], atol=1e-5)
            with self.assertRaises(ValueError):
                model.next_logits(ids * 2, {})

    def test_options_change_parameters_and_dropout_only_applies_in_training(self):
        plain, options = Model(self.configs[0]), Model(self.configs[1])
        self.assertIn("position", plain.parameters); self.assertNotIn("position", options.parameters)
        self.assertIn("output", plain.parameters); self.assertNotIn("output", options.parameters)
        ids = np.array([[1, 2, 3, 4]])
        np.testing.assert_array_equal(options(ids).data, options(ids).data)
        dropped = options(ids, training=True, rng=np.random.default_rng(0)).data
        self.assertFalse(np.allclose(dropped, options(ids).data))
        with self.assertRaises(ValueError):
            Config(width=15, heads=3, positions="rope").validate()

    def test_rope_and_tied_gradients_match_numerical_estimates(self):
        model = randomized(Config(context=16, width=8, heads=2, layers=1, positions="rope", tie_embeddings=True))
        ids, targets = np.array([[3, 1, 4, 1]]), np.array([[1, 4, 1, 5]])
        for name in ("token", "block0.q", "block0.k"):
            parameter = model.parameters[name]
            parameter.data = parameter.data.astype(np.float64)
            for other in model.parameters.values():
                other.grad = None
            cross_entropy(model(ids), targets).backward()
            for index in ((1, 0), (4, 3), (0, 2)) if name == "token" else ((0, 1), (5, 6)):
                before = parameter.data[index]
                parameter.data[index] = before + 1e-5
                plus = float(cross_entropy(model(ids), targets).data)
                parameter.data[index] = before - 1e-5
                minus = float(cross_entropy(model(ids), targets).data)
                parameter.data[index] = before
                self.assertAlmostEqual(float(parameter.grad[index]), (plus - minus) / 2e-5, delta=2e-4)

    def test_checkpoints_without_new_config_fields_load_as_original_architecture(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "legacy.npz"
            model = randomized(self.configs[0])
            model.save(file)
            with np.load(file) as archive:
                values = {name: archive[name] for name in archive.files}
            header = json.loads(str(values["header"]))
            for key in ("positions", "tie_embeddings", "dropout"):
                header["config"].pop(key)
            values["header"] = np.array(json.dumps(header))
            np.savez(file, **values)
            restored, _, _ = Model.load(file)
            self.assertEqual(restored.config, self.configs[0])
            np.testing.assert_array_equal(restored(np.array([[5, 6]])).data, model(np.array([[5, 6]])).data)


class DataFormatTests(unittest.TestCase):
    def test_bpe_version_two_separates_opening_and_closing_code_fences(self):
        text = "```python\nx = 1\n```"
        legacy = BPETokenizer([], 1)
        self.assertEqual(legacy.encode(text)[:3], [96, 96, 96])
        tokenizer = BPETokenizer.train([text] * 20, 300)
        self.assertEqual(tokenizer.version, 2)
        opening, closing = tokenizer.encode("```python"), tokenizer.encode("```")
        self.assertEqual(len(opening), 1); self.assertNotEqual(opening, closing)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(from_dict(tokenizer.to_dict()).encode(text), tokenizer.encode(text))
        self.assertEqual(from_dict(legacy.to_dict()).version, 1)

    def test_unsupervised_reply_is_context_but_never_a_target(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "wrong", "supervise": False},
                    {"role": "user", "content": "no"}, {"role": "assistant", "content": "ok"}]
        tokens, supervised = conversation(messages)
        first_eos = tokens.index(EOS)
        self.assertFalse(any(supervised[:first_eos + 1]))
        self.assertEqual(sum(supervised), len("ok") + 1)


if __name__ == "__main__":
    unittest.main()
