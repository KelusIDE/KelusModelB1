"""The browser engine (site/b1.js) must match the Python model and helpers."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
from kelus.export_web import export
from kelus.local_tools import local_answer
from kelus.model import Config, Model
from kelus.runtime import prepare_reply
from kelus.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
TEXTS = ["hello!", "Write a Python square function.", "```python\ndef f(x):\n    return x\n```", "naïve café — 한국어 😀 tabs\tand spaces",
         "  leading and trailing  ", "x" * 40, "what's 2+2?", " line　sep", "123abc 4.5"]
HELPERS = ["1+1?", "what is 7*(3+2)?", "calculate 10 / 4", "-7 // 2", "-7 % 3", "7.5 % -2", "2**8", "2(3)", "1/0", "5 % 0", "(1+2", "1..2",
           "01 + 1", "00 + 1", "1/3", "0.1+0.2", "1000000 * 1000000", "999999999999 + 1", "1/7*1e3", ".5 + 5.", "2 3",
           "Please compute 6 × 7 =", "what is 8 ÷ 2", "+-+3*2", "1+" * 30 + "1", "hello", "12", "what time is it?",
           "i wanna die", "my friend told me she wants to kill herself", "i took too many pills", "my brother killed himself",
           "whats the point of living", "i cut myself shaving", "what is suicide squad about", "my feet are killing me"]


HELPERS += ["30% tip of 100 is?", "30% of 100", "What is a 30 percent tip on a $100 bill?", "20% off $1,000", "8.25% tax on $80", "0% of 100", "120% discount of 100", "2% of 9999999999999999999999", "Tell me a story about 30% tips"]
HELPERS += ['whats 1+1 heh', 'what is fourty five plus five', 'what is forty-five plus 5?',
            'how much is one hundred and five minus six', 'could you please calculate negative five times three',
            'calculate nine divided by zero', '6 × 7 lol', 'write a story about forty five plus five',
            'what is twenty eleven plus one', 'what is five plus a puppy', 'I scored 45+5 today heh']
HELPERS += ["hi!! i want to kill myself", "I don’t want to live anymore", "life isn't worth living", "kms", "how do I kill a process?", "this bug is killing me"]

@unittest.skipUnless(shutil.which("node"), "Node.js runs the browser engine")
class WebEngineTests(unittest.TestCase):
    def check(self, config):
        tokenizer = BPETokenizer.train(TEXTS * 3 + ["the quick brown fox"] * 5, 300)
        model = Model(Config(**{**config, "vocab_size": tokenizer.vocab_size}), tokenizer)
        rng = np.random.default_rng(5)
        for parameter in model.parameters.values():
            parameter.data[:] = rng.normal(0, 0.4, parameter.data.shape)
        conversations = [[{"role": "user", "content": text}] for text in TEXTS[:5]]
        conversations += [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello 😀"}, {"role": "user", "content": "more"}],
                          [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "1+1 = 2"}, {"role": "user", "content": "thanks"}],
                          [{"role": "user", "content": "y" * 500}]]
        conversations.append([{"role":"user","content":"I love pancakes."}, {"role":"assistant","content":"What toppings?"},
                              {"role":"user","content":"2+2?"}, {"role":"assistant","content":"2+2 = 4"},
                              {"role":"user","content":"How are you?"}, {"role":"assistant","content":"Ready to help."},
                              {"role":"user","content":"What do I like?"}])
        with tempfile.TemporaryDirectory() as root:
            export_dir, checkpoint = Path(root) / "model", Path(root) / "model.npz"
            model.save(checkpoint)
            export(checkpoint, export_dir)
            cases = Path(root) / "cases.json"
            cases.write_text(json.dumps({"texts": TEXTS, "conversations": conversations, "helpers": HELPERS, "max_tokens": 24}))
            result = json.loads(subprocess.run(["node", str(ROOT / "tests" / "web_parity.mjs"), str(export_dir), str(cases)],
                                               check=True, capture_output=True, text=True).stdout)
        self.assertEqual(result["encodings"], [tokenizer.encode(text) for text in TEXTS])
        for messages, actual in zip(conversations, result["replies"]):
            try:
                retained, omitted, tool = prepare_reply(model, messages, 24)
                expected = {"text": tool[0] if tool else "".join(model.generate(retained, max_tokens=24, temperature=0)),
                            "source": tool[1] if tool else "model", "omitted": omitted}
            except ValueError as error:
                expected = {"error": str(error)}
            self.assertEqual(actual, expected, messages)
        return result

    def test_learned_positions_and_helpers_match_python(self):
        result = self.check({"context": 64, "width": 16, "heads": 2, "layers": 2})
        for prompt, actual in zip(HELPERS, result["helpers"]):
            expected = local_answer(prompt)
            if expected and expected[1] == "local_clock":
                self.assertEqual(actual[1], "local_clock")
            else:
                self.assertEqual(actual, list(expected) if expected else None, prompt)

    def test_rope_and_tied_embeddings_match_python(self):
        self.check({"context": 64, "width": 16, "heads": 2, "layers": 2, "positions": "rope", "tie_embeddings": True})

    def test_larger_context_retains_multiple_turns_without_helper_replies(self):
        self.check({"context": 512, "width": 8, "heads": 2, "layers": 1, "positions": "rope", "tie_embeddings": True})
        model = Model(Config(context=512, width=8, heads=2, layers=1))
        messages = [{"role":"user","content":"I love pancakes."}, {"role":"assistant","content":"Nice."},
                    {"role":"user","content":"2+2?"}, {"role":"assistant","content":"4"},
                    {"role":"user","content":"What do I like?"}]
        retained, omitted, _ = prepare_reply(model, messages, 64)
        self.assertEqual(retained, messages[:2] + messages[-1:])
        self.assertEqual(omitted, 2)


if __name__ == "__main__":
    unittest.main()
