"""Exercise handoff boundaries, streaming, cancellation and source preservation."""
import unittest
from types import SimpleNamespace
from kelus.cide import answer, context, handoff


class Expert:
    config = SimpleNamespace(context=96)
    tokenizer = SimpleNamespace(encode=lambda text: list(text))

    def __init__(self, text='```python\ndef add(a, b):\n    return a + b\n```'):
        self.text, self.calls = text, []

    def generate(self, messages, **options):
        self.calls.append(messages)
        yield self.text


class CideTests(unittest.TestCase):
    messages = [{'role': 'user', 'content': 'Fix the addition function.'}]

    def test_only_an_exact_model_call_delegates_and_preserves_request(self):
        expert = Expert()
        events = list(handoff(iter(['C', 'ID', 'E', '\n']), self.messages, 24, 0, load=lambda: expert))
        self.assertEqual(events[0]['source'], 'cide')
        self.assertEqual(expert.calls, [self.messages])
        self.assertIn('return a + b', events[1]['text'])

    def test_ordinary_text_search_and_quoted_calls_are_unchanged(self):
        def no_load():
            self.fail('An ordinary reply must not load Cide')
        for output in ['CIDER', 'CIDE is a coding model.', 'C', 'SEARCH: python documentation', '```\nCIDE\n```', 'hello']:
            events = list(handoff(iter(output), self.messages, 24, 0, load=no_load))
            self.assertEqual(''.join(e['text'] for e in events), output)

    def test_specialist_cannot_trigger_recursive_tools(self):
        expert = Expert('CIDE')
        events = list(handoff(['CIDE'], self.messages, 24, 0, load=lambda: expert))
        self.assertEqual(len(expert.calls), 1)
        self.assertEqual(events[-1], {'type': 'token', 'text': 'CIDE'})

    def test_cancellation_prevents_loading_or_tokens(self):
        def no_load():
            self.fail('Cancelled requests must not load Cide')
        self.assertEqual(list(handoff(['CIDE'], self.messages, 24, 0, load=no_load, cancelled=lambda: True)), [])
        self.assertEqual(list(answer(self.messages, load=no_load, cancelled=lambda: True)), [])

    def test_long_code_is_rejected_not_cut(self):
        with self.assertRaisesRegex(ValueError, 'too long'):
            context(Expert(), [{'role': 'user', 'content': 'x' * 100}], 24)
        history = [{'role': 'user', 'content': 'x' * 70}, {'role': 'assistant', 'content': 'Earlier reply.'}] + self.messages
        self.assertEqual(context(Expert(), history, 24), self.messages)

    def test_missing_expert_is_reported_not_answered_by_kani(self):
        def missing():
            raise FileNotFoundError('Cide is not trained/exported yet')
        with self.assertRaisesRegex(FileNotFoundError, 'not trained'):
            list(handoff(['CIDE'], self.messages, 24, 0, load=missing))

    def test_handoff_uses_coding_budget_and_honors_explicit_override(self):
        budgets = []
        class LongExpert(Expert):
            def generate(self, messages, **options):
                budgets.append(options['max_tokens'])
                yield 'x' * min(1200, options['max_tokens'])
        full = list(handoff(['CIDE'], self.messages, 240, 0, load=LongExpert))
        short = list(handoff(['CIDE'], self.messages, 240, 0, load=LongExpert, coding_max_tokens=64))
        self.assertEqual(len(full[-1]['text']), 1200)
        self.assertEqual(len(short[-1]['text']), 64)
        self.assertEqual(budgets, [4096, 64])


if __name__ == '__main__':
    unittest.main()
