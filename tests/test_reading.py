import json
from pathlib import Path
import shutil
import subprocess
import unittest

from kelus.reading import NO_ANSWER, prompt, select
from kelus.reading_eval import score


class ReadingTests(unittest.TestCase):
    def test_prompt_numbers_results_in_citation_order(self):
        text = prompt([('Vega', 'Vega is a star.'), ('Lyra', 'Lyra is a constellation.')], 'What is Vega?')
        self.assertEqual(text, 'Search results:\n[1] Vega: Vega is a star.\n[2] Lyra: Lyra is a constellation.\n\nQuestion: What is Vega?')

    def test_select_keeps_the_lead_and_the_sentences_about_the_question(self):
        text = ('Cusco is a city in Peru. It sits high in the Andes. In 2017, it had a population of 428,450. '
                'The city was the capital of the Inca Empire. Its elevation is around 3,400 m.')
        (title, kept), = select([('Cusco', text)], 'What is the population of Cusco?', budget=16)
        self.assertEqual(title, 'Cusco')
        self.assertEqual(kept, 'Cusco is a city in Peru. In 2017, it had a population of 428,450.')

    def test_scoring_requires_the_right_citation_and_text_from_the_passages(self):
        row = {'kind': 'answer', 'accept': ['Vega is a star. [1]'], 'messages': [{'role': 'user', 'content': prompt([('Vega', 'Vega is a star.')], 'What is Vega?')}]}
        self.assertEqual(score(row, 'Vega is a star. [1]'), {'exact': True, 'cited': True, 'refused': False, 'grounded': True})
        self.assertFalse(score(row, 'Vega is a planet. [1]')['grounded'])
        self.assertFalse(score(row, 'Vega is a star. [2]')['cited'])
        self.assertTrue(score({**row, 'kind': 'no_answer', 'accept': [NO_ANSWER]}, NO_ANSWER)['no_answer'])


    def test_a_run_on_sentence_is_cut_to_the_budget(self):
        (_, kept), = select([('List', ' '.join(f'word{i}' for i in range(500)))], 'word3?', budget=90)
        self.assertEqual(len(kept.split()), 90)

    @unittest.skipUnless(shutil.which('node'), 'Node.js runs site/reading.js')
    def test_the_apps_select_the_same_sentences_as_training(self):
        text = ('Cusco is a city in Peru. It sits high in the Andes. In 2017, it had a population of 428,450. '
                'The city was the capital of the Inca Empire. Its elevation is around 3,400 m.')
        run_on = ' '.join(f'item{i} einstein list entry' for i in range(200))
        cases = [{'results': [['Cusco', text], ['Vega', 'Vega is a star. It is bright.'], ['List', run_on]], 'question': q, 'budget': b}
                 for q in ('What is the population of Cusco?', 'what was the capital of the Inca Empire', 'Vega?') for b in (12, 30, 90)]
        script = ('import("./site/reading.js").then(m => { const cases = JSON.parse(require("fs").readFileSync(0, "utf8")); '
                  'process.stdout.write(JSON.stringify(cases.map(c => { const s = m.selectSentences(c.results, c.question, c.budget); '
                  'return [s, m.readingPrompt(s, c.question)]; }))); });')
        out = json.loads(subprocess.run(['node', '-e', script], input=json.dumps(cases), capture_output=True, text=True, check=True,
                                        cwd=Path(__file__).resolve().parent.parent).stdout)
        for case, (chosen, text) in zip(cases, out):
            expected = select([tuple(r) for r in case['results']], case['question'], case['budget'])
            self.assertEqual([tuple(r) for r in chosen], expected, case['question'])
            self.assertEqual(text, prompt(expected, case['question']))


if __name__ == '__main__':
    unittest.main()
