"""Build Cide's coding curriculum and Kani's learned handoff curriculum.

Reuses our v7 data and adds small, inspectable repair examples. No downloaded
weights or generated answers from an external model. Splits are preserved;
repair validation uses different function names. Replay validation is NOT an
unseen benchmark, because the parent Kani checkpoint already practiced it.
"""
import copy
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def example(question, answer, split, family):
    return {'split': split, 'family': family, 'messages': [
        {'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}


def is_code(row):
    return '```' in row['messages'][-1]['content']


def build(seed=2041):
    rng = random.Random(seed)
    original = [json.loads(line) for line in (ROOT / 'data/instruct-v7.jsonl').read_text().splitlines()]
    expert, router = [], []
    for split, count, negatives in [('train', 6000, 6000), ('validation', 120, 120)]:
        pool = [r for r in original if r['split'] == split]
        code = rng.sample([r for r in pool if is_code(r)], count)
        expert.extend(copy.deepcopy(code))


        for row in code:
            row = copy.deepcopy(row)
            row['messages'][-1]['content'] = 'CIDE'
            row.pop('accept', None)
            row['family'] = 'cide_handoff'
            router.append(row)
        other = [r for r in pool if not is_code(r)]

        for family in ['decide', 'reading', 'chat_pairs', 'practiced', 'feelings']:
            candidates = [r for r in other if r.get('family') == family]
            router.extend(copy.deepcopy(rng.sample(candidates, min(len(candidates), negatives // 5))))

    for split, names in [('train', ['add', 'total', 'sum_values', 'combine', 'calculate', 'sum_pair', 'add_numbers', 'get_sum']),
                         ('validation', ['heldout_sum', 'sum_two_unseen'])]:
        for name in names:
            for a, b in [('a', 'b'), ('x', 'y'), ('left', 'right'), ('first', 'second')]:
                for language in ['Python', 'JavaScript']:
                    code = (f'def {name}({a}, {b}):\n    return {a} + {b}' if language == 'Python'
                            else f'function {name}({a}, {b}) {{\n  return {a} + {b};\n}}')
                    answer = f'```{language.lower()}\n{code}\n```'
                    wrong = code.replace(' + ', ' - ')
                    questions = [f'Fix this {language} function. It should add both arguments. Return only the function.\n{wrong}',
                                 f'Write a {language} function {name}({a}, {b}) that returns {a} + {b}. Return only the function.',
                                 f'This code subtracts instead of adding. Fix it:\n```{language.lower()}\n{wrong}\n```',
                                 f'Write a {language} function {name}({a}, {b}) that adds both arguments. Return only the function.\nCurrent {language} function:\n{wrong}']
                    for question in questions:
                        expert.append(example(question, answer, split, 'repair'))
                        router.append(example(question, 'CIDE', split, 'cide_handoff'))
    for split, questions in [('train', ['who are you', 'what is your name', 'tell me about yourself', 'what can you do', 'are you Cide']),
                             ('validation', ['describe your coding abilities'])]:
        for question in questions:
            expert.append(example(question, 'I am Cide, Kani\'s experimental coding specialist. I write small Python and JavaScript functions and can make mistakes.', split, 'identity'))
    for rows in (expert, router):
        rng.shuffle(rows)
    return expert, router


if __name__ == '__main__':
    for name, rows in zip(['cide-v1', 'kani-cide-v1'], build()):
        destination = ROOT / 'data' / (name + '.jsonl')
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        print(json.dumps({'path': str(destination), 'train': sum(r['split'] == 'train' for r in rows),
                          'validation': sum(r['split'] == 'validation' for r in rows)}))
