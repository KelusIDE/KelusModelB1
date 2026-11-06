"""Standard zero-shot benchmarks: HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande and LAMBADA (OpenAI's version),
scored the way EleutherAI's lm-evaluation-harness scores them (same prompts, log-likelihood of each answer, acc and
length-normalised acc_norm), so Kani can be set beside published small models (GPT-2, Pythia, OPT, SmolLM).

The test sets are downloaded files in --data (hellaswag_val.jsonl, arc_easy_test.jsonl, arc_challenge_test.jsonl,
piqa_valid.jsonl + piqa_valid_labels.lst, winogrande_val.jsonl, lambada_test.jsonl). They are only read here: never
trained on, and their wordings stay out of every data/ file.

    .venv/bin/python -m kelus.standard_eval --checkpoint checkpoints/kani-otter-4-9-release.npz --data DIR --out result.json
"""
import argparse
import json
import math
from pathlib import Path
import re
import time

import numpy as np

TASKS = ['hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'winogrande', 'lambada']

# The goal (GPT3_GOAL.md): GPT-3's zero-shot scores at each size, from Brown et al. 2020, Table H.1 (all trained on 300B
# tokens). OpenAI scored with its own prompts (ARC normalised by the answer's unconditional probability), so a rung is
# a guide within a point or two, not an exact match. Kani's number for each task is the metric named in METRIC.
GPT3 = {
    'GPT-3 Small (125M)': {'hellaswag': 33.7, 'lambada': 42.7, 'piqa': 64.6, 'winogrande': 52.0, 'arc_easy': 43.6, 'arc_challenge': 26.6},
    'GPT-3 Medium (350M)': {'hellaswag': 43.6, 'lambada': 54.3, 'piqa': 70.2, 'winogrande': 52.1, 'arc_easy': 46.5, 'arc_challenge': 29.5},
    'GPT-3 Large (760M)': {'hellaswag': 51.0, 'lambada': 60.4, 'piqa': 72.9, 'winogrande': 57.4, 'arc_easy': 53.0, 'arc_challenge': 31.8},
    'GPT-3 XL (1.3B)': {'hellaswag': 54.7, 'lambada': 63.6, 'piqa': 75.1, 'winogrande': 58.7, 'arc_easy': 53.8, 'arc_challenge': 35.5},
    'GPT-3 2.7B': {'hellaswag': 62.8, 'lambada': 67.1, 'piqa': 75.6, 'winogrande': 62.3, 'arc_easy': 58.2, 'arc_challenge': 38.0},
    'GPT-3 6.7B': {'hellaswag': 67.4, 'lambada': 70.3, 'piqa': 78.0, 'winogrande': 64.5, 'arc_easy': 60.2, 'arc_challenge': 41.4},
    'GPT-3 13B': {'hellaswag': 70.9, 'lambada': 72.5, 'piqa': 78.5, 'winogrande': 67.9, 'arc_easy': 63.8, 'arc_challenge': 43.7},
    'GPT-3 175B': {'hellaswag': 78.9, 'lambada': 76.2, 'piqa': 81.0, 'winogrande': 70.2, 'arc_easy': 68.8, 'arc_challenge': 51.4},
}
METRIC = {'hellaswag': 'acc_norm', 'arc_challenge': 'acc_norm', 'arc_easy': 'acc', 'piqa': 'acc', 'winogrande': 'acc', 'lambada': 'acc'}


def ladder(results):
    """Kani's six scores (percent), their average, and each GPT-3 size's average and the tasks Kani matches there.
    The highest rung whose average Kani reaches is the one it has climbed to."""
    scores = {task: 100 * results[task][METRIC[task]] for task in TASKS if task in results}
    if len(scores) < len(TASKS):
        return None
    average = sum(scores.values()) / len(scores)
    rungs = [{'model': name, 'average': round(sum(t.values()) / len(t), 2), 'tasks_matched': sum(scores[k] >= v for k, v in t.items())}
             for name, t in GPT3.items()]
    reached = [r['model'] for r in rungs if average >= r['average']]
    next_rung = next((r for r in rungs if average < r['average']), None)
    return {'scores': {k: round(v, 2) for k, v in scores.items()}, 'average': round(average, 2), 'reached': reached[-1] if reached else None,
            'next': next_rung and {**next_rung, 'points_to_go': round(next_rung['average'] - average, 2)}, 'rungs': rungs}


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _hellaswag_text(text):
    text = text.strip().replace(' [title]', '. ')
    return re.sub(r'\[.*?\]', '', text).replace('  ', ' ')


# Each document: a list of (context, continuation) requests and the index of the right one.
def documents(task, data):
    data = Path(data)
    if task == 'hellaswag':
        for row in _lines(data / 'hellaswag_val.jsonl'):
            context = _hellaswag_text(row['activity_label'] + ': ' + row['ctx_a'] + ' ' + row['ctx_b'].capitalize())
            yield [(context, ' ' + _hellaswag_text(ending)) for ending in row['endings']], int(row['label'])
    elif task in ('arc_easy', 'arc_challenge'):
        for row in _lines(data / f'{task}_test.jsonl'):
            labels = row['choices']['label']
            if row['answerKey'] not in labels:
                continue
            context = 'Question: ' + row['question'] + '\nAnswer:'
            yield [(context, ' ' + text) for text in row['choices']['text']], labels.index(row['answerKey'])
    elif task == 'piqa':
        labels = (data / 'piqa_valid_labels.lst').read_text().split()
        for row, label in zip(_lines(data / 'piqa_valid.jsonl'), labels):
            context = 'Question: ' + row['goal'] + '\nAnswer:'
            yield [(context, ' ' + row['sol1']), (context, ' ' + row['sol2'])], int(label)
    elif task == 'winogrande':
        # Partial scoring: the words after the blank, given the sentence up to it with each option filled in.
        for row in _lines(data / 'winogrande_val.jsonl'):
            sentence = row['sentence']
            blank = sentence.index('_')
            after = ' ' + sentence[blank + 1:].strip()
            yield [(sentence[:blank] + row['option1'], after), (sentence[:blank] + row['option2'], after)], int(row['answer']) - 1
    elif task == 'lambada':
        for row in _lines(data / 'lambada_test.jsonl'):
            context, word = row['text'].rsplit(' ', 1)
            yield [(context, ' ' + word)], 0
    else:
        raise ValueError(task)


class Scorer:
    """Log-likelihoods of continuations for any model given encode(text) -> ids and logits(ids [B, T]) -> [B, T, V]."""

    def __init__(self, encode, logits, context, batch_tokens=16384):
        self.encode, self.logits, self.context, self.batch_tokens = encode, logits, context, batch_tokens
        self.cache = {}

    def pair(self, context, continuation):
        # lm-evaluation-harness's split: trailing spaces move to the continuation; the continuation is what encoding
        # the whole adds after the context's own tokens.
        spaces = len(context) - len(context.rstrip())
        if spaces:
            continuation, context = context[-spaces:] + continuation, context[:-spaces]
        key = (context, continuation)
        if key not in self.cache:
            whole, start = self.encode(context + continuation), len(self.encode(context))
            self.cache[key] = (whole, max(1, start))
        return self.cache[key]

    def score(self, requests):
        """[(sum log-probability, every continuation token was the most likely)] for each request."""
        import mlx.core as mx
        items = []
        for index, (context, continuation) in enumerate(requests):
            whole, start = self.pair(context, continuation)
            cut = max(0, len(whole) - (self.context + 1))
            items.append((index, whole[cut:], start - cut))
        results = [None] * len(items)
        items.sort(key=lambda item: -len(item[1]))
        position = 0
        while position < len(items):
            longest = len(items[position][1]) - 1
            size = max(1, min(64, self.batch_tokens // max(1, longest)))
            batch = items[position:position + size]
            position += size
            ids = np.zeros((len(batch), longest), dtype=np.int32)
            for row, (_, tokens, _) in enumerate(batch):
                ids[row, :len(tokens) - 1] = tokens[:-1]
            logits = self.logits(mx.array(ids)).astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            best = mx.argmax(logits, axis=-1)
            mx.eval(logprobs, best)
            logprobs, best = np.array(logprobs), np.array(best)
            for row, (index, tokens, start) in enumerate(batch):
                targets = np.array(tokens[start:]), np.arange(start - 1, len(tokens) - 1)
                results[index] = (float(logprobs[row, targets[1], targets[0]].sum()), bool((best[row, targets[1]] == targets[0]).all()))
        return results


def evaluate(scorer, data, tasks=TASKS, limit=None, log=print):
    report = {}
    for task in tasks:
        started = time.time()
        docs = list(documents(task, data))[:limit]
        flat = [request for requests, _ in docs for request in requests]
        scores = scorer.score(flat)
        correct = normalised = 0
        at = 0
        total_logprob = 0.0
        for requests, gold in docs:
            mine = scores[at:at + len(requests)]
            at += len(requests)
            if task == 'lambada':
                correct += mine[0][1]
                total_logprob += mine[0][0]
                continue
            sums = [s for s, _ in mine]
            lengths = [len(c.encode('utf-8')) for _, c in requests]
            correct += int(np.argmax(sums)) == gold
            normalised += int(np.argmax([s / n for s, n in zip(sums, lengths)])) == gold
        result = {'acc': correct / len(docs), 'documents': len(docs)}
        if task == 'lambada':
            result['perplexity'] = math.exp(-total_logprob / len(docs))
        elif task in ('hellaswag', 'arc_easy', 'arc_challenge', 'piqa'):
            result['acc_norm'] = normalised / len(docs)
        report[task] = result
        log(f"{task}: " + ', '.join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in result.items())
            + f" ({time.time() - started:.0f}s)")
    return report


def kani_scorer(checkpoint):
    from .gpu import forward_fast, parameters, rope_tables
    from .model import Model
    model, header, _ = Model.load(checkpoint)
    params, (cos, sin) = parameters(model), rope_tables(model)
    try:
        from .rust_tokenizer import accelerated
        encode = accelerated(model.tokenizer).encode
    except Exception:
        encode = model.tokenizer.encode
    return Scorer(encode, lambda ids: forward_fast(params, ids, model.config, cos, sin), model.config.context), header


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--limit', type=int, default=None, help='Score only the first N documents of each task.')
    parser.add_argument('--out')
    args = parser.parse_args()
    from .gpu import limit_memory
    limit_memory(12)
    scorer, header = kani_scorer(args.checkpoint)
    report = {'model': header.get('name') or Path(args.checkpoint).stem, 'checkpoint': args.checkpoint,
              'parameters': header.get('metadata', {}).get('parameters'),
              'results': evaluate(scorer, args.data, args.tasks.split(','), args.limit)}
    report['gpt3'] = ladder(report['results'])
    if report['gpt3']:
        g = report['gpt3']
        print(f"average {g['average']:.1f} · reached {g['reached'] or 'no GPT-3 size yet'}"
              + (f" · next {g['next']['model']} (average {g['next']['average']:.1f}, {g['next']['points_to_go']:.1f} points to go)" if g['next'] else ''))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
