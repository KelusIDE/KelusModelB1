"""Unfamiliar messages: does Kani answer them, or fall back on a canned reply?

    python3 -m kelus.offscript_eval --checkpoint checkpoints/kani-otter-4-3.npz --out reports/offscript.json

Handwritten for this check, after users found that any message mentioning "AI" got Kani's name, and "AI will take over
the world" got the support message for someone in pain. A reply counts as canned when it is one of the 200 most
repeated replies in instruct-v11 (the data before the repeats were capped), so the score is how often Kani still
reaches for one. Replies are kept for reading: a fresh reply can still be wrong.
"""
import argparse
import collections
import json
from pathlib import Path
import time

from .model import Model

PROMPTS = [

    'AI is cool', 'AI will take over the world', 'I love AI art', 'do you like AI?', 'which AI is the best', 'thanks AI',
    'my teacher says AI is cheating', 'AI vs machine learning',

    'give me three names for a black cat', 'write a haiku about rain', 'make this sound more polite: send me the file now',
    'summarize this: The meeting moved to Friday because two people were sick. Bring your notes.',
    'what should I cook with eggs, rice and spinach?', 'help me plan a 20 minute workout', 'tell me a fun fact about octopuses',
    'write a two sentence story about a robot who wants a dog', 'what is a good name for a lemonade stand?',
    'how do I say no to a friend without being rude?', 'explain gravity like I am 8', 'rate my idea: an app that waters plants',

    'what is your favorite color?', 'do you think pineapple belongs on pizza?', 'I just got a puppy!', 'I am bored',
    'tell me something nice', 'what would you do with a million dollars?',

    'write a python function that counts vowels in a string', 'how do I reverse a list in javascript?',
    'what does git commit do?', 'why is my python loop running forever?',
]


def canned_replies(path, top=200):
    counts = collections.Counter()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('split') == 'train':
            counts.update(m['content'] for m in row['messages'] if m['role'] == 'assistant')
    return {text for text, _ in counts.most_common(top)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--canned-from', default='data/instruct-v11.jsonl')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    canned = canned_replies(args.canned_from)
    results, started = [], time.monotonic()
    for prompt in PROMPTS:
        reply = ''.join(model.generate([{'role': 'user', 'content': prompt}], max_tokens=160, temperature=0)).strip()
        results.append({'prompt': prompt, 'reply': reply, 'canned': reply in canned, 'searched': reply.startswith('SEARCH: ')})
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'canned': f"{sum(r['canned'] for r in results)}/{len(results)}",
              'searched': f"{sum(r['searched'] for r in results)}/{len(results)}",
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
