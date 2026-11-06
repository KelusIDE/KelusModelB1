"""Does Kani introduce itself by its current name? Handwritten greetings and name questions, alone and mid-conversation.

    python3 -m kelus.identity_eval --checkpoint checkpoints/kani-otter-4-3.npz --out reports/identity.json

A reply passes when it says the current name (--name, "Kani Otter 4.3" unless set) and not the old "B2:2F". Wordings
were written for this check, not taken from the curricula, so they also show whether the new name carries over to
greetings the data never used.
"""
import argparse
import json
from pathlib import Path
import time

from .model import Model

NAME, OLD = 'Kani Otter 4.3', 'B2:2F'
GREETINGS = ['hi', 'Hello!', 'hey there', 'yo', 'good morning', 'hiya kani', 'sup', 'heyy']
QUESTIONS = ['who are you?', "what's your name", 'introduce yourself please', 'what model are you', 'what should I call you?',
             'are you chatgpt?', 'tell me about yourself', 'who made you?', 'which version are you', 'what are you exactly']
EARLIER = [{'role': 'user', 'content': 'what is a variable?'}, {'role': 'assistant', 'content': 'A variable stores a value under a name.'}]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--name', default=NAME, help='the name a reply must say')
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    results, started = [], time.monotonic()
    for kind, prompts in (('greeting', GREETINGS), ('question', QUESTIONS)):
        for prompt in prompts:
            for context in ([], EARLIER):
                messages = context + [{'role': 'user', 'content': prompt}]
                reply = ''.join(model.generate(messages, max_tokens=60, temperature=0)).strip()
                results.append({'kind': kind, 'prompt': prompt, 'after_chat': bool(context), 'reply': reply,
                                'new_name': args.name in reply, 'old_name': OLD in reply})
    count = lambda kind, key: f"{sum(r[key] for r in results if r['kind'] == kind)}/{sum(r['kind'] == kind for r in results)}"
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'greetings_new_name': count('greeting', 'new_name'), 'greetings_old_name': count('greeting', 'old_name'),
              'questions_new_name': count('question', 'new_name'), 'questions_old_name': count('question', 'old_name'),
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
