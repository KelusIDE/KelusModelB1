"""Does Kani stay on the subject? Short follow-ups, a change of subject, and things that happened to the user.

    python3 -m kelus.ontopic_eval --checkpoint training-runs/kani-otter-4-5-eval/s4000.npz --out reports/ontopic.json

Handwritten after users said Kani "really gets off track" (2026-10-29): "I want to learn guitar" then "how long will it
take?" got "SEARCH: Guitar", and exam stress, a friend's anger and cancelled plans all got one "feeling down" reply.
These prompts and subjects are held out of instruct-v14 (data/make_instruct_v14.py leaves them out).
  - FOLLOW_UPS: a subject, Kani's reply, then a question that only makes sense with the subject. Passes when the reply
    (a lookup or an answer) names the subject.
  - SWITCHES: the same openings, then a new question that stands alone. Passes when the reply is about the new
    question and does not drag the old subject in.
  - FEELINGS: something that happened to the user. Passes when the reply mentions what happened, without repeating
    the message back or turning it into a lookup; bad news must not be congratulated, and good news must not get
    "sorry". The words leave out ones that Kani's old catch-all reply happens to use ("friend").
Word checks are a floor, not a grade: replies are kept for reading.
"""
import argparse
import json
from pathlib import Path
import re
import time

from .model import Model


FOLLOW_UPS = [
    ('I want to learn piano', 'Piano is a lovely instrument to learn. Do you have a keyboard at home?', r'piano',
     ['how long will it take?', 'how much do lessons cost?', 'where can I take lessons near me?']),
    ("I'm thinking about learning French", 'Bonne idée! Is it for travel, school or just for fun?', r'french',
     ['how long does it take?', "what's the best way to start?"]),
    ("We're going to Rome in May", 'Rome in May sounds wonderful. Is it your first visit?', r'rome',
     ["what's the weather like then?", 'what should we eat there?', 'where should we stay?']),
    ('I might adopt a rabbit', 'Rabbits can be great pets. Have you had one before?', r'rabbit|bunn',
     ['what do they eat?', 'how long do they live?']),
    ('I want to get into skateboarding', "Fun! Do you have a board yet, or are you starting fresh?", r'skat',
     ['what do I need?', 'is it hard to learn?']),
    ("I'm buying my first electric car", 'Exciting! Do you know which models you are looking at?', r'electric|\bevs?\b',
     ['how much should I spend?', 'what should I look for?']),
    ('my daughter wants to learn to surf', 'Surfing is a great thing to learn young. How old is she?', r'surf',
     ['what age can kids start?', 'where can she learn around here?']),
    ("I'm setting up a fish tank", 'A fish tank is a fun project. Freshwater or saltwater?', r'fish|aquarium|tank',
     ['what do I need?', 'how often should I clean it?']),
]


SWITCHES = [
    ("what's the capital of Canada?", r'canada|ottawa'),
    ('who invented the light bulb?', r'light ?bulb|edison'),
    ('how do bees make honey?', r'bee|honey'),
]


FEELINGS = [
    ("I'm stressed about my exams next week", r'exam|test|stud', 'bad'),
    ("my friend is mad at me and I don't know why", r'upset|mad|angry|argu|fight', 'bad'),
    ("I'm so nervous about my driving test tomorrow", r'driv|test', 'bad'),
    ('my best friend is moving away next month', r'mov|miss|away|distance|visit', 'bad'),
    ('I got rejected from my dream college', r'college|school|reject|application|admission', 'bad'),
    ('my grandma is in the hospital', r'grand|hospital', 'bad'),
    ('my coworker took credit for my work', r'credit|cowork|colleague|work|boss|manager', 'bad'),
    ('my phone broke and I lost all my photos', r'phone|photo|picture', 'bad'),
    ('my sister and I keep fighting', r'sister|fight|argu', 'bad'),
    ('I failed my driving test again', r'driv|test', 'bad'),
    ("I just got my driver's license!", r'licen|driv', 'good'),
    ('I finally finished my first marathon', r'marathon|run|race', 'good'),
    ('my team won the championship!', r'team|won|win|champion', 'good'),
    ('I got into the college I wanted!', r'college|school|accept', 'good'),
]

FEELING_FOLLOW_UPS = [
    ("my friend is mad at me and I don't know why",
     "That's hard, especially when you don't know the reason. Did anything happen before they got upset?",
     'she stopped replying after I cancelled plans', r'cancel|plan|repl|messag|text|sorry|apolog'),
    ("I'm nervous about my piano recital", 'Recital nerves are really common. When is it?',
     "it's on saturday and I keep messing up the same part", r'recital|part|practi|piano|play'),
]


def user(text):
    return {'role': 'user', 'content': text}


def assistant(text):
    return {'role': 'assistant', 'content': text}


def prompts():
    """The messages to keep out of training data: openings, new questions and feelings. Follow-up wordings such as
    "how long will it take?" are shared with training on purpose; the test is whether a new subject carries over."""
    texts = [o for o, *_ in FOLLOW_UPS] + [q for q, _ in SWITCHES] + [m for m, *_ in FEELINGS]
    texts += [t for first, _, follow, _ in FEELING_FOLLOW_UPS for t in (first, follow)]
    return texts


def answered(message, reply, words, wrong):
    """The reply is about what happened, in Kani's own words, and is not a lookup or the wrong mood."""
    echo = ' '.join(re.findall(r'[a-z]+', message.lower())[:6])
    return (bool(re.search(words, reply, re.I)) and not re.search(wrong, reply, re.I)
            and not reply.startswith(('SEARCH:', 'PLACES:')) and not ' '.join(re.findall(r'[a-z]+', reply.lower())).startswith(echo))


def run(model, max_tokens=120):
    def reply(messages):
        return ''.join(model.generate(messages, max_tokens=max_tokens, temperature=0)).strip()

    results = []
    for opening, answer, subject, follow_ups in FOLLOW_UPS:
        for follow in follow_ups:
            text = reply([user(opening), assistant(answer), user(follow)])
            results.append({'kind': 'follow_up', 'opening': opening, 'prompt': follow, 'reply': text,
                            'passed': bool(re.search(subject, text, re.I))})
        for question, words in SWITCHES:
            text = reply([user(opening), assistant(answer), user(question)])
            results.append({'kind': 'switch', 'opening': opening, 'prompt': question, 'reply': text,
                            'passed': bool(re.search(words, text, re.I)) and not re.search(subject, text, re.I)})
    for message, words, mood in FEELINGS:
        text = reply([user(message)])
        wrong = r'congrat' if mood == 'bad' else r'\bsorry\b'
        results.append({'kind': 'feeling', 'prompt': message, 'reply': text, 'passed': answered(message, text, words, wrong)})
    for first, answer, follow, words in FEELING_FOLLOW_UPS:
        text = reply([user(first), assistant(answer), user(follow)])
        results.append({'kind': 'feeling', 'opening': first, 'prompt': follow, 'reply': text,
                        'passed': answered(follow, text, words, r'congrat')})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    model, header, _ = Model.load(args.checkpoint)
    started = time.monotonic()
    results = run(model)
    score = lambda kind: f"{sum(r['passed'] for r in results if r['kind'] == kind)}/{sum(r['kind'] == kind for r in results)}"
    report = {'checkpoint': args.checkpoint, 'steps': header.get('metadata', {}).get('steps'),
              'follow_ups': score('follow_up'), 'switches': score('switch'), 'feelings': score('feeling'),
              'seconds': round(time.monotonic() - started, 1), 'results': results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('results', 'checkpoint')}))


if __name__ == '__main__':
    main()
