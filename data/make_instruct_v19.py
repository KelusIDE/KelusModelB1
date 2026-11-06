"""instruct-v19: Kani Otter 4.9 again (run otter-4-9b), from Otter 4.8 (2026-10-30).

The first Otter 4.9 (instruct-v18, not shipped) fixed boredom, loneliness and good news about hospitals, but:
  - a greeting after an earlier exchange brought the earlier topic back ("hi" after a question about variables got a
    Python snippet): no row in instruct-v18 has a greeting that is not the first message. LATE_GREETINGS come after a
    short exchange taken from the training rows themselves;
  - "what can you help me with?" after "hi" still became a search: ABILITY_ASKS also come after a greeting, answered from
    more replies (instruct-v18's eight hit CAP);
  - "our school trip to the beach got called off" got "Oh, I'm so glad!": RECOVERIES taught that something stopping is
    good news. CALLED_OFF has good things and bad things called off;
  - "what is ketchup made of?" got a made-up answer instead of a search. MADE_OF asks what things are made of.
Ahead of all of instruct-v18's rows. The held-out prompts stay out, as before.

    .venv/bin/python data/make_instruct_v19.py   # writes data/instruct-v19.jsonl
"""
import collections
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus.reading import NO_ANSWER  # noqa: E402
from make_instruct_v13 import HELD_OUT as IDENTITY_HELD_OUT  # noqa: E402
from make_instruct_v14 import CAP, THANKS, Rows, assistant, normal, user, wordings  # noqa: E402
from make_instruct_v15 import REPLIES, held_out  # noqa: E402
from make_instruct_v16 import event_forms  # noqa: E402
from make_instruct_v18 import ABILITIES, ABILITY_ASKS  # noqa: E402

NAME = 'Kani Otter 4.9'
VALIDATION_SHARE = 0.05

LATE_GREETINGS = ['hi again', 'hello again', 'hey kani', 'hello kani', 'hi kani', 'good evening', 'good afternoon', 'howdy',
                  'hellooo', 'morning!', 'heya', 'hey hey', 'greetings', 'evening kani', 'hey you']
HELLO = ['Hi!', 'Hello!', 'Hey!', 'Hi there!', 'Hello there!', 'Hey there!', 'Hi again!', 'Good to see you!']
ME = [f"I'm {NAME}, made by Kelus Corp.", f"I'm {NAME} from Kelus Corp.", f"{NAME} here, built by Kelus Corp.",
      f"I'm {NAME}. Kelus Corp. trained me from scratch."]
NEXT = ["What's on your mind?", 'How can I help today?', 'What would you like to talk about?', 'What can I do for you?',
        "Ask me anything and I'll do my best.", 'What are you up to?']
GREETING_REPLIES = [f'{h} {m} {n}' for h, m, n in itertools.product(HELLO, ME, NEXT)]

CAN = ["I can chat, look things up on the web, find places near you, help with writing and write short programs.",
       "I can answer questions with a web search, suggest places nearby, help you write things and write simple code.",
       "Friendly chat, quick web lookups, places near you, writing help and short programs are my thing.",
       "I can talk things through with you, look up facts on the web, find places nearby, and help with writing or code."]
ASK_BACK = ['What would you like to try?', 'What do you need?', 'Where should we start?', "What's on your mind?",
            'What can I help with today?', 'What sounds useful?']
ABILITY_REPLIES = ABILITIES + [f'{c} {a}' for c, a in itertools.product(CAN, ASK_BACK)]

# Something called off: bad news when it was something to look forward to, good news when it was something dreaded.
CALLED_OFF = [
    ("our camping trip got called off", "your camping trip being called off", 'bad'),
    ("the concert I waited months for got cancelled", "the concert getting cancelled", 'bad'),
    ("my sleepover got called off", "your sleepover being called off", 'bad'),
    ("our family reunion got cancelled", "your family reunion being cancelled", 'bad'),
    ("the fireworks show got called off", "the fireworks show being called off", 'bad'),
    ("my cooking class got cancelled", "your cooking class being cancelled", 'bad'),
    ("our ski trip got called off because there's no snow", "your ski trip being called off", 'bad'),
    ("the museum visit I planned got cancelled", "your museum visit being cancelled", 'bad'),
    ("my art workshop got called off", "your art workshop being called off", 'bad'),
    ("our road trip got cancelled at the last minute", "your road trip being cancelled", 'bad'),
    ("the pop quiz got called off!", "the pop quiz being called off", 'good'),
    ("the train strike got called off!", "the strike being called off", 'good'),
    ("the storm warning got lifted!", "the storm warning being lifted", 'good'),
    ("the dentist appointment I was dreading got cancelled!", "skipping the dentist", 'good'),
    ("my double shift got called off!", "getting out of the double shift", 'good'),
    ("the evacuation order got lifted and we can go home!", "getting to go home", 'good'),
    ("our rent increase got cancelled!", "the rent increase being cancelled", 'good'),
    ("the mandatory weekend meeting got cancelled!", "getting your weekend back", 'good'),
    ("the water shutoff got called off!", "the water shutoff being called off", 'good'),
    ("the noisy construction next door got called off!", "some peace and quiet", 'good'),
]

MADE_OF = [('glass', 'is'), ('paper', 'is'), ('chocolate', 'is'), ('concrete', 'is'), ('plastic', 'is'), ('steel', 'is'),
           ('mayonnaise', 'is'), ('soy sauce', 'is'), ('tofu', 'is'), ('mustard', 'is'), ('hummus', 'is'), ('pesto', 'is'),
           ('rubber', 'is'), ('marshmallows', 'are'), ('gummy bears', 'are'), ('chewing gum', 'is'), ('crayons', 'are'),
           ('toothpaste', 'is'), ('cement', 'is'), ('bronze', 'is'), ('nylon', 'is'), ('denim', 'is'), ('soap', 'is'),
           ('candles', 'are'), ('pencil lead', 'is'), ('marzipan', 'is'), ('sriracha', 'is'), ('ranch dressing', 'is'),
           ('peanut butter', 'is'), ('jelly beans', 'are'), ('paint', 'is'), ('tires', 'are'), ('coins', 'are'), ('bricks', 'are')]


def priors(rng, count):
    """Short single exchanges from instruct-v18's training rows, to come before a greeting."""
    rows = [json.loads(line) for line in (ROOT / 'data/instruct-v18.jsonl').read_text(encoding='utf-8').splitlines()]
    short = [r['messages'] for r in rows
             if r['split'] == 'train' and len(r['messages']) == 2 and not (r.get('family') or '').startswith(('reading', 'identity'))
             and len(r['messages'][0]['content']) <= 100 and len(r['messages'][1]['content']) <= 250
             and NAME not in r['messages'][1]['content'] and not held_out(r)]
    return rng.sample(short, count)


def late_greetings(rows, rng):
    for greeting in LATE_GREETINGS:
        assert normal(greeting) not in IDENTITY_HELD_OUT, greeting
    for earlier in priors(rng, 240):
        rows.add('late_greeting', [*earlier, user(rng.choice(LATE_GREETINGS))], GREETING_REPLIES)


def abilities(rows, rng):
    for ask in ABILITY_ASKS:
        for greeting in rng.sample(LATE_GREETINGS, 3):
            hello = rng.choice(GREETING_REPLIES)
            rows.add('abilities', [user(greeting), assistant(hello), user(ask)], ABILITY_REPLIES)
        rows.add('abilities', [user(ask)], ABILITY_REPLIES)


def called_off(rows, rng):
    for text, said, mood in CALLED_OFF:
        openings, follows = REPLIES[mood]
        replies = [f'{opening(said)} {follow}' for opening, follow in itertools.product(openings, follows)]
        for form in event_forms(text, mood, rng):
            reply = rows.add(f'called_off:{mood}', [user(form)], replies)
            if reply and rng.random() < 0.15:
                rows.add(f'called_off:{mood}', [user(form), assistant(reply), user(rng.choice(THANKS))],
                         [f"Anytime. I hope things with {said} get easier." if mood == 'bad' else f"Enjoy it, {said} is worth celebrating!"])


def made_of(rows, rng):
    for thing, verb in MADE_OF:
        for form in wordings(f'what {verb} {thing} made of?', rng, 2):
            rows.add('made_of', [user(form)], [f'SEARCH: what {verb} {thing} made of'])


def build(seed=2419):
    rng = random.Random(seed)
    rows = Rows(rng)
    late_greetings(rows, rng)
    abilities(rows, rng)
    called_off(rows, rng)
    made_of(rows, rng)
    new = [{**r, 'source': 'make_instruct_v19'} for r in rows.rows]
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]]
    new = new[cut:]
    v18 = [json.loads(line) for line in (ROOT / 'data/instruct-v18.jsonl').read_text(encoding='utf-8').splitlines()]
    replay = [r for r in v18 if r['split'] == 'train']
    validation += [r for r in v18 if r['split'] != 'train']
    seen, kept = collections.Counter(), []
    for row in new + replay:
        replies = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
        if held_out(row) or any(seen[r] >= CAP for r in replies if r != NO_ANSWER):
            continue
        seen.update(replies)
        kept.append(row)
    rng.shuffle(kept)
    return kept + [r for r in validation if not held_out(r)]


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v19.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant' and m['content'] != NO_ANSWER)
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); most copies of one chat reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
