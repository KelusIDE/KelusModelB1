"""instruct-v13: open conversation first, Kani's skills second, and no canned replies.

Big Kani v2 chat-tuned on instruct-v12 still answered 15 of 30 unfamiliar messages with a canned line ("I just got a
puppy!" got the support message), while the same model tuned only on Dolly and OpenAssistant gave 0 of 30. v12 capped
each exact reply at 60 copies; the template families (chat_pairs, feelings, practiced) still taught matching. Here:
  - every exact reply appears at most CAP (3) times in training rows, across all sources;
  - Kani's skills from instruct-v11 are kept in a fixed amount per family (tool calls for search and places, reading
    with citations, code, phrases); everyday_dolly is dropped, since all of Dolly comes in with the instructions below;
  - human-written instructions: Dolly 15k and OpenAssistant 2, as filtered for kani-paragraph-mac-v1
    (training-runs/kani-advanced-instruction-pilot/data, data/make_advanced_instructions.py);
  - SMOLTALK_TRAIN conversations from HuggingFaceTB/smol-smoltalk, filtered as in make_instruct_v12.py;
  - the name, NAME, in many wordings (greetings, name and maker questions, "are you ChatGPT?") instead of one sentence,
    and renamed in every kept v11 reply.
User messages that match a held-out test prompt (kelus/identity_eval.py, kelus/offscript_eval.py) are left out.

    python3 data/make_instruct_v13.py   # writes data/instruct-v13.jsonl
"""
import collections
import itertools
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus import identity_eval, offscript_eval  # noqa: E402
from make_instruct_v12 import SMOLTALK, smoltalk  # noqa: E402

NAME, OLD_NAME = 'Kani Otter 4.4', 'Kani Otter 4.3'
CAP = 3
SMOLTALK_TRAIN, SMOLTALK_VALIDATION = 60_000, 1_000
INSTRUCTIONS = ROOT / 'training-runs/kani-advanced-instruction-pilot/data/instructions.jsonl'

SKILLS = {'places': 2500, 'places_fact': 600, 'places_need': 600, 'places_chat': 300, 'places_code': 234, 'reading': 2500,
          'decide': 2000, 'everyday_recipe_search': 1000, 'everyday_learn_search': 800, 'everyday_food_chat': 264,
          'everyday_phrases': 2000, 'python_compose': 2000, 'python_spec': 1000, 'python_power': 300, 'javascript_spec': 400,
          'website': 300, 'chat_pairs': 1500, 'feelings': 1000, 'practiced': 1000}


def normal(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


HELD_OUT = {normal(p) for p in identity_eval.GREETINGS + identity_eval.QUESTIONS + offscript_eval.PROMPTS}


def held_out(row):
    return any(m['role'] == 'user' and normal(m['content']) in HELD_OUT for m in row['messages'])


def renamed(row):
    for message in row['messages']:
        message['content'] = message['content'].replace(OLD_NAME, NAME)
    return row


def identity(rng):
    """Greetings and questions about Kani, each answer worded many ways (no exact reply more than CAP times)."""
    me = [f"I'm {NAME}, made by Kelus Corp.", f"I'm {NAME} from Kelus Corp.", f"{NAME} here, built by Kelus Corp.",
          f"This is {NAME}, a model Kelus Corp. trained from scratch.", f"I'm {NAME}. Kelus Corp. trained me from scratch."]
    hello = ['Hi!', 'Hello!', 'Hey!', 'Hi there!', 'Hello there!', 'Hey there!', 'Good to see you!', 'Hi, nice to meet you!']
    next_ = ["What's on your mind?", 'How can I help today?', 'What would you like to talk about?', 'What are you working on?',
             'Want to write, learn or code something?', "Ask me anything and I'll do my best.", 'What can I do for you?']
    about = ['I can chat, explain things, help with writing and write code.',
             'I help with everyday questions, writing and code, and I can search the web for facts.',
             "I'm small, so I can make mistakes, but I'm good at chat, writing and short programs.",
             'I run right on your device, and I can look things up on the web when I need facts.']
    greetings = ['hello', 'hey', 'hi there', 'hey kani', 'hello kani', 'good evening', 'good afternoon', 'howdy', 'hi!!',
                 'hellooo', 'morning!', 'hey, how are you?', 'hi, who am I talking to?', 'greetings', 'hey kani!', 'hi kani :)']
    questions = {
        'name': ['what is your name?', 'do you have a name?', 'may I ask your name?', 'whats ur name', 'what do people call you?',
                 'name?', 'what are you called?'],
        'maker': ['who built you?', 'who created you', 'who trained you?', 'which company made you?', 'who is behind you?'],
        'about': ['what can you do?', 'what are you good at?', 'describe yourself', 'can you introduce yourself',
                  'what kind of assistant are you?', 'what are you for?'],
        'version': ['what version is this?', "what's your model name?", 'which model is this?', 'are you the newest kani?'],
    }
    others = {'ChatGPT': 'OpenAI', 'GPT-4': 'OpenAI', 'Claude': 'Anthropic', 'Gemini': 'Google', 'Siri': 'Apple',
              'Copilot': 'Microsoft', 'Llama': 'Meta'}
    rows, used = [], collections.Counter()

    def add(prompt, choices, context=()):
        options = [c for c in choices if used[c] < CAP]
        if options:
            reply = rng.choice(options)
            used[reply] += 1
            rows.append({'split': 'train', 'family': 'identity_v13', 'source': 'make_instruct_v13',
                         'messages': [*context, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': reply}]})

    greeting_replies = [f'{h} {m} {n}' for h, m, n in itertools.product(hello, me, next_)]
    plain_replies = [f'{h} {n}' for h, n in itertools.product(hello, next_)]
    earlier = [[{'role': 'user', 'content': 'thanks for the help with my essay'}, {'role': 'assistant', 'content': "You're welcome! Good luck with it."}],
               [{'role': 'user', 'content': 'what is 5 times 6?'}, {'role': 'assistant', 'content': '5 times 6 is 30.'}],
               [{'role': 'user', 'content': 'I like rainy days'}, {'role': 'assistant', 'content': 'Rainy days are cozy. Do you like reading when it rains?'}]]
    for _ in range(14):
        for prompt in greetings:
            add(prompt, greeting_replies if rng.random() < 0.75 else plain_replies, rng.choice([(), (), *earlier]))
            add(prompt.capitalize(), greeting_replies)
    name_replies = [f'{m} {a}' for m, a in itertools.product(me, about)] + [f'{m} {n}' for m, n in itertools.product(me, next_)]
    maker_replies = [f'Kelus Corp. made me. {m} {a}' for m, a in itertools.product(me, about)] + \
                    [f'I was trained from scratch by Kelus Corp. {m}' for m in me]
    about_replies = [f'{m} {a} {n}' for m, a, n in itertools.product(me, about, next_)]
    version_replies = [f'This is {NAME}, the newest Kani from Kelus Corp. {a}' for a in about] + \
                      [f'{m} {a}' for m, a in itertools.product(me, about)]
    for _ in range(10):
        for kind, prompts in questions.items():
            choices = {'name': name_replies, 'maker': maker_replies, 'about': about_replies, 'version': version_replies}[kind]
            for prompt in prompts:
                add(prompt, choices, rng.choice([(), (), *earlier]))
        for other, company in others.items():
            for prompt in (f'are you {other}?', f'are you {other.lower()}', f'is this {other}?'):
                add(prompt, [f"No, I'm not {other}. {m} {other} is a different AI, made by {company}." for m in me] +
                    [f"Nope! {m} {other} comes from {company}; I'm a separate, much smaller model." for m in me])
    return [r for r in rows if not held_out(r)]


def build(seed=2413):
    rng = random.Random(seed)
    v11 = [json.loads(line) for line in (ROOT / 'data/instruct-v11.jsonl').read_text(encoding='utf-8').splitlines()]
    instructions = [json.loads(line) for line in INSTRUCTIONS.read_text(encoding='utf-8').splitlines()]
    train = smoltalk(sorted(SMOLTALK.glob('train-*.parquet')), SMOLTALK_TRAIN, 'train', rng)
    train += [r for r in instructions if r['split'] == 'train']
    train += identity(rng)
    by_family = collections.defaultdict(list)
    for row in v11:
        if row['split'] == 'train' and row.get('family') in SKILLS:
            by_family[row['family']].append(renamed(row))
    for family, rows in by_family.items():
        rng.shuffle(rows)
        train += rows[:SKILLS[family] * 3]


    rng.shuffle(train)
    seen, kept, families = collections.Counter(), [], collections.Counter()
    for row in train:
        replies = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
        family = row.get('family')
        if held_out(row) or any(seen[r] >= CAP for r in replies) or families[family] >= SKILLS.get(family, 10**9):
            continue
        seen.update(replies)
        families[family] += 1
        kept.append(row)
    validation = [renamed(r) for r in v11 if r['split'] != 'train']
    validation += [r for r in instructions if r['split'] != 'train']
    validation += smoltalk(sorted(SMOLTALK.glob('test-*.parquet')), SMOLTALK_VALIDATION, 'validation', rng)
    return kept + [r for r in validation if not held_out(r)]


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v13.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant')
    top = sum(n for _, n in replies.most_common(25)) / sum(replies.values())
    families = collections.Counter((r.get('family') or 'smoltalk').split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); the 25 most common replies are {top:.1%} of replies; '
          f'most copies of one reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
