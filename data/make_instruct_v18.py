"""instruct-v18: Kani Otter 4.9, after Otter 4.8 (shipped 2026-10-30).

Otter 4.8 (instruct-v17) got the mood right on 11 of 12 unseen messages, but:
  - "what can you do?" became "SEARCH: what can I do for you": no row says what Kani can do. ABILITIES answers that,
    asked in wordings that are not kelus/probes.py's held-out ones (its "abilities" list and context priors);
  - good news with a hospital in it was met with sympathy ("my grandpa got out of the hospital today!" got "I'm sorry
    about ... That's really rough."). RECOVERIES are such good news: someone home from the hospital, a clear scan,
    something lost that came back;
  - "i'm so bored" got "I'm sorry about boredom. That's really rough.": the events' frame took over feelings that are not
    events. instruct-v14's SITUATIONS (bored, lonely, can't sleep, procrastinating...) are written again with new
    wordings.
Ahead of all of instruct-v17's training rows. The held-out prompts stay out, as in v17.

    .venv/bin/python data/make_instruct_v18.py   # writes data/instruct-v18.jsonl
"""
import collections
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
from kelus import probes  # noqa: E402
from kelus.reading import NO_ANSWER  # noqa: E402
from make_instruct_v14 import CAP, THANKS, Rows, assistant, normal, situations, user, wordings  # noqa: E402
from make_instruct_v15 import FIRST, held_out  # noqa: E402

NAME, OLD_NAME = 'Kani Otter 4.9', 'Kani Otter 4.8'
VALIDATION_SHARE = 0.05

ABILITY_ASKS = ["what can you help me with?", "what kind of things can you do?", "what are you able to do?", "how can you help me?",
                "what stuff can you help with", "what do you know how to do?", "what kind of questions can you answer?",
                "what are your features?", "so what can you actually do", "what can i use you for?", "what kinds of things do you help with?",
                "what are you useful for?"]
ABILITIES = [
    "I can chat, look things up on the web, find places near you, help with writing and write short programs. What would you like to try?",
    "Lots of everyday things: talking through your day, answering questions with a web search, finding places nearby, and small bits of code. What do you need?",
    "I can answer questions (I search the web for facts), suggest places near you, help you write things and write simple programs. Where should we start?",
    "Chat with you, search the web when you ask about facts, find restaurants or parks nearby, help with writing, and write short code. What's on your mind?",
    "I'm good at friendly chat, quick web lookups, finding places near you, writing help and short programs. I'm small, so I can make mistakes. What would you like to do?",
    "I can talk things through with you, look up facts on the web, find places nearby, and help with writing or simple code. What can I help with today?",
    "Ask me a question and I'll search the web for it, ask for places near you, or ask for help writing something or a short program. Or we can just chat!",
    "Things like answering questions from the web, finding places nearby, helping you write a message or a story, and writing small programs. What sounds useful?",
]
HELD_OUT_ASKS = {normal(p) for lists in probes.CHAT.values() for p in lists} | {normal(prior) for prior, _ in probes.CONTEXT_PRIORS}

# Good news that names something bad (a hospital, a scan, a loss), said back.
RECOVERIES = [
    ("my mom is finally home from the hospital!", "your mom being home"),
    ("my uncle beat cancer!", "your uncle beating cancer"),
    ("the biopsy came back benign!", "the biopsy coming back clear"),
    ("my dog's surgery went perfectly!", "your dog's surgery going well"),
    ("I found the ring I thought I lost!", "finding your ring"),
    ("my broken wrist finally healed!", "your wrist healing"),
    ("my fever finally broke and I feel human again!", "feeling better"),
    ("my missing cat came home after three days!", "your cat coming home"),
    ("the flood missed our house!", "the flood missing your house"),
    ("my brother is out of the ICU!", "your brother leaving the ICU"),
    ("I'm finally done with chemo!", "finishing chemo"),
    ("my baby niece is home from the NICU!", "your niece coming home"),
    ("my teacher graded my test wrong and I actually passed!", "passing after all"),
    ("my lost luggage finally showed up!", "getting your luggage back"),
    ("we got our stolen car back!", "getting your car back"),
    ("my aunt woke up from her coma!", "your aunt waking up"),
    ("the layoffs skipped my team!", "your team being spared"),
    ("my dad is cancer free!", "your dad being cancer free"),
    ("my cousin got discharged from the hospital today!", "your cousin coming home"),
    ("the doctor says my knee won't need surgery after all!", "not needing surgery"),
    ("my best friend's heart surgery was a success!", "your friend's surgery going well"),
    ("the wildfire stopped before it reached our town!", "your town being safe"),
]
RELIEF = ([lambda r: f"That's such a relief! {FIRST(r)} is wonderful news.", lambda r: f"Oh, I'm so glad! {FIRST(r)} is great news.",
           lambda r: f"What a relief, I'm so happy about {r}!"],
          ["How are you feeling now?", "You must be so relieved. How is everyone doing?", "That must take a weight off. How are you celebrating?"])


def renamed(row):
    for message in row['messages']:
        message['content'] = message['content'].replace(OLD_NAME, NAME)
    return row


def abilities(rows, rng):
    for ask in ABILITY_ASKS:
        assert normal(ask) not in HELD_OUT_ASKS and 'what can you do' not in normal(ask), ask
        for form in wordings(ask, rng, 3):
            rows.add('abilities', [user(form)], ABILITIES)


def recoveries(rows, rng):
    openings, follows = RELIEF
    for text, said in RECOVERIES:
        replies = [f'{opening(said)} {follow}' for opening in openings for follow in follows]
        for form in wordings(text, rng, 4):
            reply = rows.add('recovery', [user(form)], replies)
            if reply and rng.random() < 0.15:
                rows.add('recovery', [user(form), assistant(reply), user(rng.choice(THANKS))], [f"Anytime! Enjoy {said}."])


def build(seed=2418):
    rng = random.Random(seed)
    rows = Rows(rng)
    abilities(rows, rng)
    recoveries(rows, rng)
    situations(rows, rng)
    new = [renamed({**r, 'source': 'make_instruct_v18'}) for r in rows.rows]
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]]
    new = new[cut:]
    v17 = [renamed(json.loads(line)) for line in (ROOT / 'data/instruct-v17.jsonl').read_text(encoding='utf-8').splitlines()]
    replay = [r for r in v17 if r['split'] == 'train']
    validation += [r for r in v17 if r['split'] != 'train']
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
    destination = ROOT / 'data/instruct-v18.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant' and m['content'] != NO_ANSWER)
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); most copies of one chat reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
