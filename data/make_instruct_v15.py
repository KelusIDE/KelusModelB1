"""instruct-v15: Kani Otter 4.6, after Otter 4.5's tests (2026-10-29).

Otter 4.5 (instruct-v14) kept the subject in follow-ups (17/18), but:
  - feelings: it recombined v14's 26 hand-written situations, so "my phone broke" got "Ask your teacher what went wrong" and
    "I just got my driver's license!" got "a new pet!". EVENTS adds about a hundred more things that happen, each answered
    by saying back what happened ("Oh no, losing your wallet sounds awful."), so the reply follows the message, not the
    nearest template. None of them is a kelus/ontopic_eval.py event;
  - reading: it never said "the search results don't say" (0/45). instruct-v13's cap of 3 copies per reply had dropped
    all but a few of instruct-v11's 684 examples of that fixed reply; here NO_ANSWER rows come back, exempt from the cap;
  - search: it answered some facts from memory, wrongly ("Charles Sanders Peirce was a British nobleman"). More of
    instruct-v11's fact questions that are looked up come back;
  - switches: v14 had 16 standalone questions to change the subject with, so at most 48 rows, all fact questions; Otter
    4.5 then turned "give me tips to sleep better" after "I want to learn guitar" into a guitar search. Here the new message
    after a subject is one of instruct-v11's looked-up fact questions, or a request from the replayed conversations
    (SmolTalk, Dolly, OpenAssistant) answered as it is, with no trace of the old subject;
  - thank-you turns: fewer, so a second message about what happened is not taken for thanks.
Ahead of a REPLAY sample of instruct-v14's training rows. decide_eval's questions (instruct-v9 validation) and the other
held-out prompts are left out.

    .venv/bin/python data/make_instruct_v15.py   # writes data/instruct-v15.jsonl
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
from make_instruct_v14 import (ACTIVITIES, CITIES, OPENERS, PETS, PET_QUESTIONS, PURCHASES, PURCHASE_QUESTIONS, TRIP_QUESTIONS,  # noqa: E402
                               TRIP_SAYS, THANKS, CAP, Rows, assistant, held_out as v14_held_out, normal, user, wordings)

NAME, OLD_NAME = 'Kani Otter 4.6', 'Kani Otter 4.5'
REPLAY = 12_000
NO_ANSWERS, READING_ANSWERS, LOOKUPS, SWITCH_ROWS, REQUEST_SWITCHES = 600, 1000, 1500, 400, 700
VALIDATION_SHARE = 0.05

# (what the person says, what happened said back to them, 'bad' / 'nervous' / 'good')
EVENTS = [
    ("my laptop died and I lost my homework", "losing your homework when your laptop died", 'bad'),
    ("I lost my wallet today", "losing your wallet", 'bad'),
    ("my car broke down on the highway", "your car breaking down on the highway", 'bad'),
    ("I missed my flight", "missing your flight", 'bad'),
    ("my bike got stolen", "having your bike stolen", 'bad'),
    ("I got a parking ticket", "getting a parking ticket", 'bad'),
    ("my roommate keeps eating my food", "your roommate eating your food", 'bad'),
    ("my boss yelled at me in front of everyone", "your boss yelling at you in front of everyone", 'bad'),
    ("I didn't get the promotion", "missing out on the promotion", 'bad'),
    ("my rent went up again", "your rent going up again", 'bad'),
    ("my cat is sick", "your cat being sick", 'bad'),
    ("my best friend forgot my birthday", "your best friend forgetting your birthday", 'bad'),
    ("I burned dinner and now everyone is hungry", "burning dinner", 'bad'),
    ("my flight got cancelled", "your flight getting cancelled", 'bad'),
    ("I broke my arm", "breaking your arm", 'bad'),
    ("my favorite team lost the final", "your team losing the final", 'bad'),
    ("I spilled coffee all over my notes", "spilling coffee all over your notes", 'bad'),
    ("my grandpa passed away", "losing your grandpa", 'bad'),
    ("my parents are getting divorced", "your parents getting divorced", 'bad'),
    ("I failed my audition", "not getting through your audition", 'bad'),
    ("my teacher was unfair to me today", "your teacher being unfair to you", 'bad'),
    ("nobody came to my party", "nobody coming to your party", 'bad'),
    ("I got turned down for the job I really wanted", "being turned down for the job you wanted", 'bad'),
    ("I accidentally deleted my whole project", "deleting your whole project", 'bad'),
    ("my dog ran away", "your dog running away", 'bad'),
    ("my apartment flooded", "your apartment flooding", 'bad'),
    ("I got food poisoning", "getting food poisoning", 'bad'),
    ("my crush likes someone else", "finding out your crush likes someone else", 'bad'),
    ("my brother broke my headphones", "your brother breaking your headphones", 'bad'),
    ("the bus left without me and I was late for work", "missing the bus and being late for work", 'bad'),
    ("my plant died", "your plant dying", 'bad'),
    ("I lost my keys", "losing your keys", 'bad'),
    ("my new shoes got ruined in the rain", "your new shoes getting ruined", 'bad'),
    ("my landlord won't fix the heater", "your landlord not fixing the heater", 'bad'),
    ("my order never arrived", "your order never arriving", 'bad'),
    ("I got a really bad sunburn", "getting a bad sunburn", 'bad'),
    ("my game save got corrupted", "losing your game save", 'bad'),
    ("I forgot my lines on stage", "forgetting your lines on stage", 'bad'),
    ("my friend told everyone my secret", "your friend telling everyone your secret", 'bad'),
    ("I have a cavity", "getting a cavity", 'bad'),
    ("my vacation got cancelled", "your vacation getting cancelled", 'bad'),
    ("I didn't get into the band", "not getting into the band", 'bad'),
    ("I scratched my mom's car", "scratching your mom's car", 'bad'),
    ("the magazine rejected my story", "the magazine rejecting your story", 'bad'),
    ("I have no money left until payday", "running out of money before payday", 'bad'),
    ("my friends went out without inviting me", "your friends going out without inviting you", 'bad'),
    ("I got lost in a new city", "getting lost in a new city", 'bad'),
    ("my computer got a virus", "your computer getting a virus", 'bad'),
    ("I tripped and fell in front of my whole class", "falling in front of your whole class", 'bad'),
    ("my best friend and I aren't close anymore", "drifting apart from your best friend", 'bad'),
    ("my favorite show got cancelled", "your favorite show getting cancelled", 'bad'),
    ("I locked myself out of my house", "locking yourself out of your house", 'bad'),
    ("my tooth hurts so much", "your toothache", 'bad'),
    ("my neighbor's dog barks all night", "your neighbor's dog barking all night", 'bad'),
    ("I got a bad review at work", "getting a bad review at work", 'bad'),
    ("my mom is in the hospital", "your mom being in the hospital", 'bad'),
    ("my uncle lost his job", "your uncle losing his job", 'bad'),
    ("someone hit my car in the parking lot", "someone hitting your car in the parking lot", 'bad'),
    ("I sprained my ankle before the game", "spraining your ankle before the game", 'bad'),
    ("my group project partners did nothing", "doing the group project alone", 'bad'),
    ("I have my first date tonight", "your first date", 'nervous'),
    ("I'm moving to a new city next week", "moving to a new city", 'nervous'),
    ("I have a dentist appointment tomorrow and I'm scared", "your dentist appointment", 'nervous'),
    ("I'm having surgery next week", "your surgery", 'nervous'),
    ("I have to tell my parents I failed a class", "telling your parents about the class", 'nervous'),
    ("I'm flying for the first time", "your first flight", 'nervous'),
    ("I have a big swim meet on Saturday", "your swim meet", 'nervous'),
    ("I have to ask my boss for a raise", "asking your boss for a raise", 'nervous'),
    ("I'm singing at the talent show tonight", "singing at the talent show", 'nervous'),
    ("I'm meeting my partner's parents tomorrow", "meeting your partner's parents", 'nervous'),
    ("I have to give a toast at my brother's wedding", "the toast at your brother's wedding", 'nervous'),
    ("I'm going to a party where I don't know anyone", "a party where you don't know anyone", 'nervous'),
    ("I have my citizenship interview next month", "your citizenship interview", 'nervous'),
    ("I get my blood test results tomorrow", "getting your blood test results", 'nervous'),
    ("I have a soccer tryout tomorrow", "your soccer tryout", 'nervous'),
    ("I'm starting at a new school on Monday", "starting at a new school", 'nervous'),
    ("I have to present my project to the whole company", "presenting your project to the whole company", 'nervous'),
    ("I'm going camping alone for the first time", "camping alone for the first time", 'nervous'),
    ("I just got a raise!", "your raise", 'good'),
    ("I won the science fair!", "winning the science fair", 'good'),
    ("I finished my first painting", "finishing your first painting", 'good'),
    ("my sister just had a baby!", "your sister's new baby", 'good'),
    ("I got my first paycheck", "your first paycheck", 'good'),
    ("I learned to ride a bike today", "learning to ride a bike", 'good'),
    ("I finally finished my thesis", "finishing your thesis", 'good'),
    ("we bought our first house!", "your first house", 'good'),
    ("I got engaged!", "your engagement", 'good'),
    ("I passed my swimming test", "passing your swimming test", 'good'),
    ("I made the honor roll", "making the honor roll", 'good'),
    ("my band played our first show", "your band's first show", 'good'),
    ("I won a chess tournament", "winning the chess tournament", 'good'),
    ("I finally fixed the bug I was stuck on", "fixing that bug", 'good'),
    ("my article got published", "getting your article published", 'good'),
    ("I got a scholarship!", "your scholarship", 'good'),
    ("my parents surprised me with a trip", "the surprise trip", 'good'),
    ("I got a perfect score on my spelling test", "a perfect score on your spelling test", 'good'),
    ("my channel hit 1,000 subscribers", "hitting 1,000 subscribers", 'good'),
    ("I finished reading a whole novel in Spanish", "finishing a whole novel in Spanish", 'good'),
    ("I got the lead role in the play", "getting the lead role", 'good'),
    ("my garden grew its first tomatoes", "your first tomatoes", 'good'),
    ("I paid off my credit card", "paying off your credit card", 'good'),
    ("I earned my black belt", "earning your black belt", 'good'),
    ("I made a new friend at school today", "making a new friend", 'good'),
    ("my little brother learned to read", "your little brother learning to read", 'good'),
    ("I baked my first loaf of bread and it worked", "your first loaf of bread", 'good'),
    ("I got my dream internship!", "your dream internship", 'good'),
]
FIRST = lambda text: text[0].upper() + text[1:]
REPLIES = {
    'bad': ([lambda r: f"I'm sorry about {r}. That's really rough.", lambda r: f"Oh no, {r} sounds awful.",
             lambda r: f"Ugh, I'm sorry. {FIRST(r)} is so frustrating.", lambda r: f"{FIRST(r)} is a lot to deal with, and it's okay to feel upset."],
            ["Do you want to talk about what happened?", "How are you holding up?", "Is there anything I can help you figure out?",
             "What's bothering you most about it?"]),
    'nervous': ([lambda r: f"It makes sense to feel nervous about {r}.", lambda r: f"{FIRST(r)} is a big deal, so a few nerves are normal.",
                 lambda r: f"Lots of people get butterflies about {r}."],
                ["What part are you most worried about?", "Want to talk through how it might go?",
                 "Is there anything you could get ready tonight so you feel more prepared?"]),
    'good': ([lambda r: f"Congratulations on {r}!", lambda r: f"That's amazing, congrats on {r}!", lambda r: f"Wow, {r}! That's huge."],
             ["How are you going to celebrate?", "How does it feel?", "Tell me more, what was the best part?"]),
}


def renamed(row):
    for message in row['messages']:
        message['content'] = message['content'].replace(OLD_NAME, NAME)
    return row


def decide_held_out():
    rows = [json.loads(line) for line in (ROOT / 'data/instruct-v9.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return {normal(m['content']) for r in rows if r['split'] == 'validation' for m in r['messages'] if m['role'] == 'user'}


DECIDE_HELD_OUT = decide_held_out()


def held_out(row):
    return v14_held_out(row) or any(m['role'] == 'user' and normal(m['content']) in DECIDE_HELD_OUT for m in row['messages'])


def events(rows, rng):
    for text, said, mood in EVENTS:
        openings, follows = REPLIES[mood]
        replies = [f'{opening(said)} {follow}' for opening, follow in itertools.product(openings, follows)]
        for form in wordings(text, rng, 3):
            reply = rows.add(f'event:{mood}', [user(form)], replies)
            if reply and rng.random() < 0.15:
                rows.add(f'event:{mood}', [user(form), assistant(reply), user(rng.choice(THANKS))],
                         [f"Anytime. I hope things with {said} get easier." if mood == 'bad' else f"Good luck with {said}!" if mood == 'nervous'
                          else f"Enjoy it, {said} is worth celebrating!"])


def openings(rng):
    """Subjects from instruct-v14's follow-ups, as an opening message and Kani's question back."""
    found = []
    for says, _, _, questions, _ in ACTIVITIES:
        found += [(say, questions, {}) for say in says]
    for animal, _, says in PETS:
        found += [(say, PET_QUESTIONS, {'animal': animal}) for say in says]
    for city in CITIES:
        found += [(say.format(city=city), TRIP_QUESTIONS, {'city': city}) for say in TRIP_SAYS]
    for thing, _, says in PURCHASES:
        found += [(say, PURCHASE_QUESTIONS, {'thing': thing}) for say in says]
    return found


def v11_rows(rng):
    v11 = [json.loads(line) for line in (ROOT / 'data/instruct-v11.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [r for r in v11 if r['split'] == 'train' and not held_out(r)]
    rng.shuffle(train)
    no_answers = [r for r in train if r.get('family') == 'reading' and r['messages'][-1]['content'] == NO_ANSWER][:NO_ANSWERS]
    answers = [r for r in train if r.get('family') == 'reading' and r['messages'][-1]['content'] != NO_ANSWER][:READING_ANSWERS]
    lookups = [r for r in train if r.get('family') == 'decide' and len(r['messages']) == 2 and r['messages'][-1]['content'].startswith('SEARCH: ')]
    return no_answers, answers, lookups


def build(seed=2415):
    rng = random.Random(seed)
    rows = Rows(rng)
    events(rows, rng)
    no_answers, answers, lookups = v11_rows(rng)
    # Changing the subject: after an opening, a fact question that is looked up as it was asked.
    subjects = openings(rng)
    switches = []
    for row in lookups[LOOKUPS:LOOKUPS + SWITCH_ROWS]:
        say, questions, names = rng.choice(subjects)
        opening = [user(say), assistant(f'{rng.choice(OPENERS)} {rng.choice(questions).format(**names)}')]
        switches.append({'split': 'train', 'family': 'follow_up:switch', 'source': 'make_instruct_v15', 'messages': opening + row['messages']})
    new = rows.rows + switches + [renamed(r) for r in answers + lookups[:LOOKUPS]]
    fixed = [{**r, 'family': 'reading_no_answer'} for r in no_answers]
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]]
    new = new[cut:]
    v14 = [json.loads(line) for line in (ROOT / 'data/instruct-v14.jsonl').read_text(encoding='utf-8').splitlines()]
    replay = [renamed(r) for r in v14 if r['split'] == 'train']
    rng.shuffle(replay)
    # Changing the subject to a request: an opening, then a one-turn request from the replayed conversations, answered as it is.
    requests = [r for r in replay[REPLAY:] if len(r['messages']) == 2 and str(r.get('source', '')).startswith(('smol-smoltalk', 'dolly', 'oasst'))
                and len(r['messages'][0]['content']) <= 200 and len(r['messages'][1]['content']) <= 1200
                and not r['messages'][1]['content'].startswith(('SEARCH:', 'PLACES:'))][:REQUEST_SWITCHES]
    for row in requests:
        say, questions, names = rng.choice(subjects)
        opening = [user(say), assistant(f'{rng.choice(OPENERS)} {rng.choice(questions).format(**names)}')]
        new.append({**row, 'family': 'follow_up:switch_request', 'source': 'make_instruct_v15', 'messages': opening + row['messages']})
    validation += [renamed(r) for r in v14 if r['split'] != 'train']
    # New rows first, so they keep their replies when the cap thins out repeats. NO_ANSWER is a fixed tool reply,
    # not a canned chat reply: its rows are not capped.
    seen, kept = collections.Counter(), list(fixed)
    for row in new + replay[:REPLAY]:
        replies = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
        if held_out(row) or any(seen[r] >= CAP for r in replies if r != NO_ANSWER):
            continue
        seen.update(replies)
        kept.append(row)
    rng.shuffle(kept)
    return kept + [r for r in validation if not held_out(r)]


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/instruct-v15.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant' and m['content'] != NO_ANSWER)
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); most copies of one chat reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
