"""instruct-v17: Kani Otter 4.8, after Otter 4.6 (shipped) and 4.7 (not shipped) (2026-10-30).

Every Otter so far gets the mood wrong when bad news names something people want: "I got rejected from my dream college"
and "my coworker took credit for my work" got "That's amazing, congrats on ...! How are you going to celebrate?". The
model keys on the good thing named (college, credit, best friend), not on what happened to it. PAIRS gives the same
thing going well and going badly, side by side, each answered the way instruct-v15 answers events (Otter 4.6 learned that
frame; 4.7's hand-written line per event was attached to the wrong events). None of kelus/ontopic_eval.py's subjects
(exams, driving tests, colleges, marathons, championships, phones, a grandma in hospital, a coworker, a sister, a best
friend moving) is used. instruct-v16's jokes come along: Otter 4.7 learned them without trouble.
Ahead of a REPLAY sample of instruct-v15's training rows (Otter 4.6's data) and all of its "the search results don't
say" rows. decide_eval's questions (instruct-v9 validation) and the other held-out prompts are left out.

    .venv/bin/python data/make_instruct_v17.py   # writes data/instruct-v17.jsonl
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
from make_instruct_v14 import CAP, THANKS, Rows, assistant, user  # noqa: E402
from make_instruct_v15 import REPLIES, held_out  # noqa: E402
from make_instruct_v16 import event_forms, jokes  # noqa: E402

NAME, OLD_NAME = 'Kani Otter 4.8', 'Kani Otter 4.6'
REPLAY = 10_000
VALIDATION_SHARE = 0.05

# (good news, said back), (bad news about the same thing, said back)
PAIRS = [
    (("I got the job I interviewed for!", "getting the job"), ("I didn't get the job I interviewed for", "not getting the job")),
    (("my scholarship application was accepted!", "your scholarship coming through"),
     ("my scholarship application got rejected", "your scholarship application being rejected")),
    (("my cousin is coming to stay with us for the summer!", "your cousin coming for the summer"),
     ("my cousin is moving back overseas for good", "your cousin moving back overseas")),
    (("my soccer team made the playoffs!", "making the playoffs"), ("my soccer team got knocked out of the playoffs", "your team getting knocked out")),
    (("the doctor said my test results are all clear!", "the all-clear from your doctor"),
     ("the doctor found something on my scan", "the doctor finding something on your scan")),
    (("we adopted a puppy today!", "your new puppy"), ("we had to give our puppy back to the shelter", "having to give your puppy back")),
    (("I got my whole security deposit back!", "getting your deposit back"), ("my landlord kept my whole security deposit", "your landlord keeping your deposit")),
    (("my photo won the contest!", "your photo winning the contest"), ("my photo didn't even place in the contest", "your photo not placing in the contest")),
    (("my partner and I just had our one-year anniversary!", "your one-year anniversary"),
     ("my partner and I just split up", "splitting up with your partner")),
    (("our offer on the house got accepted!", "your offer getting accepted"), ("we lost the house we made an offer on", "losing the house you wanted")),
    (("my visa got approved!", "your visa approval"), ("my visa got denied", "your visa being denied")),
    (("my manager praised my work in the meeting", "your manager praising your work"),
     ("my manager blamed me for the team's mistake", "your manager blaming you for the team's mistake")),
    (("a publisher wants my book!", "a publisher wanting your book"), ("another publisher turned down my book", "another publisher turning down your book")),
    (("I made the basketball team!", "making the basketball team"), ("I got cut from the basketball team", "getting cut from the basketball team")),
    (("my flight got upgraded to first class!", "your first-class upgrade"), ("my flight got delayed by six hours", "your six-hour delay")),
    (("someone found my lost dog and brought him home!", "getting your dog back"), ("we still haven't found my lost dog", "your dog still being missing")),
    (("I got an A on my history essay!", "your A on the history essay"), ("I got a D on my history essay", "the D on your history essay")),
    (("I got extra shifts at work this month!", "your extra shifts"), ("my hours at work got cut", "your hours getting cut")),
    (("my dad's surgery went really well!", "your dad's surgery going well"), ("my dad's surgery didn't go as planned", "your dad's surgery not going as planned")),
    (("an old friend messaged me after years!", "hearing from an old friend"), ("an old friend blocked me without saying why", "an old friend blocking you")),
    (("our band got booked for a real gig!", "your band's first real gig"), ("our gig got cancelled at the last minute", "your gig getting cancelled")),
    (("my vegetable garden is finally growing!", "your garden growing"), ("a storm flattened my whole garden", "losing your garden to the storm")),
    (("I finally bought my first car!", "your first car"), ("my car got towed", "your car getting towed")),
    (("my cake came out perfect for the party!", "your perfect cake"), ("my cake collapsed right before the party", "your cake collapsing")),
    (("the sun came out for our wedding!", "sunshine on your wedding day"), ("it poured all through our wedding", "rain on your wedding day")),
    (("I got the apartment I wanted!", "getting the apartment"), ("someone else got the apartment I wanted", "missing out on the apartment")),
    (("I beat the hardest level in the game!", "beating that level"), ("I lost my 200-day streak", "losing your 200-day streak")),
    (("my brother is coming home from the army!", "your brother coming home"), ("my brother is being deployed overseas", "your brother being deployed")),
    (("my cast finally came off!", "getting your cast off"), ("the doctor says I need my cast for two more months", "two more months in a cast")),
    (("I got tickets to the concert!", "getting concert tickets"), ("the concert sold out before I could get tickets", "missing out on concert tickets")),
    (("I found twenty dollars in my old jacket!", "finding twenty dollars"), ("I lost twenty dollars somewhere today", "losing twenty dollars")),
    (("they called me back for a second interview!", "the callback"), ("they never called me back after the interview", "never hearing back")),
    (("I won the spelling bee!", "winning the spelling bee"), ("I got out in the first round of the spelling bee", "going out in the first round")),
    (("my painting sold at the art fair!", "selling your painting"), ("nobody bought anything at my art fair stall", "a slow day at your stall")),
    (("the vet said my cat is healthy again!", "your cat being healthy again"), ("the vet said my cat needs surgery", "your cat needing surgery")),
    (("my son took his first steps today!", "your son's first steps"), ("my son broke his leg at the playground", "your son breaking his leg")),
    (("I got promoted to team lead!", "your promotion to team lead"), ("they gave the team lead job to someone else", "missing out on the team lead job")),
    (("we're finally going on vacation tomorrow!", "your vacation"), ("we had to cut our vacation short", "cutting your vacation short")),
    (("the police found my stolen bike!", "getting your bike back"), ("the police said they won't look for my stolen bike", "the police not looking for your bike")),
    (("my mom found a new job after months of looking!", "your mom finding a new job"), ("my mom lost her job", "your mom losing her job")),
    (("we're moving into a bigger house!", "moving into a bigger house"), ("we have to move out of our house", "having to move out")),
    (("my app got accepted to the app store!", "your app being accepted"), ("the app store rejected my app again", "the app store rejecting your app")),
    (("our fundraiser hit its goal!", "your fundraiser hitting its goal"), ("our fundraiser barely raised anything", "your fundraiser falling short")),
    (("my neighbor returned the ladder he borrowed, finally!", "getting your ladder back"), ("my neighbor broke the ladder he borrowed", "your neighbor breaking your ladder")),
    (("my science project got picked for the state fair!", "your project going to the state fair"),
     ("my science project didn't get picked for the state fair", "your project not getting picked")),
    (("my roommate paid me back the money she owed!", "getting paid back"), ("my roommate still won't pay me back", "your roommate not paying you back")),
    (("my favorite band announced a tour near me!", "your favorite band touring near you"), ("my favorite band broke up", "your favorite band breaking up")),
    (("my grades went up this semester!", "your grades going up"), ("my grades dropped this semester", "your grades dropping")),
    (("my little sister got the lead in her school play!", "your little sister getting the lead"),
     ("my little sister didn't get a part in the school play", "your little sister not getting a part")),
    (("the bakery I opened had its best week ever!", "your bakery's best week"), ("the bakery I opened might have to close", "your bakery maybe closing")),
]


def renamed(row):
    for message in row['messages']:
        message['content'] = message['content'].replace(OLD_NAME, NAME)
    return row


def pairs(rows, rng):
    for good, bad in PAIRS:
        for (text, said), mood in ((good, 'good'), (bad, 'bad')):
            openings, follows = REPLIES[mood]
            replies = [f'{opening(said)} {follow}' for opening, follow in itertools.product(openings, follows)]
            for form in event_forms(text, mood, rng):
                reply = rows.add(f'pair:{mood}', [user(form)], replies)
                if reply and rng.random() < 0.15:
                    rows.add(f'pair:{mood}', [user(form), assistant(reply), user(rng.choice(THANKS))],
                             [f"Anytime. I hope things with {said} get easier." if mood == 'bad' else f"Enjoy it, {said} is worth celebrating!"])


def build(seed=2417):
    rng = random.Random(seed)
    rows = Rows(rng)
    pairs(rows, rng)
    jokes(rows, rng)
    new = [renamed({**r, 'source': 'make_instruct_v17'}) for r in rows.rows]
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]]
    new = new[cut:]
    v15 = [json.loads(line) for line in (ROOT / 'data/instruct-v15.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [renamed(r) for r in v15 if r['split'] == 'train']
    fixed = [r for r in train if r.get('family') == 'reading_no_answer']
    replay = [r for r in train if r.get('family') != 'reading_no_answer']
    rng.shuffle(replay)
    validation += [renamed(r) for r in v15 if r['split'] != 'train']
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
    destination = ROOT / 'data/instruct-v17.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant' and m['content'] != NO_ANSWER)
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); most copies of one chat reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
