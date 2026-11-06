"""instruct-v16: Kani Otter 4.7, after Otter 4.6's tests (2026-10-30).

Otter 4.6 (instruct-v15) kept the subject better (switches 24/24) and said "the search results don't say" again (44/45),
but in chat:
  - feelings: v15 answered each event by saying it back ("I'm sorry about losing your wallet."), and the model learned the
    frame, not the words: "my cat died this morning" got "I'm sorry about your cat being dying", "i'm so bored" got
    "I'm sorry about boredom. That's really rough." Here the same EVENTS are answered with a short reaction, one
    hand-written line about that event, and a question; v15's event rows are not replayed;
  - feelings typed with "guess what," "ugh" or "kani," in front were looked up ("SEARCH: i got a bad grade"); those forms
    are added to the events;
  - jokes: "ok tell me a joke" became "SEARCH: what a simple joke"; v15 had one plain joke row. JOKES adds forty, asked in
    wordings that are not kelus/probes.py's held-out ones, alone, after a subject, and again ("another one").
Ahead of a REPLAY sample of instruct-v15's training rows and all of its "the search results don't say" rows.
decide_eval's questions (instruct-v9 validation) and the other held-out prompts are left out.

    .venv/bin/python data/make_instruct_v16.py   # writes data/instruct-v16.jsonl
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
from make_instruct_v14 import CAP, OPENERS, THANKS, Rows, assistant, normal, user  # noqa: E402
from make_instruct_v15 import EVENTS, REPLIES, held_out, openings  # noqa: E402

NAME, OLD_NAME = 'Kani Otter 4.7', 'Kani Otter 4.6'
REPLAY = 8_000
VALIDATION_SHARE = 0.05

# One line about each of instruct-v15's EVENTS, in the same order.
EVENT_LINES = [
    "Check whether your school account or cloud drive kept an autosave, and tell your teacher what happened today.",
    "Cancel your cards first, then retrace your steps; lots of wallets get handed in at a front desk or police station.",
    "Getting somewhere safe comes first, and roadside assistance or your insurance may cover the tow.",
    "Go to the airline desk right away; they can often put you on the next flight.",
    "Report it to the police with the serial number if you have it, and keep an eye on local resale sites.",
    "If the sign was unclear, you can usually contest it, so take a photo of the spot.",
    "A calm talk usually works better than a note on the fridge, and a labeled shelf can help.",
    "Being called out in front of people is humiliating, and you deserve to be spoken to with respect.",
    "It's fair to ask your manager what would make you the pick next time.",
    "Check your lease and the local rules on increases; sometimes a smaller raise can be negotiated.",
    "A vet visit is the safest call, especially if they stop eating or start hiding.",
    "It's okay to tell them it hurt; most friends feel terrible and want to make it right.",
    "Ordering in or making something quick like pasta can still save the night.",
    "The airline has to rebook you, and in many places you can ask for a refund instead.",
    "Rest it, keep it raised, and let people help with the everyday stuff for a while.",
    "Losing a final hurts more than any other game, because they got so close.",
    "Lay the pages flat to dry, and a classmate might let you copy anything that smudged.",
    "Losing a grandparent leaves a big hole, and grief can come in waves.",
    "It isn't your fault, even if it feels that way, and whatever you feel about it makes sense.",
    "Almost every performer has a pile of no's behind them, and each audition makes the next one easier.",
    "If it keeps happening, writing down what happened and talking to a counselor can help.",
    "That's a lonely feeling, and it doesn't mean people don't care about you.",
    "Wanting it that much shows you knew what you were aiming for, and you can ask them for feedback.",
    "Check the trash or recycle bin first, then any version history or backups like Git or cloud sync.",
    "Call local shelters and vets, and post a recent photo in neighborhood groups.",
    "Take photos of the damage for insurance and your landlord before you clean up.",
    "Sip water or an electrolyte drink in small amounts, and see a doctor if you can't keep fluids down.",
    "It stings, but it says nothing about what you're worth.",
    "It's fair to ask him to help pay for new ones.",
    "A late morning happens to everyone, and a quick message to your boss usually smooths it over.",
    "Most plants die from too much water, so the next one might do better with a little less.",
    "Check your coat pockets and the last place you sat down; keys hide in the silliest spots.",
    "Stuffing them with newspaper and letting them dry slowly can save more than you'd think.",
    "Put the request in writing; in many places heat is something they have to fix.",
    "Contact the seller with your order number; most will resend it or refund you.",
    "Cool showers, aloe and lots of water help, and stay out of the sun while it heals.",
    "Some games keep a backup or cloud save, so check the settings before you start over.",
    "Audiences notice much less than you think, and most actors have done it at least once.",
    "Trust takes a hit when someone does that, and you're allowed to be angry.",
    "Fillings are quick these days, and getting it done early keeps it from getting worse.",
    "Check whether you can get a refund or a credit, so you can rebook later.",
    "Keep practicing; a lot of bands hold tryouts again next season.",
    "Telling her yourself, soon, will go better than her finding it.",
    "Rejections are part of every writer's life, so send it to the next magazine on your list.",
    "See whether any bills can wait a few days, and look for food banks or community help if it's tight.",
    "Feeling left out hurts; it might have been an oversight, so it's okay to ask.",
    "Your phone's map and a nearby shop or café are good places to get your bearings.",
    "Disconnect from the internet, run a full scan, and change your passwords from another device.",
    "Everyone has a moment like that, and people forget it much faster than you think.",
    "Friendships change as people grow, and it's okay to miss how things were.",
    "It's sad when a story just stops; sometimes fans get a show picked up somewhere else.",
    "A locksmith or your landlord can get you in, and a spare key with a friend helps next time.",
    "Book a dentist soon; until then, a cold compress and pain relief from a pharmacy can help.",
    "A friendly talk with your neighbor is a good first step, since they may not know.",
    "Ask your manager for one or two specific things to work on, so it turns into a plan.",
    "That's scary, and it's normal to feel worried all the time right now.",
    "It's kind of you to care about him; sometimes just checking in means a lot.",
    "Look for a note or cameras nearby, and take photos for your insurance.",
    "Rest it, ice it and keep it raised; playing on it too soon can make it worse.",
    "It's worth telling your teacher who did what, calmly and with examples.",
    "Asking them about themselves takes the pressure off you.",
    "Finding one café, gym or club you like early on helps a new city feel like home.",
    "Tell them you're nervous; dentists can explain each step and go slower.",
    "Write down your questions for the doctor so nothing gets forgotten.",
    "Going in with a plan for how you'll fix it can make the talk easier.",
    "Get there early, and remember that a few bumps in the air are normal.",
    "Stick to the warm-up you know, and focus on your own lane.",
    "Bring a short list of what you've done this year; it makes the ask easy to say yes to.",
    "Pick one friendly face in the crowd and sing to them.",
    "Ask them about themselves and be yourself; they mostly want to see that you're kind.",
    "Keep it short, one story and one wish for them, and practice it out loud.",
    "Asking someone how they know the host is an easy way to start talking.",
    "Practicing the questions out loud with a friend helps a lot.",
    "Waiting is often the hardest part, so plan something to keep your mind busy.",
    "Coaches notice effort and attitude as much as skill.",
    "Most kids are nervous on the first day too, and a simple \"hi, I'm new\" goes a long way.",
    "Practice the first minute until it's smooth; the rest usually follows.",
    "Tell someone where you'll be and when you'll be back, and set up camp before dark.",
    "You earned that.",
    "All that work on your project paid off.",
    "Finishing is the hardest part, and now you have something to hang up.",
    "You've got a new niece or nephew!",
    "Your first paycheck is a real milestone.",
    "That takes guts and a lot of wobbling.",
    "That's months of hard work, done.",
    "A place of your own is huge.",
    "That's wonderful news.",
    "All that practice in the pool paid off.",
    "Your hard work showed.",
    "A first show is something you'll always remember.",
    "That takes a lot of focus and practice.",
    "There's nothing like the moment it finally works.",
    "Seeing your name in print is a great feeling.",
    "That will make a big difference.",
    "What a lovely surprise.",
    "Every single word, nice.",
    "People really like what you make.",
    "That's a big step in learning a language.",
    "You must have nailed the audition.",
    "Nothing tastes better than tomatoes you grew yourself.",
    "That's a weight off your shoulders.",
    "That takes years of training.",
    "A new friend is the best part of any day.",
    "You must be proud of him.",
    "Fresh bread you made yourself is the best.",
    "That's a big step toward the job you want.",
]
REACTIONS = {
    'bad': ["Oh no, I'm so sorry.", "Ugh, that's rough.", "That's awful, I'm sorry.", "Oh no.", "I'm really sorry, that sounds hard."],
    'nervous': ["That's a big moment.", "It's normal to feel nervous.", "Butterflies make sense before something like that.",
                "That's exciting and scary at the same time."],
    'good': ["Congratulations!", "That's amazing!", "Wow, well done!", "That's great news!", "Yay, congrats!"],
}
PREFIXES = {'bad': ['ugh ', 'guess what, ', 'kani, '], 'nervous': ['kani, ', 'so '], 'good': ['guess what, ', 'kani!! ']}

JOKES = [
    ("Why don't scientists trust atoms? Because they make up everything.", 'science'),
    ("Why did the scarecrow win an award? Because he was outstanding in his field.", ''),
    ("What do you call a fake noodle? An impasta.", 'food'),
    ("Why did the bicycle fall over? It was two-tired.", ''),
    ("What do you call a bear with no teeth? A gummy bear.", 'animals'),
    ("Why can't a nose be twelve inches long? Because then it would be a foot.", 'math'),
    ("What did the ocean say to the beach? Nothing, it just waved.", 'ocean'),
    ("Why did the math book look so sad? It had too many problems.", 'math'),
    ("How does a penguin build its house? Igloos it together.", 'animals'),
    ("What do you call cheese that isn't yours? Nacho cheese.", 'food'),
    ("Why don't eggs tell jokes? They'd crack each other up.", 'food'),
    ("What do you call a sleeping dinosaur? A dino-snore.", 'animals'),
    ("Why did the cookie go to the doctor? Because it felt crummy.", 'food'),
    ("What did one wall say to the other wall? I'll meet you at the corner.", ''),
    ("Why are skeletons so calm? Nothing gets under their skin.", ''),
    ("What do you call a fish with no eyes? A fsh.", 'ocean'),
    ("Why did the golfer bring two pairs of pants? In case he got a hole in one.", 'sports'),
    ("I'm reading a book about anti-gravity. It's impossible to put down.", 'science'),
    ("Why did the tomato blush? Because it saw the salad dressing.", 'food'),
    ("What kind of tree fits in your hand? A palm tree.", ''),
    ("Why do cows wear bells? Because their horns don't work.", 'animals'),
    ("What do you call a factory that makes okay products? A satisfactory.", ''),
    ("Why don't skeletons fight each other? They don't have the guts.", ''),
    ("Why was the computer cold? It left its Windows open.", 'computers'),
    ("What did the zero say to the eight? Nice belt.", 'math'),
    ("Why did the student eat his homework? The teacher said it was a piece of cake.", 'school'),
    ("How do you plan a party in space? You planet.", 'space'),
    ("What's orange and sounds like a parrot? A carrot.", 'food'),
    ("Why did the coffee file a police report? It got mugged.", 'food'),
    ("What do you call a dog that does magic tricks? A labracadabrador.", 'animals'),
    ("Why don't cats play poker in the jungle? Too many cheetahs.", 'animals'),
    ("What do you call a can opener that doesn't work? A can't opener.", ''),
    ("What did the grape do when it got stepped on? It let out a little wine.", 'food'),
    ("Parallel lines have so much in common. It's a shame they'll never meet.", 'math'),
    ("What has four wheels and flies? A garbage truck.", ''),
    ("Why did the banana go to the doctor? It wasn't peeling well.", 'food'),
    ("I only know 25 letters of the alphabet. I don't know y.", 'school'),
    ("Why do bees have sticky hair? Because they use honeycombs.", 'animals'),
    ("Well, I used to hate facial hair, but it grew on me.", ''),
    ("Why did the moon skip dinner? It was already full.", 'space'),
    ("Why was the broom late? It over-swept.", ''),
    ("Why did the soccer player bring string to the game? So he could tie the score.", 'sports'),
    ("Why did the computer go to the doctor? It had a virus.", 'computers'),
    ("Why was the teacher wearing sunglasses? Her students were so bright.", 'school'),
]
JOKE_ASKS = ["tell me a funny joke", "do you know a good joke?", "i need a laugh", "make me laugh", "got any dad jokes?",
             "say something funny", "cheer me up with a joke", "what's your best joke?", "tell me a dad joke",
             "i want to hear a joke", "give me a joke", "share a joke with me", "hit me with a joke", "any good jokes?",
             "tell me a pun", "can you make me laugh?"]
JOKE_TOPICS = {'food': 'food', 'animals': 'animals', 'math': 'math', 'school': 'school', 'space': 'space', 'computers': 'computers',
               'ocean': 'the ocean', 'sports': 'sports', 'science': 'science'}
ANOTHER = ["another one", "haha another", "one more!", "lol tell me another", "ok another joke", "that was good, more"]
JOKE_INTROS = ['', '', "Here's one: ", 'Okay! ', 'Sure! ']
HELD_OUT_JOKES = {normal(p) for p in list(probes.CHAT['joke']) + [prior for prior, _ in probes.CONTEXT_PRIORS]}


def renamed(row):
    for message in row['messages']:
        message['content'] = message['content'].replace(OLD_NAME, NAME)
    return row


def event_forms(text, mood, rng):
    """The message as typed: as it is, lower case, trailing dots, or after an opener like "guess what,"."""
    plain = text.rstrip('!?.')
    forms = [text, text.lower(), f'{plain}...', f'hey kani, {text}'] + [f'{prefix}{text.lower()}' for prefix in PREFIXES[mood]]
    return rng.sample(list(dict.fromkeys(forms)), 4)


def events(rows, rng):
    assert len(EVENT_LINES) == len(EVENTS)
    for (text, said, mood), line in zip(EVENTS, EVENT_LINES):
        follows = REPLIES[mood][1]
        replies = [f'{reaction} {line} {follow}' for reaction in REACTIONS[mood] for follow in follows]
        for form in event_forms(text, mood, rng):
            reply = rows.add(f'event:{mood}', [user(form)], replies)
            if reply and rng.random() < 0.15:
                rows.add(f'event:{mood}', [user(form), assistant(reply), user(rng.choice(THANKS))],
                         ["Anytime. I hope it gets easier from here." if mood == 'bad' else "Good luck, you've got this!" if mood == 'nervous'
                          else "Enjoy it, that's worth celebrating!"])


def jokes(rows, rng):
    told = [joke for joke, _ in JOKES]
    by_topic = collections.defaultdict(list)
    for joke, topic in JOKES:
        if topic:
            by_topic[topic].append(joke)
    told_with = lambda choices: [f'{intro}{joke}' for joke in choices for intro in JOKE_INTROS]
    subjects = openings(rng)
    for ask in JOKE_ASKS:
        assert normal(ask) not in HELD_OUT_JOKES
        for form in rng.sample([ask, ask.capitalize(), f'kani {ask}', f'ok {ask}', f'{ask} lol'], 3):
            first = rows.add('joke', [user(form)], told_with(told))
            if first and rng.random() < 0.5:
                rows.add('joke', [user(form), assistant(first), user(rng.choice(ANOTHER))],
                         told_with([joke for joke in told if joke not in first]))
        # After a subject, a joke is still a joke, not a search about the subject.
        say, questions, names = rng.choice(subjects)
        opening = [user(say), assistant(f'{rng.choice(OPENERS)} {rng.choice(questions).format(**names)}')]
        rows.add('joke:switch', opening + [user(ask)], told_with(told))
    for topic, said in JOKE_TOPICS.items():
        for ask in [f'tell me a joke about {said}', f'do you know any {said} jokes?', f'{said} joke pls']:
            rows.add('joke:topic', [user(ask)], told_with(by_topic[topic]))


def build(seed=2416):
    rng = random.Random(seed)
    rows = Rows(rng)
    events(rows, rng)
    jokes(rows, rng)
    new = [{**r, 'source': 'make_instruct_v16'} for r in rows.rows]
    rng.shuffle(new)
    cut = int(len(new) * VALIDATION_SHARE)
    validation = [{**r, 'split': 'validation'} for r in new[:cut]]
    new = new[cut:]
    v15 = [json.loads(line) for line in (ROOT / 'data/instruct-v15.jsonl').read_text(encoding='utf-8').splitlines()]
    # v15's own event rows taught the "I'm sorry about <said back>" frame; they are not replayed.
    train = [renamed(r) for r in v15 if r['split'] == 'train' and not str(r.get('family', '')).startswith('event:')]
    fixed = [r for r in train if r.get('family') == 'reading_no_answer']
    replay = [r for r in train if r.get('family') != 'reading_no_answer']
    rng.shuffle(replay)
    validation += [renamed(r) for r in v15 if r['split'] != 'train' and not str(r.get('family', '')).startswith('event:')]
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
    destination = ROOT / 'data/instruct-v16.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    train = [r for r in rows if r['split'] == 'train']
    replies = collections.Counter(m['content'] for r in train for m in r['messages'] if m['role'] == 'assistant' and m['content'] != NO_ANSWER)
    families = collections.Counter((r.get('family') or 'smoltalk').split(':')[0].split('/')[0] for r in train)
    print(f'{destination}: {len(rows):,} rows ({len(train):,} train); most copies of one chat reply: {replies.most_common(1)[0][1]}')
    print(dict(families.most_common()))
