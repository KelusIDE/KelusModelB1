"""Places: questions about where to go become a places call, answered from Google Maps.

The reply "PLACES: <what> @ <where>" is not shown. <where> is "here" when the person means near them (the
app then asks for their approximate location), or the place they named. The app looks the places up
(Google Places, through Kani's search service) and lists them with Maps links; the model only decides and
writes the call. Every example is made up here: no Google Maps content is used for training.

Covered, widely: food and drink, errands, health, leisure and travel; near-me wording ("nearest", "around
here", "close by"); named places ("in Rome", "near Central Park"); open-ended asks ("where should I go
tonight?", "I'm hungry, where should I eat?"); modifiers ("cheap", "vegan", "open now"); openers, lowercase,
missing punctuation and common misspellings, which the call spells correctly.
Contrasts keep their usual replies: facts about places search the web, code about places is written, and
chat about places stays chat. Validation holds out whole things, cities, landmarks and phrasings, so it
measures generalizing, not recall.

v2 adds "I need ..." and boredom messages that want a chat reply, not a place: v1's model learned that "I need a
pep talk" or "entertain me" meant somewhere to go.

    python3 data/make_places.py      # writes data/places-v2.jsonl
"""
import json
from pathlib import Path
import random
import re

from make_decide import query as search_query

ROOT = Path(__file__).resolve().parent.parent


THINGS = {
    'food': [('restaurant', 'restaurants'), ('pizza place', 'pizza places'), ('sushi restaurant', 'sushi restaurants'), ('ramen shop', 'ramen shops'),
             ('burger joint', 'burger joints'), ('taco truck', 'taco trucks'), ('Thai restaurant', 'Thai restaurants'), ('Indian restaurant', 'Indian restaurants'),
             ('Italian restaurant', 'Italian restaurants'), ('Mexican restaurant', 'Mexican restaurants'), ('Chinese restaurant', 'Chinese restaurants'),
             ('Korean barbecue place', 'Korean barbecue places'), ('brunch spot', 'brunch spots'), ('bakery', 'bakeries'), ('coffee shop', 'coffee shops'),
             ('cafe', 'cafes'), ('ice cream shop', 'ice cream shops'), ('dessert place', 'dessert places'), ('bar', 'bars'), ('pub', 'pubs'),
             ('steakhouse', 'steakhouses'), ('seafood restaurant', 'seafood restaurants'), ('food truck', 'food trucks'), ('noodle shop', 'noodle shops'),
             ('boba shop', 'boba shops'), ('donut shop', 'donut shops'), ('diner', 'diners'), ('fast food place', 'fast food places'), ('buffet', 'buffets'),
             ('Vietnamese restaurant', 'Vietnamese restaurants'), ('Greek restaurant', 'Greek restaurants'), ('dumpling house', 'dumpling houses'),
             ('wine bar', 'wine bars'), ('juice bar', 'juice bars'), ('sandwich shop', 'sandwich shops'), ('fried chicken place', 'fried chicken places')],
    'errand': [('gas station', 'gas stations'), ('pharmacy', 'pharmacies'), ('grocery store', 'grocery stores'), ('supermarket', 'supermarkets'), ('ATM', 'ATMs'),
               ('bank', 'banks'), ('post office', 'post offices'), ('laundromat', 'laundromats'), ('hardware store', 'hardware stores'), ('car wash', 'car washes'),
               ('EV charging station', 'EV charging stations'), ('phone repair shop', 'phone repair shops'), ('barber', 'barbers'), ('hair salon', 'hair salons'),
               ('dry cleaner', 'dry cleaners'), ('bike shop', 'bike shops'), ('mechanic', 'mechanics'), ('pet store', 'pet stores'), ('florist', 'florists'),
               ('convenience store', 'convenience stores'), ('electronics store', 'electronics stores'), ('shoe repair shop', 'shoe repair shops'), ('print shop', 'print shops')],
    'health': [('urgent care clinic', 'urgent care clinics'), ('dentist', 'dentists'), ('vet', 'vets'), ('hospital', 'hospitals'), ('eye doctor', 'eye doctors'),
               ('physical therapist', 'physical therapists'), ('walk-in clinic', 'walk-in clinics')],
    'leisure': [('park', 'parks'), ('museum', 'museums'), ('movie theater', 'movie theaters'), ('bowling alley', 'bowling alleys'), ('gym', 'gyms'),
                ('library', 'libraries'), ('beach', 'beaches'), ('hiking trail', 'hiking trails'), ('zoo', 'zoos'), ('mall', 'malls'), ('bookstore', 'bookstores'),
                ('playground', 'playgrounds'), ('swimming pool', 'swimming pools'), ('arcade', 'arcades'), ('escape room', 'escape rooms'), ('karaoke bar', 'karaoke bars'),
                ('art gallery', 'art galleries'), ('aquarium', 'aquariums'), ('climbing gym', 'climbing gyms'), ('skate park', 'skate parks'), ('dog park', 'dog parks'),
                ('farmers market', 'farmers markets'), ('night market', 'night markets'), ('botanical garden', 'botanical gardens'), ('golf course', 'golf courses'),
                ('yoga studio', 'yoga studios'), ('comedy club', 'comedy clubs'), ('concert venue', 'concert venues'), ('ice rink', 'ice rinks'), ('trampoline park', 'trampoline parks')],
    'travel': [('hotel', 'hotels'), ('hostel', 'hostels'), ('train station', 'train stations'), ('bus station', 'bus stations'), ('parking garage', 'parking garages'),
               ('car rental', 'car rentals'), ('subway station', 'subway stations'), ('tourist information center', 'tourist information centers'), ('campground', 'campgrounds')],
}
MODIFIERS = {
    'food': ['cheap', 'good', 'best', 'top rated', 'affordable', 'quiet', 'family-friendly', 'late night', 'vegan', 'halal', 'gluten-free', 'fancy', 'cozy', 'local'],
    'errand': ['24 hour', 'good', 'cheap', 'closest'],
    'health': ['24 hour', 'good', 'top rated'],
    'leisure': ['free', 'kid-friendly', 'good', 'best', 'quiet', 'fun', 'top rated', 'indoor'],
    'travel': ['cheap', 'good', 'best', 'pet-friendly'],
}
CITIES = ['Tokyo', 'Paris', 'New York', 'Seoul', 'London', 'Toronto', 'Sydney', 'Berlin', 'Mumbai', 'Cairo', 'Mexico City', 'Busan', 'Chicago',
          'Los Angeles', 'Madrid', 'Rome', 'Bangkok', 'Lagos', 'Lima', 'Vancouver', 'Istanbul', 'Dubai', 'Singapore', 'Hong Kong', 'Barcelona',
          'Amsterdam', 'Lisbon', 'Prague', 'Vienna', 'Buenos Aires', 'Sao Paulo', 'Cape Town', 'Nairobi', 'Jakarta', 'Manila', 'Hanoi', 'Taipei',
          'Osaka', 'Melbourne', 'Auckland', 'Dublin', 'Edinburgh', 'Montreal', 'San Francisco', 'Seattle', 'Austin', 'Miami', 'Boston', 'Denver',
          'Copenhagen', 'Stockholm', 'Oslo', 'Helsinki', 'Warsaw', 'Budapest', 'Athens', 'Marrakesh', 'Kyoto', 'Delhi', 'Bangalore', 'Kuala Lumpur',
          'Brooklyn', 'Shibuya', 'Hongdae', 'downtown Chicago', 'the Mission District', 'Soho', 'Gangnam', 'Le Marais', 'Kreuzberg']
LANDMARKS = ['Times Square', 'the Eiffel Tower', 'Central Park', 'Shibuya Station', 'the Colosseum', 'the Golden Gate Bridge', 'Big Ben', 'the Sydney Opera House',
             'the Louvre', 'Union Square', 'the CN Tower', 'Gyeongbokgung Palace', 'the Brandenburg Gate', 'the Space Needle', 'Grand Central Terminal',
             'the Burj Khalifa', 'Marina Bay Sands', 'the Sagrada Familia', 'Tokyo Tower', 'the British Museum', 'Pike Place Market', 'Navy Pier']

HELD_THINGS = {'ramen shop', 'laundromat', 'escape room', 'dog park', 'hostel', 'eye doctor', 'wine bar', 'print shop', 'ice rink'}
HELD_CITIES = {'Lisbon', 'Taipei', 'Nairobi', 'Montreal', 'Kreuzberg', 'Auckland'}
HELD_LANDMARKS = {'the Space Needle', 'Tokyo Tower', 'Navy Pier'}



NEAR = [('{xs} near me', None), ('{x} near me', None), ('any {xs} near me?', None), ('is there {a} {x} near me?', None),
        ('is there {a} {x} around here?', None), ('{xs} nearby', None), ('find me {a} {x} nearby', None), ('find {xs} near me', None),
        ("where's the nearest {x}?", None), ('where is the closest {x}', None), ('nearest {x}', None), ('closest {x} to me', None),
        ('can you recommend {a} {x} near me?', None), ('recommend {a} {x} close by', None), ('suggest some {xs} around here', None),
        ('show me {xs} near me', None), ("i'm looking for {a} {x} nearby", None), ('what are some {xs} near me', None),
        ('what {xs} are open near me right now', '{xs} open now'), ('{xs} open now near me', '{xs} open now'), ('need {a} {x} nearby', None),
        ('{xs} within walking distance', None), ("what's a good {x} near me", 'good {x}'), ('are there any {xs} close by?', None),
        ('where is {a} {x} around here', None), ('directions to the nearest {x}', None), ('any good {xs} around here?', 'good {xs}'),
        ('list {xs} near my location', None), ('{xs} close to me', None), ('i need {a} {x}, is there one close by?', None),
        ('where can i find {a} {x} near here?', None), ('help me find {a} {x} close to me', None), ('take me to the nearest {x}', None),
        ('which {x} near me is best?', 'best {x}'), ('{x} around here?', None)]
HELD_NEAR = {'where can i find {a} {x} near here?', 'help me find {a} {x} close to me', 'take me to the nearest {x}', 'which {x} near me is best?', '{x} around here?'}

NAMED = [('best {xs} in {p}', 'best {xs}'), ('{xs} in {p}', None), ('{x} in {p}', None), ('where can i find {a} {x} in {p}', None),
         ('recommend {a} {x} in {p}', None), ('any good {xs} in {p}?', 'good {xs}'), ("i'm visiting {p}, any {xs} you'd recommend?", None),
         ('top rated {xs} in {p}', 'top rated {xs}'), ('{xs} near {p}', None), ('{x} close to {p}', None), ('where is {a} {x} near {p}?', None),
         ('find {xs} around {p}', None), ("what's the best {x} in {p}", 'best {x}'), ('{xs} in {p} open now', '{xs} open now'),
         ('going to {p} tomorrow, where are good {xs}?', 'good {xs}'), ('show me {xs} in {p}', None)]
HELD_NAMED = {'find {xs} around {p}', 'going to {p} tomorrow, where are good {xs}?', 'show me {xs} in {p}'}

INTENTS_HERE = [('where should i eat?', 'restaurants'), ('where should i eat tonight', 'restaurants'), ("i'm hungry, where should i go?", 'restaurants'),
                ('where can i grab lunch around here', 'lunch'), ('where should i get dinner nearby?', 'dinner'), ('where can i get breakfast near me', 'breakfast'),
                ('where can i get a coffee around here', 'coffee'), ('i need coffee, anywhere close?', 'coffee'), ('where should i go tonight?', 'things to do tonight'),
                ('where should i go?', 'places to go'), ('what is there to do around here?', 'things to do'), ("i'm bored, where can i go?", 'fun things to do'),
                ('what fun things can i do nearby', 'fun things to do'), ('where can i take my kids around here', 'family activities'),
                ('where should we go on a date nearby?', 'date night spots'), ('where can i go for a walk near me', 'parks'), ('i need gas, where can i fill up', 'gas station'),
                ('i need to get some cash, where is an atm', 'ATM'), ('my car needs charging, where can i charge it nearby', 'EV charging station'),
                ('where can i buy groceries around here', 'grocery store'), ('where can i get a haircut near me', 'barber'), ('where can i work out near me', 'gym'),
                ('where can i watch a movie tonight around here', 'movie theater'), ('where can i get my phone fixed nearby', 'phone repair shop'),
                ('where can i get a drink around here', 'bars'), ('where can i study near me, somewhere quiet', 'quiet cafe'),
                ('my dog needs a checkup, any vets close by?', 'vet'), ('i have a toothache, is there a dentist near me', 'dentist'), ('where can i park around here', 'parking'),
                ('where can i stay tonight near here', 'hotels'), ('where can i get dessert nearby', 'dessert'), ("i'm craving sushi, where can i get some near me", 'sushi'),
                ('i want pizza, where should i go around here', 'pizza'), ('i feel like tacos tonight, any spots near me', 'tacos'), ('somewhere to eat near me?', 'restaurants'),
                ('places to eat nearby', 'places to eat'), ('what should i do this weekend around here', 'things to do this weekend'),
                ('where can i get cheap food near me', 'cheap food'), ('recommend me a place to eat nearby', 'restaurants'), ('where should i go for lunch', 'lunch'),
                ('good places to hang out near me', 'places to hang out'), ('where can i buy flowers nearby', 'florist'), ('where can i print something near me', 'print shop'),
                ('where can i do laundry around here', 'laundromat'), ('where can i get my car fixed nearby', 'mechanic'), ('where can i swim near me', 'swimming pool'),
                ('i want to go shopping, where should i go nearby', 'shopping'), ('what restaurants do you recommend near me', 'restaurants'),
                ('where should i eat near me that is not too expensive', 'affordable restaurants')]
HELD_INTENTS_HERE = {'where can i do laundry around here', 'where can i print something near me', 'good places to hang out near me', 'where can i swim near me',
                     'where should i eat near me that is not too expensive'}
INTENTS_NAMED = [('where should i eat in {p}?', 'restaurants'), ('what should i do in {p}', 'things to do'), ('where should i go in {p}?', 'places to visit'),
                 ("i'm visiting {p}, where should i go?", 'places to visit'), ('things to do in {p}', 'things to do'), ('where should i stay in {p}', 'hotels'),
                 ('what to see in {p}', 'sights'), ('fun things to do in {p} at night', 'fun things to do at night'), ('where can i get good coffee in {p}', 'good coffee'),
                 ('date ideas in {p}', 'date night spots'), ('where to eat near {p}', 'restaurants'), ('what can i do near {p} with kids', 'family activities'),
                 ('places to visit in {p}', 'places to visit'), ('where should i go shopping in {p}', 'shopping')]
HELD_INTENTS_NAMED = {'where should i go shopping in {p}', 'what can i do near {p} with kids'}

OPENERS = [''] * 12 + ['hey kani, ', 'kani ', 'can you tell me ', 'please ', 'yo ', 'hi! ', 'quick question: ', 'ok ', 'hey, ', 'kani, ']
ENDINGS = [''] * 10 + [' please', ' thanks', ' pls', ' :)', ' thx']

TYPOS = {'restaurant': ['resturant', 'restaurent', 'restraunt', 'restureat', 'resteraunt'], 'restaurants': ['resturants', 'restaurents', 'restraunts'],
         'nearby': ['near by', 'nearbye'], 'near me': ['nearme', 'neer me', 'near mee'], 'pharmacy': ['pharmacey', 'farmacy'], 'coffee': ['coffe', 'cofee'],
         'museum': ['musuem', 'museam'], 'recommend': ['recomend', 'recommned', 'reccomend'], 'nearest': ['nearst', 'neareast'], 'theater': ['theatre', 'theator'],
         'grocery': ['grocey', 'grocerry'], 'where': ['wher', 'whre'], 'station': ['staion', 'statoin'], 'around': ['aroud', 'arround'], 'closest': ['closet', 'clossest']}
STOP = re.compile(r'^(?:a|an|the|some|any)\s+', re.I)


def article(phrase):
    return 'an' if phrase[0].lower() in 'aeiou' or phrase.startswith(('EV', 'ATM')) else 'a'


def ask(rng, text):
    """A person's way of typing `text`: an opener, sometimes lowercase or a misspelling, sometimes no question mark."""
    if rng.random() < 0.25:
        words = [w for w in TYPOS if re.search(rf'\b{re.escape(w)}\b', text)]
        if words:
            word = rng.choice(words)
            text = re.sub(rf'\b{re.escape(word)}\b', rng.choice(TYPOS[word]), text, count=1)
    opener = rng.choice(OPENERS)
    text = opener + text + rng.choice(ENDINGS)
    if rng.random() < 0.3:
        text = text.rstrip('?')
    if rng.random() < 0.55 and not opener:
        text = text[0].upper() + text[1:]
    return text


def call(what, where):
    return f'PLACES: {STOP.sub("", what).strip()} @ {where}'


def row(split, family, question, reply, source):
    return {'split': split, 'family': family, 'source': source, 'accept': [reply],
            'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reply}]}


def fill(template, x, xs, p=''):
    return template.format(x=x, xs=xs, a=article(x), p=p)


def places(rng, split, count):
    """Place questions and their calls. Training uses nothing held out; each validation question has one held-out
    part (the thing, the phrasing or the place) and draws the rest from everything."""
    everything = {'things': [(kind, s, pl) for kind, pairs in THINGS.items() for s, pl in pairs], 'near': NEAR, 'named': NAMED,
                  'here': INTENTS_HERE, 'there': INTENTS_NAMED, 'places': CITIES + LANDMARKS}
    held_out = {'things': lambda t: t[1] in HELD_THINGS, 'near': lambda t: t[0] in HELD_NEAR, 'named': lambda t: t[0] in HELD_NAMED,
                'here': lambda t: t[0] in HELD_INTENTS_HERE, 'there': lambda t: t[0] in HELD_INTENTS_NAMED,
                'places': lambda p: p in HELD_CITIES or p in HELD_LANDMARKS}
    seen_parts = {key: [item for item in items if not held_out[key](item)] for key, items in everything.items()}
    held_parts = {key: [item for item in items if held_out[key](item)] for key, items in everything.items()}
    forms = {'near': 0.45, 'here': 0.15, 'named': 0.25, 'there': 0.15}
    rows, seen = [], set()
    while len(rows) < count:
        if split == 'train':
            pools, form = seen_parts, rng.choices(list(forms), list(forms.values()))[0]
        else:
            part = rng.choice(['things', 'phrasing', 'places'])
            form = rng.choice({'things': ['near', 'named'], 'phrasing': list(forms), 'places': ['named', 'there']}[part])
            pools = dict(everything)
            if part == 'phrasing':
                pools[form] = held_parts[form]
            else:
                pools[part] = held_parts[part]
        kind, x, xs = rng.choice(pools['things'])
        if form in ('near', 'named'):
            template, what = rng.choice(pools[form])
            if what is None and rng.random() < 0.35:
                modifier = rng.choice(MODIFIERS[kind])
                x, xs = f'{modifier} {x}', f'{modifier} {xs}'
            p = rng.choice(pools['places']) if form == 'named' else ''
            question, looked = fill(template, x, xs, p), fill(what or ('{xs}' if '{xs}' in template else '{x}'), x, xs)
        else:
            template, looked = rng.choice(pools[form])
            p = rng.choice(pools['places']) if form == 'there' else ''
            question = template.format(p=p)
        where = p or 'here'

        if re.search(r'\b(\w+) \1\b', question) or re.search(r'\b(\w+) \1\b', looked):
            continue
        text = ask(rng, question)
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        rows.append(row(split, 'places', text, call(looked, where), 'near' if where == 'here' else 'named'))
    return rows


CODE = [('Write a Python function that finds the nearest {x} from a list of (name, x, y) tuples and my position.',
         '```python\nimport math\n\n\ndef nearest(places, x, y):\n    return min(places, key=lambda place: math.dist((place[1], place[2]), (x, y)))\n```'),
        ('write a function that sorts {xs} by rating, best first',
         '```python\ndef by_rating(places):\n    return sorted(places, key=lambda place: place["rating"], reverse=True)\n```'),
        ('How do I compute the distance between two places from their latitude and longitude in Python?',
         '```python\nimport math\n\n\ndef distance_km(lat1, lon1, lat2, lon2):\n    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))\n'
         '    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2\n'
         '    return 6371 * 2 * math.asin(math.sqrt(a))\n```'),
        ('write python code that keeps only the {xs} that are open, from a list of dicts with an "open" key',
         '```python\ndef open_places(places):\n    return [place for place in places if place["open"]]\n```')]
CHAT = [('do you like {xs}?', "I can't visit places myself, but I can find {xs} near you. Just ask!"),
        ('can you go to {a} {x} with me?', "I can't go anywhere, but I can find {xs} near you if you'd like."),
        ('i just went to a really nice {x}', 'That sounds lovely! What did you like most about it?'),
        ('i love {xs}', 'Nice! What do you like most about them?'),
        ('my favorite {x} closed down :(', "Oh no, that's sad when a favorite place closes. Want me to look for another {x} near you?")]

NEEDS = [('i need a pep talk', "Here's something true: you're trying, and that counts. Take a deep breath, drink some water, and do one small thing next."),
         ('i need some motivation', 'You can do it! Start with one small step right now; the next one gets easier. What are you working on?'),
         ('i need advice', "Sure, I'm here to help. What's going on?"), ('i need to vent', "I'm listening. Tell me what's going on."),
         ('i need a hug', 'Sending you a big virtual hug! Want to talk about what is going on?'),
         ('i need a break', 'A break sounds like a good idea. Stand up, stretch, drink some water, and come back in a few minutes.'),
         ('i need help with my homework', "Happy to help! What's the question?"), ('i need help with my code', 'Sure! Paste the code and tell me what it should do.'),
         ('i need ideas for a project', 'Fun! You could build a to-do list app, a quiz game, or a weather dashboard. What do you like?'),
         ('i need someone to talk to', "I'm here. What's on your mind?"), ('i need a joke', 'Why do programmers prefer dark mode? Because light attracts bugs!'),
         ('i need to focus', 'Try 25 minutes of focused work with notifications off, then a 5 minute break.'),
         ('i need to calm down', 'Take a slow breath in for four seconds, hold it, and let it out for six. Want to tell me what happened?'),
         ('entertain me', "Let's fix that! We could write a tiny game, a random quote generator, or a to-do list app. Want to build one together?"),
         ("i'm bored", "Let's fix that! We could write a tiny game, a random quote generator, or a to-do list app. Want to build one together?"),
         ('i need something to do', "Let's fix that! We could write a tiny game, a random quote generator, or a to-do list app. Want to build one together?"),
         ("i've been stuck on this bug for hours", 'Debugging can be really frustrating, and every programmer gets stuck sometimes. Take a short break, then test one small piece at a time.'),
         ("i'm going through a breakup", "I'm really sorry. Breakups hurt, and it's okay to feel sad. Do you want to talk about it?")]
NEED_ENDINGS = ['', ' right now', ' today', ' lol', ' please', ' asap', ' honestly', '...']
HELD_NEEDS = {'i need to calm down', 'i need something to do'}

FACTS = ['what is {a} {x}?', 'history of {xs}', 'how many {xs} are in {p}?', 'when was {l} built?', 'how tall is {l}?', 'who designed {l}?',
         'why are {xs} popular?', 'what is the population of {c}?', 'what language do they speak in {c}?', 'how old is {l}?']


def contrasts(rng, split, count):
    """Questions that mention places but are not asking where to go: their usual replies."""
    held = split == 'validation'
    things = [(s, pl) for pairs in THINGS.values() for s, pl in pairs if (s in HELD_THINGS) == held]
    cities = [c for c in CITIES if (c in HELD_CITIES) == held]
    landmarks = [l for l in LANDMARKS if (l in HELD_LANDMARKS) == held]
    rows, seen = [], set()
    while len(rows) < count:
        x, xs = rng.choice(things)
        kind = rng.random()
        if kind < 0.55:
            question = rng.choice(FACTS).format(x=x, xs=xs, a=article(x), p=rng.choice(cities), c=rng.choice(cities), l=rng.choice(landmarks))
            text = ask(rng, question)
            reply, family = f'SEARCH: {search_query(text)}', 'places_fact'
        elif kind < 0.75:
            question, reply = rng.choice(CODE)
            text, reply, family = question.format(x=x, xs=xs), reply, 'places_code'
        else:
            question, reply = rng.choice(CHAT)
            text, family = ask(rng, question.format(x=x, xs=xs, a=article(x))), 'places_chat'
            reply = reply.format(x=x, xs=xs)
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        rows.append(row(split, family, text, reply, 'contrast'))
    return rows


def needs(rng, split, count):
    """Messages about needing something (a pep talk, a break) or being bored: chat, not places."""
    pool = [n for n in NEEDS if (n[0] in HELD_NEEDS) == (split == 'validation')]
    rows, seen = [], set()
    for _ in range(count * 20):
        if len(rows) >= count:
            break
        question, reply = rng.choice(pool)
        text = ask(rng, question + rng.choice(NEED_ENDINGS))
        if text.lower() not in seen:
            seen.add(text.lower())
            rows.append(row(split, 'places_need', text, reply, 'contrast'))
    return rows


def build(seed=2405, train=9000, validation=400, contrast_train=2500, contrast_validation=200, need_train=1500, need_validation=60):
    rng = random.Random(seed)
    rows = places(rng, 'train', train) + places(rng, 'validation', validation)
    rows += contrasts(rng, 'train', contrast_train) + contrasts(rng, 'validation', contrast_validation)
    rows += needs(rng, 'train', need_train) + needs(rng, 'validation', need_validation)
    train_questions = {r['messages'][0]['content'].lower() for r in rows if r['split'] == 'train'}
    rows = [r for r in rows if r['split'] == 'train' or r['messages'][0]['content'].lower() not in train_questions]
    rng.shuffle(rows)
    destination = ROOT / 'data/places-v2.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for r in rows:
        counts[(r['split'], r['family'])] = counts.get((r['split'], r['family']), 0) + 1
    print(destination, len(rows), dict(sorted(counts.items())))


if __name__ == '__main__':
    build()
