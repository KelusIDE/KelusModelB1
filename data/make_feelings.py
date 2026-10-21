"""Supportive everyday conversation for Kani: breakups, bad days, stress, good news.

Each situation has hand-written replies and many ways a person might say it, with
optional greetings, endings, missing apostrophes and small typos. About a fifth of
the phrasings per situation are held out for validation, so the validation split
measures new wordings. Validation records list every accepted reply.

Crisis messages always get the same reply pointing to a person and a crisis line,
including when they arrive mixed with another feeling or after an earlier turn.

    python3 data/make_feelings.py   # prints counts; make_instruct_v3 imports build()
"""
import json
import random
import re

CRISIS = ("I'm really sorry you're in this much pain. You deserve support right now, and I'm only a small AI, "
          "so please reach out to a person. In the US, call or text 988 for the Suicide and Crisis Lifeline. "
          "Elsewhere, call your local emergency number or a crisis line, or tell someone you trust right away.")

SITUATIONS = {
    'breakup': (
        ["I'm really sorry. Breakups hurt, and it's okay to feel sad for a while. Be gentle with yourself, talk to a friend "
         "you trust, and take it one day at a time. Do you want to talk about it, or should I try to cheer you up?",
         "That sounds really painful, and I'm sorry you're going through it. Your feelings make sense. Try to rest, eat "
         "something, and lean on people who care about you. I'm here if you want to talk."],
        ['i broke up with {someone}', '{partner} broke up with me', 'my {partner} dumped me', 'i just got dumped',
         'we broke up', "i'm going through a breakup", 'my relationship just ended', 'my {partner} left me',
         'i got broken up with', '{partner} and i split up', 'i ended things with my {partner}', 'my heart is broken',
         "i can't stop thinking about my ex", 'i miss my ex so much', 'my {partner} cheated on me and we broke up']),
    'sad': (
        ["I'm sorry you're feeling down. Bad days happen, and they do pass. Want to tell me what happened? A short walk, "
         "some water, or a message to a friend can help a little.",
         "That sounds hard, and I'm glad you told me. Be kind to yourself today. If you want, tell me what's going on."],
        ["i'm sad", 'i feel down', "i'm having a bad day", 'today was awful', 'i feel terrible', "i'm not okay",
         'everything is going wrong today', 'i feel really low', "i've been crying all day", 'i feel so sad right now',
         'my day was horrible', "i'm in a bad mood", 'i feel empty today', 'nothing is going right']),
    'lonely': (
        ["Feeling lonely is really hard, and a lot of people feel it too. Sending one small message to someone can help. "
         "Clubs, classes, or online groups about something you enjoy are good ways to meet people. I'm happy to chat "
         "in the meantime."],
        ['i feel lonely', 'i have no friends', 'nobody talks to me', 'i feel so alone', "i don't have anyone to talk to",
         'everyone ignores me', 'i eat lunch alone every day', 'i feel left out', 'nobody wants to hang out with me']),
    'stressed': (
        ["That sounds like a lot. Pick one small task and do only that for the next 25 minutes, then take a short break. "
         "Write the rest down so it's out of your head. You don't have to finish everything at once."],
        ["i'm so stressed", 'i have too much homework', "i'm overwhelmed", 'work is crushing me',
         "i have so much to do and i don't know where to start", "i'm stressed about school", 'i have four deadlines this week',
         'everything is piling up', "i can't keep up with everything", 'my boss keeps giving me more work']),
    'nervous': (
        ["Feeling nervous usually means it matters to you. Take a few slow breaths, prepare one small piece at a time, "
         "and remember you only have to do your best. You've got this."],
        ["i'm nervous about my interview", "i'm anxious", "i have a test tomorrow and i'm nervous",
         "i'm scared about my presentation", 'i have a job interview tomorrow', "i'm worried about my exam",
         'my driving test is tomorrow and i am scared', "i'm so nervous i feel sick", 'i have to give a speech and i am terrified']),
    'disappointed': (
        ["I'm sorry, that's really disappointing. One result doesn't define you. Give yourself a moment to feel it, then "
         "think about one thing to try differently next time. You can still get there."],
        ['i failed my exam', "i didn't get the job", 'i got rejected', 'i messed up', 'i failed my driving test',
         "i didn't make the team", 'my project got rejected', 'i got a bad grade', 'i lost the competition',
         'i did not get into the school i wanted']),
    'tired': (
        ["It sounds like you need some rest. If you can, take a break, drink some water, and step away from screens for "
         "a while. A good night's sleep makes most things feel more manageable."],
        ["i'm so tired", "i can't sleep", "i'm exhausted", "i haven't slept in days", 'i stayed up all night',
         'i feel burned out', 'i have no energy', "i'm worn out"]),
    'angry': (
        ["That sounds really frustrating, and it's okay to feel angry. Take a few deep breaths or a short walk before "
         "you respond to anything. When you're ready, tell me what happened."],
        ["i'm so angry", 'my friend made me mad', "i'm furious", 'my brother keeps annoying me', 'people are so rude',
         'my teacher was unfair to me', 'i want to scream', 'someone was really mean to me today']),
    'coding_frustration': (
        ["Debugging can be really frustrating, and every programmer gets stuck sometimes. Take a short break, then test "
         "one small change at a time. If you share the code and the error, I can try to help."],
        ['this bug is driving me crazy', 'i want to give up on coding', "i'm frustrated with programming",
         "i've been stuck on this bug for hours", "i'm too dumb to code", 'coding makes me want to cry',
         'i hate my code right now', "i'll never be a good programmer"]),
    'conflict': (
        ["I'm sorry, fights with people we care about hurt. Give it a little time to cool down, then talk calmly about "
         "how you felt. A simple, honest message often helps more than you'd expect."],
        ['i had a fight with my best friend', 'my friend is ignoring me', 'my parents yelled at me',
         'i argued with my mom', 'my best friend is mad at me', "my sister and i aren't talking",
         'i said something mean to my friend', 'my friends left me out']),
    'good_news': (
        ["That's awesome, congratulations! You worked for this, so take a moment to enjoy it. What's next?"],
        ['i got the job', 'i passed my exam', 'i finally fixed my bug', 'my code works', 'i got into college',
         'i won the game', 'i got an a on my test', 'i finished my first website', 'i got promoted',
         'i passed my driving test']),
    'birthday': (
        ["Happy birthday! I hope you have a great day. Do something today that makes you happy."],
        ["it's my birthday", 'today is my birthday', 'i turned 16 today', 'its my bday', 'i am a year older today']),
    'cheer_up': (
        ["You matter, and a hard day doesn't mean every day will be hard. You've gotten through tough days before, and "
         "you can get through this one too. I'm rooting for you.",
         "Here's something true: you're trying, and that counts. Take a deep breath, drink some water, and do one small "
         "thing that makes you feel good today. Better days are coming."],
        ['cheer me up', 'can you make me feel better', 'say something nice', 'i need some motivation',
         'tell me something positive', 'please make me feel better', 'can you encourage me', 'i need a pep talk',
         'give me some motivation', 'can you cheer me up']),
    'bored': (
        ["Let's fix that! We could write a tiny game, a random quote generator, or a to-do list app. Want to build one "
         "together?"],
        ["i'm bored", 'i have nothing to do', "i'm so bored right now", 'what should i do im bored', 'entertain me']),
    'better': (
        ["I'm glad you feel a little better. Take care of yourself, and come back anytime you want to talk or build something."],
        ['i feel a bit better now', 'ok i feel better', 'that actually helped', 'i feel better', "i'm feeling better now",
         'that made me feel better']),
    'crisis': (
        [CRISIS],
        ['i want to die', 'i want to kill myself', 'i want to hurt myself', "i don't want to live anymore",
         "i'm thinking about suicide", "life isn't worth living", 'i wish i was dead', "i'm going to end it all",
         'i want to disappear forever', 'nobody would care if i died', "i've been cutting myself", 'i feel suicidal',
         'i want to end my life', "i don't see a reason to keep going"]),
}

SOMEONE = ['someone', 'my girlfriend', 'my boyfriend', 'my partner', 'my gf', 'my bf']
PARTNER = ['girlfriend', 'boyfriend', 'partner', 'gf', 'bf']
OPENERS = ['', '', '', 'hi ', 'hi!! ', 'hey ', 'hey kelus, ', 'hello, ', 'ugh ', 'so ', 'um ', 'honestly ', 'kelus ',
           'kani, ', 'hi kani ', 'well ', 'guess what, ', 'idk ']
ENDINGS = ['', '', '', '.', '!', '...', ' :(', ' lol', ' and idk what to do', '??', ' :/', ' today', ' right now']


def fill(core, rng):
    return core.replace('{someone}', rng.choice(SOMEONE)).replace('{partner}', rng.choice(PARTNER))


def noisy(text, rng):
    if rng.random() < 0.3:
        text = text.replace("'", '')
    if rng.random() < 0.15 and len(text) > 8:
        i = rng.randrange(2, len(text) - 2)
        if text[i].isalpha() and text[i + 1].isalpha():
            text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if rng.random() < 0.25:
        text = text[0].upper() + text[1:]
    elif rng.random() < 0.05:
        text = text.upper()
    return text


def message(core, rng, ending=True):
    return noisy(rng.choice(OPENERS) + fill(core, rng) + (rng.choice(ENDINGS) if ending else ''), rng)


# Further crisis wordings, trained whenever crisis phrasings are not held out.
MORE_CRISIS = ["i don't want to be here anymore", "i'm going to kill myself", 'everyone would be better off without me',
               'i want to stop existing', "i can't go on living like this", "i'm planning to end my life",
               'i want to overdose', 'i keep thinking about killing myself', 'there is no point in living']


def phrasings(seed=11, train_all=()):
    """Split each situation's phrasings: about 20% held out for validation.

    Situations in `train_all` train on every phrasing (plus MORE_CRISIS for crisis);
    their validation then measures new noise on seen wordings, not new wordings.
    """
    rng = random.Random(seed)
    split = {}
    for key, (_, cores) in SITUATIONS.items():
        shuffled = cores[:]
        rng.shuffle(shuffled)
        held = max(1, len(cores) // 5)
        split[key] = {'validation': shuffled[:held], 'train': shuffled[held:]}
        if key in train_all:
            every = cores + (MORE_CRISIS if key == 'crisis' else [])
            split[key] = {'validation': every, 'train': every}
    return split


def build(seed=2029, per_situation=1100, crisis=2400, validation=40, train_all=()):
    rng = random.Random(seed)
    split = phrasings(train_all=train_all)
    rows = []
    for name, part in (('train', 'train'), ('validation', 'validation')):
        for key, (replies, _) in SITUATIONS.items():
            count = (crisis if key == 'crisis' else per_situation) if name == 'train' else validation
            for _ in range(count):
                rows.append({'split': name, 'family': 'feelings', 'situation': key, 'accept': replies, 'messages': [
                    {'role': 'user', 'content': message(rng.choice(split[key][part]), rng)},
                    {'role': 'assistant', 'content': rng.choice(replies)}]})
        # Conversations: a feeling, then a follow-up such as wanting cheering up, feeling better, or a crisis.
        for _ in range(4000 if name == 'train' else 100):
            first = rng.choice([k for k in SITUATIONS if k not in ('crisis', 'better', 'cheer_up', 'good_news', 'birthday', 'bored')])
            second = rng.choice(['cheer_up', 'cheer_up', 'better', 'better', 'crisis', 'sad'])
            turns = []
            for key in (first, second):
                replies = SITUATIONS[key][0]
                turns += [{'role': 'user', 'content': message(rng.choice(split[key][part]), rng)},
                          {'role': 'assistant', 'content': rng.choice(replies)}]
            rows.append({'split': name, 'family': 'feelings', 'situation': second, 'accept': SITUATIONS[second][0], 'messages': turns})
        # A crisis mixed into another feeling in one message still gets the crisis reply.
        for _ in range(800 if name == 'train' else 40):
            other = rng.choice(['breakup', 'sad', 'lonely', 'disappointed', 'conflict', 'stressed'])
            text = fill(rng.choice(split[other][part]), rng) + rng.choice([' and ', '. ', ', ', ' and now ']) + fill(rng.choice(split['crisis'][part]), rng)
            rows.append({'split': name, 'family': 'feelings', 'situation': 'crisis', 'accept': [CRISIS], 'messages': [
                {'role': 'user', 'content': noisy(rng.choice(OPENERS) + text, rng)}, {'role': 'assistant', 'content': CRISIS}]})
    return rows


if __name__ == '__main__':
    rows = build()
    counts = {}
    for row in rows:
        counts[f"{row['split']}/{row['situation']}"] = counts.get(f"{row['split']}/{row['situation']}", 0) + 1
    print(json.dumps(counts, indent=1))
    for row in random.Random(1).sample(rows, 8):
        print(re.sub(r'\s+', ' ', row['messages'][-2]['content'])[:90], '->', row['messages'][-1]['content'][:50])
