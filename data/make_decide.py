"""Deciding to search: questions that need facts get a search call instead of a guess or a decline.

The reply "SEARCH: <query>" is not shown to the person. The app runs that web search, then
asks the model again with the results (kelus.reading.prompt) so it answers from them with a
citation. Chat, feelings, coding concepts and code (the instruct-v4 replay) are answered
directly, so the model learns when not to search too.

Sources of questions, all searched as written (minus greetings like "hey kani,"):
  - every question in data/reading-v1.jsonl, asked without the passages
  - questions about the same articles in other forms ("how big is ...", "history of ...")
  - questions that need current information (weather, news, scores, prices)
  - instruct-v4 questions whose practiced reply was "I do not have enough training ..."
    (rewritten in data/make_instruct_v6.py), except gibberish, which keeps the decline
Validation questions come only from held-out articles.

    python3 data/make_decide.py      # writes data/decide-v1.jsonl
"""
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent.parent
OPENER = re.compile(r'^(?:(?:hey|hi|hello|yo|ok|okay|alright|and|so|btw|also|hmm|question:)[,!]?\s+|(?:hey )?(?:kani|kelus)[,!]?\s+'
                    r'|quick question:\s+|do you know\s+|can you tell me\s+)+', re.I)
FILLER = re.compile(r'(?:[\s,]+(?:please|pls|plz|lol|lmao|thanks|thx|:\)|:\(|:d))+[\s?.!]*$', re.I)
MORE = ['How big is {t}?', 'History of {t}', 'Give me facts about {t}.', 'Why is {t} important?', 'what is {t} known for',
        'Can you look up {t}?', 'I want to learn about {t}.', 'Where does {t} come from?', 'What is the history of {t}?']
PLACES = ['Tokyo', 'Paris', 'New York', 'Seoul', 'London', 'Toronto', 'Sydney', 'Berlin', 'Mumbai', 'Cairo', 'Mexico City', 'Busan',
          'Chicago', 'Los Angeles', 'Madrid', 'Rome', 'Bangkok', 'Lagos', 'Lima', 'Vancouver']
TOPICS = ['space exploration', 'electric cars', 'AI', 'the economy', 'climate change', 'the Olympics', 'Minecraft', 'smartphones',
          'video games', 'the stock market', 'NASA', 'football', 'K-pop', 'renewable energy', 'the election']
CURRENT = ["what's the weather in {p}", 'weather in {p} today', 'is it going to rain in {p} tomorrow', 'latest news about {topic}',
           "what's new with {topic}", '{topic} news today', 'what time is it in {p} right now', 'who won the last world cup',
           'who won the super bowl this year', "what's the price of bitcoin", 'how much is a tesla model 3', 'when is the next solar eclipse',
           'what movies are out this week', 'who is the president of {country}', 'who is the prime minister of {country}',
           "what's the population of {p} now", 'best restaurants in {p}', 'when does the next iphone come out', 'what happened today in {topic}',
           'top songs this week', 'how much does a ps5 cost', 'exchange rate dollar to euro', 'what is the tallest building in the world',
           'how far is the moon from earth', 'how many countries are in the world']
COUNTRIES = ['France', 'Japan', 'South Korea', 'Canada', 'Brazil', 'India', 'Germany', 'Kenya', 'Australia', 'Mexico']
OPENERS = [''] * 10 + ['hey kani, ', 'kani ', 'can you tell me ', 'quick question: ', 'yo ', 'hi! ']


def query(question):
    """What gets searched: the question without greetings or trailing punctuation."""
    text = FILLER.sub('', OPENER.sub('', question.strip())).rstrip(' ?.!')
    return text or question.strip()


def record(split, question, source):
    return {'split': split, 'family': 'decide', 'source': source, 'accept': [f'SEARCH: {query(question)}'],
            'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': f'SEARCH: {query(question)}'}]}


def ask(rng, text):
    opener = rng.choice(OPENERS)
    text = opener + (text[0].lower() + text[1:] if opener else text)
    return text if rng.random() > 0.2 else text.lower()


def build(seed=2034):
    rng = random.Random(seed)
    rows, seen = [], set()

    def add(split, question, source):
        if question.lower() not in seen:
            seen.add(question.lower())
            rows.append(record(split, question, source))
    for line in open(ROOT / 'data/reading-v1.jsonl', encoding='utf-8'):
        r = json.loads(line)
        add(r['split'], r['messages'][0]['content'].rsplit('Question: ', 1)[1], 'reading')
        if rng.random() < 0.15:
            add(r['split'], ask(rng, rng.choice(MORE).format(t=r['title'])), 'article')
    for _ in range(3000):
        text = rng.choice(CURRENT).format(p=rng.choice(PLACES), topic=rng.choice(TOPICS), country=rng.choice(COUNTRIES))
        add('train', ask(rng, text) + rng.choice(['', '?', '??', ' please']), 'current')
    rng.shuffle(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/decide-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row['source']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': counts}))
