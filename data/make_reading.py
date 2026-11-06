"""Reading practice: answer a question from search results, with a citation, or say they don't answer it.

Built from the Wikipedia articles in data/articles-v2 (CC BY-SA 4.0, attribution in
documents.jsonl). Each record's user turn is kelus.reading.prompt(): one to three
numbered passages and a question. The answer copies the sentence (or date) that answers
it and cites the passage, e.g. "Vega is the brightest star in ... [2]". About one in
eight questions has no answering passage; the reply is then kelus.reading.NO_ANSWER.

Questions come from fixed patterns in the text: what something is (the lead sentence),
birth and death dates from the lead's parentheses, and sentences about population,
capitals, founding, location, makeup, invention and discovery. Validation uses only the
held-out articles (articles.held_out), so it measures reading new text, not recall.

    python3 data/make_reading.py      # writes data/reading-v1.jsonl
"""
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus.reading import NO_ANSWER, SENTENCE, prompt, select  # noqa: E402

NONASCII_PAREN = re.compile(r' ?\([^()]*[^\x00-\x7f][^()]*\)')
MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
DATE = rf'(?:\d{{1,2}} {MONTH} \d{{3,4}}|{MONTH} \d{{1,2}}, \d{{3,4}}|c\. \d{{3,4}}|\d{{3,4}}(?: BC| BCE| AD)?)'
LIFE = re.compile(rf'\((?:[^()]*?; )?(?:born )?({DATE})(?: ?[–-] ?({DATE}))?\)')
FACTS = [
    (re.compile(r'\bpopulation\b[^.]*\d'), ['What is the population of {t}?', 'How many people live in {t}?']),
    (re.compile(r'\bcapital (?:city )?(?:is|of)\b'), ['What is the capital of {t}?']),
    (re.compile(r'\b(?:founded|established) (?:in|on) [^.]*\d{3,4}'), ['When was {t} founded?', 'When was {t} established?']),
    (re.compile(r'\binvented by\b'), ['Who invented {t}?']),
    (re.compile(r'\binvented in [^.]*\d{3,4}'), ['When was {t} invented?']),
    (re.compile(r'\bdiscovered by\b'), ['Who discovered {t}?']),
    (re.compile(r'\b(?:is|are) located\b|\blies (?:in|on)\b'), ['Where is {t}?', 'Where is {t} located?']),
    (re.compile(r'\b(?:is|are) made (?:of|from)\b|\bconsists? of\b'), ['What is {t} made of?']),
]
OVERVIEW = ['What is {t}?', 'Tell me about {t}.', 'What do you know about {t}?', 'Explain {t}.', '{t}?', 'what is {t}']
PERSON = ['Who was {t}?', 'Who is {t}?', 'Tell me about {t}.', 'who was {t}']
OPENERS = [''] * 12 + ['hey kani, ', 'kani, ', 'quick question: ', 'do you know ']


def clean(text):
    text = NONASCII_PAREN.sub('', re.sub(r'\[\d+\]', '', text))
    return ' '.join(re.sub(r'\s*\(\s*[,;:]?\s*\)', '', text).split())


def sentences(text):
    return [s for s in SENTENCE.split(text) if s]


def article(row):
    """Paragraphs with headings and short lines dropped, the raw lead kept for dates."""
    raw = [p.strip() for p in row['text'].split('\n') if p.strip() and not p.strip().startswith('=')]
    paragraphs = [clean(p) for p in raw if len(p.split()) >= 12]
    return {'title': row['title'], 'raw_lead': raw[0] if raw else '', 'paragraphs': paragraphs, 'split': row['split']}


def lead(doc, limit):
    """The first sentences of the article, at most `limit` words."""
    out, used = [], 0
    for s in sentences(doc['paragraphs'][0]):
        if used + len(s.split()) > limit and out:
            break
        out.append(s)
        used += len(s.split())
    return ' '.join(out)


def window(doc, sentence):
    """Sentences around `sentence` in its paragraph, so the answer is in the passage."""
    for paragraph in doc['paragraphs'][:6]:
        parts = sentences(paragraph)
        if sentence in parts:
            i = parts.index(sentence)
            return ' '.join(parts[max(0, i - 2):i + 3])
    return sentence


def questions(doc):
    """(question template, answer text, passage) for what this article can answer."""
    title, found = doc['title'], []
    first = sentences(doc['paragraphs'][0])[0] if doc['paragraphs'] else ''
    life = LIFE.search(doc['raw_lead'][:400])
    person = bool(life) and re.search(r' (?:was|is) an? ', first)
    if 8 <= len(first.split()) <= 60:
        answer = first if len(first.split()) > 18 else ' '.join(sentences(doc['paragraphs'][0])[:2])
        found.append((PERSON if person else OVERVIEW, answer, lead(doc, 110)))
    if person:
        born, died = life.group(1), life.group(2)
        on = lambda date: ('on ' if re.match(r'\d|' + MONTH, date) and not re.fullmatch(r'\d{3,4}(?: BC| BCE| AD)?', date) else 'in ') + date
        found.append((['When was {t} born?', 'when was {t} born', 'What year was {t} born?'], f'{title} was born {on(born)}.', lead(doc, 110)))
        if died:
            found.append((['When did {t} die?', 'What year did {t} die?'], f'{title} died {on(died)}.', lead(doc, 110)))
    for pattern, templates in FACTS:
        for paragraph in doc['paragraphs'][:4]:
            hit = next((s for s in sentences(paragraph) if pattern.search(s) and 8 <= len(s.split()) <= 45), None)
            if hit:
                found.append((templates, hit, window(doc, hit)))
                break
    return found


def ask(rng, templates, title):
    opener, text = rng.choice(OPENERS), rng.choice(templates).format(t=title)
    text = opener + (text[0].lower() + text[1:] if opener else text)
    return text[0].lower() + text[1:] if rng.random() < 0.25 else text


def record(rng, doc, others, split):
    """One question about `doc`, or (one time in eight) about it with only other articles shown."""
    options = questions(doc)
    if not options:
        return None

    specific = options[1:] if len(options) > 1 and options[0][0] in (OVERVIEW, PERSON) else []
    templates, answer, passage = rng.choice(specific) if specific and rng.random() < 0.7 else rng.choice(options)
    question = ask(rng, templates, doc['title'])
    distractors = [(o['title'], lead(o, rng.choice([50, 70, 90]))) for o in rng.sample(others, rng.choice([0, 1, 1, 2]))]
    if rng.random() < 0.125:
        results = distractors or [(o['title'], lead(o, 70)) for o in rng.sample(others, 1)]
        reply, kind = NO_ANSWER, 'no_answer'
    else:
        results = distractors + [(doc['title'], passage)]
        rng.shuffle(results)
        cite = 1 + next(i for i, (t, _) in enumerate(results) if t == doc['title'])
        reply, kind = f'{answer} [{cite}]', 'answer'

    results = select(results, question, budget=rng.choice([90, 110])) if rng.random() < 0.5 else results
    shown = ' '.join(p for _, p in results)
    date = re.search(rf'{DATE}\.$', answer) if answer.startswith((doc['title'] + ' was born', doc['title'] + ' died')) else None
    if kind == 'answer' and not (date and date.group(0)[:-1] in doc['raw_lead']) and answer.rstrip('.') not in shown:
        return None
    return {'split': split, 'family': 'reading', 'kind': kind, 'title': doc['title'], 'accept': [reply],
            'messages': [{'role': 'user', 'content': prompt(results, question)}, {'role': 'assistant', 'content': reply}]}


def build(seed=2033, per_article=24, max_tokens=620):
    from kelus.tokenizer import from_dict
    tokenizer = from_dict(json.loads((ROOT / 'data/foundation/tokenizer.json').read_text()))
    rows = [json.loads(line) for line in open(ROOT / 'data/articles-v2/documents.jsonl', encoding='utf-8') if '"source": "wikipedia"' in line]
    docs = [d for d in map(article, rows) if d['paragraphs']]
    rng, out, seen = random.Random(seed), [], set()
    for split in ('train', 'validation'):
        pool = [d for d in docs if d['split'] == split]
        for doc in pool:
            others = [o for o in pool if o is not doc]
            for _ in range(per_article if split == 'train' else 8):
                row = record(rng, doc, rng.sample(others, 6), split)
                if row is None or row['messages'][0]['content'] in seen:
                    continue
                if sum(len(tokenizer.encode(m['content'])) for m in row['messages']) > max_tokens:
                    continue
                seen.add(row['messages'][0]['content'])
                out.append(row)
    rng.shuffle(out)
    return out


if __name__ == '__main__':
    rows = build()
    destination = ROOT / 'data/reading-v1.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['split']}/{row['kind']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': counts}))
