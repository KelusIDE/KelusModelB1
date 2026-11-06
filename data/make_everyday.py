"""Everyday skills: learning a language, learning something new, and cooking.

Human-written sources, each kept for what it teaches well:
  - Dolly 15k (databricks, CC BY-SA 3.0): how-tos, explanations, brainstorming, writing help, and questions answered
    from a given passage. Its open trivia questions are left out: Kani searches the web for facts instead of
    answering them from memory.
  - OpenAssistant 2 (Apache-2.0): multi-turn conversations, the best-ranked reply at each turn, in English, Spanish,
    German, French, Portuguese, Chinese, Japanese and Korean. Conversations naming another assistant are left out.
  - Aya (Cohere, Apache-2.0): questions and answers written by native speakers of the same languages.
  - Tatoeba (CC BY 2.0 FR, via manythings.org/anki): English sentences with translations, as phrasebook practice ("How do you say
    ... in Korean?", "What does ... mean?", a few phrases to learn, a quiz).
  - Recipes and learning go to the web: "How do I make pancakes?" becomes "SEARCH: pancakes recipe" (dish names from
    the Wikibooks Cookbook, CC BY-SA) and the app shows the recipe as a page gives it (site/recipes.js); "teach me
    about optics" becomes "SEARCH: optics" (topics from Kani's reading data) and Kani answers from the results.
    Talk about food that asks for nothing ("I love pancakes!") stays a chat reply.
Every example must fit the model's context whole with `--tokenizer` (long ones would teach cut-off replies).
Validation holds out about 3% of every family.

    python3 data/make_everyday.py                                  # data/everyday-v1.jsonl, for Kani's English tokenizer
    python3 data/make_everyday.py --tokenizer kaggle/kani_tokenizer_multilingual.json --out data/everyday-v1-multilingual.jsonl
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kelus.tokenizer import conversation, from_dict  # noqa: E402

SOURCES = ROOT / 'data/everyday-sources'
LANGUAGES = {'en': 'English', 'ko': 'Korean', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese', 'ja': 'Japanese', 'zh': 'Chinese'}
OASST_LANGUAGES = {'en': 'en', 'es': 'es', 'de': 'de', 'fr': 'fr', 'pt-BR': 'pt', 'zh': 'zh', 'ja': 'ja', 'ko': 'ko'}
AYA_LANGUAGES = {'English': 'en', 'Korean': 'ko', 'German': 'de', 'French': 'fr', 'Spanish': 'es', 'Portuguese': 'pt', 'Japanese': 'ja',
                 'Simplified Chinese': 'zh', 'Traditional Chinese': 'zh'}
TATOEBA = {'ko': 'kor', 'de': 'deu', 'fr': 'fra', 'es': 'spa', 'pt': 'por', 'ja': 'jpn', 'zh': 'cmn'}
PAIRS_PER_LANGUAGE = 6000
OTHER_ASSISTANTS = re.compile(r'open ?assistant|LAION|chatgpt|openai|as an ai language model', re.I)


CONTEXT = ssl.create_default_context()
try:
    import certifi
    CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass


def split_of(key, share=0.03):
    """A stable split: the same prompt is always held out, whatever else changes."""
    return 'validation' if int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % 1000 < share * 1000 else 'train'


def row(family, messages, language='en', source=''):
    return {'split': split_of(family + messages[0]['content']), 'family': family, 'language': language, 'source': source,
            'messages': [{'role': m['role'], 'content': m['content'].strip()} for m in messages]}


def dolly():
    rows = []
    for line in open(SOURCES / 'databricks-dolly-15k.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['category'] == 'open_qa' or not r['response'].strip() or 'http' in r['response']:
            continue
        question = r['instruction'].strip() + (f"\n\n{r['context'].strip()}" if r['context'].strip() else '')
        rows.append(row('everyday_dolly', [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': r['response']}],
                        source=f"dolly/{r['category']}"))
    return rows


def oasst():
    """One path per conversation tree: the best-ranked reply each time, up to six messages."""
    rows = []
    for line in gzip.open(SOURCES / '2023-11-05_oasst2_ready.trees.jsonl.gz', 'rt', encoding='utf-8'):
        tree = json.loads(line)
        node, messages = tree['prompt'], []
        if node['lang'] not in OASST_LANGUAGES:
            continue
        while node and len(messages) < 6:
            if node.get('deleted') or node.get('synthetic'):
                break
            messages.append({'role': 'user' if node['role'] == 'prompter' else 'assistant', 'content': node['text']})
            replies = [r for r in node.get('replies', []) if not r.get('deleted')]
            if node['role'] == 'prompter':
                ranked = [r for r in replies if r.get('rank') is not None]
                node = min(ranked, key=lambda r: r['rank']) if ranked else None
            else:
                node = next((r for r in replies if r.get('replies')), replies[0] if replies else None)
        if messages and messages[-1]['role'] == 'user':
            messages.pop()
        if len(messages) < 2 or any(OTHER_ASSISTANTS.search(m['content']) for m in messages):
            continue
        rows.append(row('everyday_oasst', messages, OASST_LANGUAGES[tree['prompt']['lang']], 'oasst2'))
    return rows


def aya():
    import pyarrow.parquet as pq
    table = pq.read_table(SOURCES / 'data/train-00000-of-00001.parquet', columns=['inputs', 'targets', 'language'])
    rows = []
    for question, answer, language in zip(*(table.column(c).to_pylist() for c in ('inputs', 'targets', 'language'))):
        if language in AYA_LANGUAGES and question.strip() and answer.strip() and not OTHER_ASSISTANTS.search(answer):
            rows.append(row('everyday_aya', [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}],
                            AYA_LANGUAGES[language], f'aya/{language}'))
    return rows


def tatoeba_pairs(code, rng):
    """Up to PAIRS_PER_LANGUAGE (English, translation) pairs of Tatoeba sentences for one language, one per English sentence."""
    path = SOURCES / 'tatoeba' / f'{TATOEBA[code]}.txt'
    if not path.exists():
        import io
        import zipfile
        request = urllib.request.Request(f'https://www.manythings.org/anki/{TATOEBA[code]}-eng.zip', headers={'User-Agent': 'Mozilla/5.0 (KaniDataBuilder)'})
        archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(request, timeout=60, context=CONTEXT).read()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(f'{TATOEBA[code]}.txt'))
    pairs = {}
    for line in open(path, encoding='utf-8'):
        fields = line.rstrip('\n').split('\t')
        if len(fields) >= 2 and 1 <= len(fields[0].split()) <= 12 and fields[1].strip():
            pairs.setdefault(fields[0].strip(), fields[1].strip())
    pairs = list(pairs.items())
    rng.shuffle(pairs)
    return pairs[:PAIRS_PER_LANGUAGE]


ASK_TO = ['How do you say "{en}" in {lang}?', 'how do i say "{en}" in {lang}', 'Translate to {lang}: {en}', 'What is "{en}" in {lang}?',
          'Can you translate "{en}" into {lang}?', '{lang} for "{en}"?']
ASK_FROM = ['What does "{x}" mean?', 'What does "{x}" mean in English?', 'Translate to English: {x}', 'what is the meaning of "{x}"']
TEACH = ['Teach me some {lang} phrases', 'I want to learn {lang}. Give me a few phrases to start.', 'Can you help me learn {lang}?',
         'Give me some easy {lang} sentences to practice', "I'm learning {lang}, what should I practice today?"]
QUIZ = ['Quiz me on {lang}', 'Give me a {lang} quiz question', 'Test my {lang}']


def phrasebook(rng):
    rows = []
    for code, lang in LANGUAGES.items():
        if code == 'en':
            continue
        pairs = tatoeba_pairs(code, rng)
        for english, other in pairs:
            if rng.random() < 0.6:
                question = rng.choice(ASK_TO).format(en=english, lang=lang)
                answer = f'In {lang}: {other}'
            else:
                question = rng.choice(ASK_FROM).format(x=other)
                answer = f'It means "{english}" ({lang}).'
            rows.append(row('everyday_phrases', [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}], code, 'tatoeba'))
        short = [p for p in pairs if len(p[0].split()) <= 6]
        for i in range(0, min(len(short), 1500) - 4, 5):
            group = short[i:i + 5]
            lines = '\n'.join(f'{n}. {other} — {english}' for n, (english, other) in enumerate(group, 1))
            rows.append(row('everyday_phrases', [{'role': 'user', 'content': rng.choice(TEACH).format(lang=lang)},
                                                 {'role': 'assistant', 'content': f'Here are a few {lang} phrases to practice:\n\n{lines}\n\n'
                                                                                  'Say each one out loud, then try to use one today.'}], code, 'tatoeba'))
            english, other = group[0]
            rows.append(row('everyday_phrases', [{'role': 'user', 'content': rng.choice(QUIZ).format(lang=lang)},
                                                 {'role': 'assistant', 'content': f'How do you say "{english}" in {lang}?\n\n'
                                                                                  f'(Answer: {other})'}], code, 'tatoeba'))
    return rows


def wikibooks(**params):
    url = 'https://en.wikibooks.org/w/api.php?' + urllib.parse.urlencode({'format': 'json', 'formatversion': '2', **params})
    request = urllib.request.Request(url, headers={'User-Agent': 'KaniDataBuilder/1.0 (training data for the Kani assistant)'})
    error = None
    for attempt in range(7):
        try:
            return json.load(urllib.request.urlopen(request, timeout=60, context=CONTEXT))
        except OSError as failure:
            error = failure
            wait = getattr(failure, 'headers', None) and failure.headers.get('Retry-After')
            time.sleep(int(wait) if wait and wait.isdigit() else min(60, 2 ** attempt * 2))
    raise RuntimeError(f'Wikibooks did not answer ({error}): {url[:120]}')


TEMPLATE = re.compile(r'\{\{[^{}]*\}\}')
LINK = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]')


def plain(wikitext):
    """Wiki markup as plain text: no templates, files, references or formatting; links as their words."""
    text = re.sub(r'<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->', '', wikitext, flags=re.S)
    while TEMPLATE.search(text):
        text = TEMPLATE.sub('', text)
    text = re.sub(r'\[\[(?:File|Image|Category):[^\]]*(?:\[\[[^\]]*\]\][^\]]*)*\]\]', '', text, flags=re.I)
    text = LINK.sub(lambda m: m.group(1).replace('Cookbook:', ''), text)
    text = re.sub(r'\[https?://\S+\s*([^\]]*)\]', r'\1', text)
    text = re.sub(r"'{2,}", '', re.sub(r'<[^>]+>', '', text))
    return text


def recipe_pages():
    """Title and plain text of every page in Category:Recipes (cached in data/everyday-sources)."""
    cache = SOURCES / 'wikibooks-recipes.jsonl'
    if cache.exists():
        return [json.loads(line) for line in open(cache, encoding='utf-8')]
    titles, more = [], {}
    while True:
        data = wikibooks(action='query', list='categorymembers', cmtitle='Category:Recipes', cmlimit='500', cmtype='page', **more)
        titles += [m['title'] for m in data['query']['categorymembers']]
        if 'continue' not in data:
            break
        more = data['continue']
    pages = []
    for i in range(0, len(titles), 50):
        data = wikibooks(action='query', prop='revisions', rvprop='content', rvslots='main', titles='|'.join(titles[i:i + 50]))
        for page in data['query']['pages']:
            if page.get('revisions'):
                pages.append({'title': page['title'], 'text': plain(page['revisions'][0]['slots']['main']['content'])})
        time.sleep(0.5)
    with open(cache, 'w', encoding='utf-8') as out:
        for page in pages:
            out.write(json.dumps(page, ensure_ascii=False) + '\n')
    return pages


SECTION = re.compile(r'^==+\s*(.+?)\s*==+\s*$', re.M)
ASK_RECIPE = ['How do I make {dish}?', 'recipe for {dish}', 'Can you give me a recipe for {dish}?', 'how to cook {dish}', 'How do you make {dish}?',
              'I want to cook {dish} tonight. How do I make it?', 'teach me to make {dish}', 'what do I need to make {dish}?',
              'can you find me a good {dish} recipe', 'how is {dish} made?', 'easy {dish} recipe please', 'steps to make {dish}']
ASK_LEARN = ['teach me about {topic}', 'I want to learn about {topic}', 'Can you explain {topic} to me?', 'help me understand {topic}',
             'explain {topic} simply', "I'm curious about {topic}. What should I know?", 'Tell me the basics of {topic}.',
             'what is {topic} and why does it matter?', 'can you teach me something about {topic}?', 'give me a quick lesson on {topic}']
FOOD_CHAT = [('I love {dish}!', "Yum, that sounds delicious! If you ever want to make {dish}, I can look up a recipe for you."),
             ('I just made {dish} for dinner', 'Nice! How did it turn out?'),
             ('my mom makes the best {dish}', "That's lovely. Family recipes are the best kind. Do you know how she makes it?"),
             ('do you like {dish}?', "I can't eat, but {dish} sounds great. Want me to find a recipe?")]


def sections(text):
    parts, found = SECTION.split(text), {}
    for name, body in zip(parts[1::2], parts[2::2]):
        found[name.lower()] = [line.strip(' -*•#:\t') for line in body.splitlines() if line.strip(' -*•#:\t')]
    return found


def dish_names():
    """Dish names from the Wikibooks Cookbook's recipe pages: "Cookbook:Steak au Poivre (Pepper Steak)" -> "steak au poivre"."""
    names = []
    for page in recipe_pages():
        name = re.sub(r'\s*\(.*?\)\s*', ' ', page['title'].split(':', 1)[-1]).strip(" '")
        if 2 <= len(name) <= 40 and not re.search(r'[0-9]|recipe', name, re.I):
            names.append(name if name.isupper() else name.lower())
    return list(dict.fromkeys(names))


def recipe_searches(rng):
    """Recipe questions become a recipe search: the app shows the recipe a page gives (site/recipes.js)."""
    rows = []
    for dish in dish_names():
        question = rng.choice(ASK_RECIPE).format(dish=dish)
        rows.append(row('everyday_recipe_search', [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': f'SEARCH: {dish} recipe'}],
                        'en', 'wikibooks titles'))
        if rng.random() < 0.08:
            asked, reply = rng.choice(FOOD_CHAT)
            rows.append(row('everyday_food_chat', [{'role': 'user', 'content': asked.format(dish=dish)},
                                                   {'role': 'assistant', 'content': reply.format(dish=dish)}], 'en', 'contrast'))
    return rows


def learn_searches(rng):
    """Learning questions become a search for the topic; Kani answers from the results, with sources."""
    titles = {}
    for line in open(ROOT / 'data/reading-v1.jsonl', encoding='utf-8'):
        for title in re.findall(r'^\[\d+\] ([^:\n]{2,60}):', json.loads(line)['messages'][0]['content'], re.M):
            titles[re.sub(r'\s*\(.*?\)', '', title).strip()] = True
    rows = []
    for topic in titles:
        question = rng.choice(ASK_LEARN).format(topic=topic if topic.isupper() else topic.lower())
        rows.append(row('everyday_learn_search', [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': f'SEARCH: {topic}'}],
                        'en', 'reading titles'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'data/big-v1/tokenizer.json')
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/everyday-v1.jsonl')
    args = parser.parse_args()
    rng = random.Random(2410)
    tokenizer = from_dict(json.loads(args.tokenizer.read_text()))
    rows = dolly() + oasst() + aya() + phrasebook(rng) + recipe_searches(rng) + learn_searches(rng)
    kept, dropped = [], {}
    for r in rows:
        tokens, _ = conversation(r['messages'], tokenizer=tokenizer)
        if len(tokens) <= args.context - 8:
            kept.append(r)
        else:
            dropped[r['family']] = dropped.get(r['family'], 0) + 1
    rng.shuffle(kept)
    args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    counts = {}
    for r in kept:
        key = f"{r['split']}/{r['family']}/{r['language']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'records': len(kept), 'dropped_too_long': dropped, 'counts': dict(sorted(counts.items()))}, indent=1))


if __name__ == '__main__':
    main()
