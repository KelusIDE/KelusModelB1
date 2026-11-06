"""Common English words for kelus.paint.fix_spelling, so real words are never "corrected".

fix_spelling changes a word that is one typo away from exactly one drawable subject. Without
this list, ordinary words near a subject were changed too: "pink hair" became "pink chair" and
"three strawberries" became "tree strawberries". Words of 4+ letters that appear at least 5
times in Kani's own training text count as real; one-off typos ("bnana") do not.

    python3 data/make_words.py   # writes kelus/english_words.txt
"""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ['data/reading-v1.jsonl', 'data/instruct-v7.jsonl', 'data/paint-v4.jsonl']

counts = Counter()
for source in SOURCES:
    with open(ROOT / source, encoding='utf-8') as lines:
        for line in lines:
            counts.update(re.findall(r'[a-z]+', line.lower()))
words = sorted(w for w, k in counts.items() if k >= 5 and len(w) >= 4)
(ROOT / 'kelus/english_words.txt').write_text('\n'.join(words) + '\n', encoding='utf-8')
print(json.dumps({'words': len(words), 'sources': SOURCES}))
