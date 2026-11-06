"""paint-v4b: paint-v4 with the fruit and anime training records repeated, for a focused second round.

Painter v4 (4000 steps at batch 2 from v3) mixed up single fruits ("a banana" drew a watermelon slice),
because the new subjects were a small share of only 8,000 sampled examples. v4b shows each fruit and
anime record three times; validation is unchanged.

    python3 data/make_paint_boost.py   # writes data/paint-v4b.jsonl
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(line) for line in open(ROOT / 'data/paint-v4.jsonl', encoding='utf-8')]
new = [r for r in rows if r['split'] == 'train' and (r['scene'].get('still_life') or r['scene'].get('anime'))]
rows += new * 2
random.Random(2041).shuffle(rows)
(ROOT / 'data/paint-v4b.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
print(json.dumps({'records': len(rows), 'repeated': len(new)}))
