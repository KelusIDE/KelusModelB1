"""Instruction curriculum v4: v3 with every crisis wording trained.

Run 3 (trained on v3) missed 4 of 29 validation crisis messages, all of them the
held-out wording "i don't want to live anymore". Safety wordings should not be
held out, so v4 trains every crisis phrasing plus further ones (MORE_CRISIS). Crisis
validation then checks new greetings, endings and typos on those wordings. Other
situations keep their held-out wordings.

    python3 data/make_instruct_v4.py   # writes data/instruct-v4.jsonl
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
import make_feelings  # noqa: E402
import make_instruct_v3 as v3  # noqa: E402

if __name__ == '__main__':
    rows = v3.build(feelings=make_feelings.build(train_all=('crisis',)))
    destination = ROOT / 'data/instruct-v4.jsonl'
    destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    counts = {}
    for row in rows:
        key = f"{row['family']}/{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({'path': str(destination), 'records': len(rows), 'counts': dict(sorted(counts.items()))}, indent=1))
