"""Files the site's English tutor needs beside the voice models (2026-10-30).

    python3 -I data/make_voice_assets.py CMUDICT_DICT WHISPER_TOKENIZER_JSON

- site/voice/cmudict.txt: the CMU Pronouncing Dictionary (BSD-style licence, kept in site/voice/CMUDICT-LICENSE), one
  word per line with its first pronunciation in ARPAbet ("think TH IH1 NG K"). site/voice/pronounce.js turns it into the
  phonemes Kokoro reads and the sounds the tutor teaches.
- site/voice/whisper-tokens.json: Whisper base.en's byte-level BPE (vocab in id order and merges), from the model's
  tokenizer.json, so the tutor can score the words a learner was asked to say.

Both inputs are downloaded data: they are parsed as text and nothing in them is run.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'site/voice'
WORD = re.compile(r"^[a-z][a-z']*$")
PHONE = re.compile(r'^[A-Z]{1,2}[012]?$')


def cmudict(path):
    entries = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        if '(' in word or not WORD.match(word) or not phones or not all(PHONE.match(p) for p in phones):
            continue
        entries.setdefault(word, ' '.join(phones))
    return entries


def whisper_tokens(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    vocab = data['model']['vocab']
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    assert None not in tokens and len(tokens) == 50257, len(tokens)
    merges = [m if isinstance(m, str) else ' '.join(m) for m in data['model']['merges']]
    return {'tokens': tokens, 'merges': merges}


if __name__ == '__main__':
    dictionary, tokenizer = sys.argv[1:3]
    OUT.mkdir(parents=True, exist_ok=True)
    entries = cmudict(dictionary)
    (OUT / 'cmudict.txt').write_text(''.join(f'{w} {p}\n' for w, p in sorted(entries.items())), encoding='utf-8')
    (OUT / 'whisper-tokens.json').write_text(json.dumps(whisper_tokens(tokenizer), ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'{len(entries):,} words; wrote {OUT}/cmudict.txt and whisper-tokens.json')
