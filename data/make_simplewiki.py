"""Text corpus for big Kani: Simple English Wikipedia (CC BY-SA 4.0) plus the existing articles-v2 corpus.

Downloads the latest simplewiki pages-articles dump, keeps encyclopedia articles (namespace 0, not
redirects), turns wikitext into plain paragraphs (templates, tables, references, files and markup
removed; link text kept), and tokenizes with Kani's own BPE tokenizer. The output is a
kelus-raw-corpus-v1 directory for kelus.pretrain / kelus.gpu_train: articles-v2's token streams
(foundation + 2,034 Wikipedia articles) followed by the simplewiki documents. One article in 200 goes
to validation, chosen by a hash of its title.

    python3 data/make_simplewiki.py        # writes data/big-v1/
"""
import bz2
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DUMP = 'https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2'

COMMENT = re.compile(r'<!--.*?-->', re.S)
REF = re.compile(r'<ref[^>/]*/>|<ref[^>]*>.*?</ref>', re.S | re.I)
TAG = re.compile(r'</?[a-zA-Z][^>]*>')
FILE = re.compile(r'\[\[(?:File|Image|Category|Media):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]', re.I)
LINK = re.compile(r'\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]')
EXTERNAL = re.compile(r'\[https?://[^\s\]]+\s*([^\]]*)\]')
HEADING = re.compile(r'^=+\s*(.*?)\s*=+\s*$', re.M)
SKIP_SECTIONS = re.compile(r'^(References|Other websites|Related pages|Notes|Sources|Further reading|External links)$', re.I)


def strip_nested(text, open_, close):
    """Remove {{templates}} or {| tables |} including nested ones."""
    out, depth, i = [], 0, 0
    while i < len(text):
        if text.startswith(open_, i):
            depth += 1
            i += len(open_)
        elif depth and text.startswith(close, i):
            depth -= 1
            i += len(close)
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return ''.join(out)


def plain(wikitext):
    text = COMMENT.sub('', wikitext)
    text = REF.sub('', text)
    text = strip_nested(text, '{{', '}}')
    text = strip_nested(text, '{|', '|}')
    text = FILE.sub('', text)
    text = LINK.sub(r'\1', text)
    text = EXTERNAL.sub(r'\1', text)
    text = TAG.sub('', text)
    text = text.replace("'''", '').replace("''", '')
    paragraphs, skipping = [], False
    for block in re.split(r'\n\s*\n', text):
        lines = []
        for line in block.splitlines():
            heading = HEADING.match(line.strip())
            if heading:
                if lines and not skipping:
                    paragraphs.append(' '.join(lines))
                lines, skipping = [], bool(SKIP_SECTIONS.match(heading.group(1)))
            elif line.strip() and not line.lstrip().startswith(('|', '!', '{', '}', '__')):
                lines.append(re.sub(r'^[*#:;]+\s*', '', line.strip()))
        if lines and not skipping:
            paragraphs.append(' '.join(lines))
    return '\n\n'.join(p for p in paragraphs if len(p) > 20)


def articles(path):
    namespace = None
    with bz2.open(path, 'rb') as stream:
        for _, element in ET.iterparse(stream, events=('end',)):
            tag = element.tag.rsplit('}', 1)[-1]
            if tag != 'page':
                continue
            ns = element.find('{*}ns')
            redirect = element.find('{*}redirect')
            title = element.find('{*}title').text or ''
            revision = element.find('{*}revision')
            text = revision.find('{*}text').text if revision is not None else None
            if ns is not None and ns.text == '0' and redirect is None and text:
                yield title, text
            element.clear()


def encode(item):
    from kelus.tokenizer import BOS, EOS, from_dict
    global TOKENIZER
    if 'TOKENIZER' not in globals():
        TOKENIZER = from_dict(json.loads((ROOT / 'data/foundation/tokenizer.json').read_text()))
    title, wikitext = item
    body = plain(wikitext)
    if len(body) < 300:
        return None
    text = f'{title}\n\n{body}'
    split = 'validation' if int(hashlib.sha256(title.encode()).hexdigest(), 16) % 200 == 0 else 'train'
    return title, split, np.asarray([BOS] + TOKENIZER.encode(text) + [EOS], dtype='<u2')


def build(out=ROOT / 'data/big-v1'):
    from kelus.corpus import digest
    raw = ROOT / 'data/simplewiki'
    raw.mkdir(parents=True, exist_ok=True)
    dump = raw / 'simplewiki-latest-pages-articles.xml.bz2'
    if not dump.exists():
        import shutil, ssl
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        request = urllib.request.Request(DUMP, headers={'User-Agent': 'KelusKani/0.1 (https://github.com/KelusIDE/KelusModelB1)'})
        with urllib.request.urlopen(request, timeout=300, context=context) as response, open(dump.with_suffix('.part'), 'wb') as target:
            shutil.copyfileobj(response, target, 1 << 20)
        dump.with_suffix('.part').rename(dump)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'data/articles-v2'
    (out / 'tokenizer.json').write_bytes((base / 'tokenizer.json').read_bytes())
    totals, documents, titles = {}, {}, []
    streams = {split: open(out / f'{split}.bin', 'wb') for split in ('train', 'validation')}
    for split in streams:
        data = (base / f'{split}.bin').read_bytes()
        streams[split].write(data)
        totals[split], documents[split] = len(data) // 2, 0
    with Pool(8) as pool:
        for result in pool.imap_unordered(encode, articles(dump), chunksize=64):
            if result is None:
                continue
            title, split, tokens = result
            streams[split].write(tokens.tobytes())
            totals[split] += len(tokens)
            documents[split] += 1
            titles.append({'title': title, 'split': split, 'url': 'https://simple.wikipedia.org/wiki/' + title.replace(' ', '_')})
    for stream in streams.values():
        stream.close()
    (out / 'documents.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in titles))
    sources = {'simple_wikipedia': {'dump': DUMP, 'license': 'CC BY-SA 4.0', 'attribution': 'Simple English Wikipedia contributors',
                                    'articles': len(titles), 'list': 'documents.jsonl'},
               'base_corpus': str(base), 'split': 'simplewiki: 1 in 200 articles by title hash to validation'}
    (out / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    manifest = {'format': 'kelus-raw-corpus-v1', 'dtype': '<u2', 'vocabulary': 4096, 'tokenizer_sha256': digest(out / 'tokenizer.json'),
                'tokens': totals, 'simplewiki_documents': documents, 'sources_sha256': digest(out / 'sources.json'),
                'files': {s: digest(out / f'{s}.bin') for s in streams}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    print(json.dumps(build()))
