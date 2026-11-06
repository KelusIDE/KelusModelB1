"""Pinned web/SaaS code curriculum for our own Cide checkpoint.

Run as `python3 -m data.make_cide_saas`. Downloads source, never weights.
No downloaded code is executed. Keeps attribution and document-level splits.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import tarfile

import numpy as np
from kelus.corpus import digest, identity, rows, split_for
from kelus.model import Model
from kelus.tokenizer import BOS, EOS, conversation

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    dict(name='nextjs', repo='nextjs/saas-starter', revision='6e33e58b1e553a41fe22e6b941a7229a002de361',
         sha256='64c88ceb9e90d1f8621378325925073e314c75455744b873a10ee0523e1b3e18', license='MIT'),
    dict(name='fastapi', repo='fastapi/full-stack-fastapi-template', revision='b1125d8839482cb7bbae603ddc6d0081bf87532e',
         sha256='95ec61b80d51c427c96c5004a987f6240edceddefdd8e5b70b1e5750d4393a94', license='MIT'),
]
LANGUAGES = {'.py':'python', '.js':'javascript', '.mjs':'javascript', '.ts':'typescript',
             '.tsx':'tsx', '.jsx':'jsx', '.html':'html', '.css':'css', '.sql':'sql', '.prisma':'prisma'}
MDN_PREFIXES = ('files/en-us/web/javascript/', 'files/en-us/web/html/', 'files/en-us/web/css/',
                'files/en-us/web/http/', 'files/en-us/learn_web_development/')


def accepted(path):
    parts = Path(path).parts
    return (not any(p.startswith('.') or p in ('node_modules', 'dist', 'generated', 'client', 'coverage') for p in parts)
            and Path(path).suffix in LANGUAGES and not path.endswith(('.d.ts', '.min.js', '.min.css')))


def collect(root, foundation):
    documents = []; seen = set(); sources = []
    (root/'downloads').mkdir(parents=True, exist_ok=True)
    (root/'licenses').mkdir(exist_ok=True)
    def add(row):
        key = identity(row['text'])
        if key in seen or len(row['text']) < 80: return
        seen.add(key); documents.append({**row, 'id':key})
    for source in SOURCES:
        archive = root/'downloads'/(source['name']+'.tar.gz')
        if not archive.exists():
            temp = archive.with_suffix('.partial')
            try:
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '120',
                                '--max-filesize', '16777216', f"https://codeload.github.com/{source['repo']}/tar.gz/{source['revision']}",
                                '-o', str(temp)], check=True)
                if digest(temp) != source['sha256']: raise ValueError('Archive fingerprint mismatch')
                temp.replace(archive)
            finally: temp.unlink(missing_ok=True)
        if digest(archive) != source['sha256']: raise ValueError('Archive fingerprint mismatch: '+source['name'])
        licenses = []
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile() or member.size > 256000: continue
                path = member.name.partition('/')[2]
                if not path or '..' in Path(path).parts: continue
                if Path(path).name.upper().startswith(('LICENSE', 'NOTICE')):
                    target = source['name']+'-'+path.replace('/', '_')
                    (root/'licenses'/target).write_bytes(tar.extractfile(member).read()); licenses.append(target)
                if not accepted(path): continue
                text = tar.extractfile(member).read().decode('utf-8').replace('\r\n', '\n')
                add(dict(source=source['name'], path=path, text=text, split=split_for(identity(text)),
                         license=source['license'], url=f"https://github.com/{source['repo']}/blob/{source['revision']}/{path}"))
        if not licenses: raise ValueError('Missing source license')
        sources.append({**source, 'license_files':licenses})
    manifest = json.loads((foundation/'sources.json').read_text())
    if digest(foundation/'documents.jsonl') != manifest['documents_sha256']: raise ValueError('Foundation documents changed')
    for source in manifest['sources']:
        if source['name'] != 'mdn': continue
        sources.append(source)
        for name in source['licenses']: shutil.copy2(foundation/'licenses'/name, root/'licenses'/name)
    for row in rows(foundation/'documents.jsonl'):
        if row['source'] == 'mdn' and row['path'].startswith(MDN_PREFIXES): add(row)

    documents.sort(key=lambda r:(r['split'] != 'validation', r['id']))
    (root/'documents.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in documents))
    (root/'sources.json').write_text(json.dumps({'sources':sources, 'documents_sha256':digest(root/'documents.jsonl'),
        'counts':dict(Counter(r['source']+':'+r['split'] for r in documents)),
        'split':'Whole normalized document hash, modulo 32. MDN retains its original foundation split.',
        'limitations':'Related snippets may cross splits; MDN validation may have been seen by parent checkpoints. Loss is not a SaaS capability benchmark.'}, indent=2)+'\n')
    return documents


def example(question, answer, split, family, **extra):
    return dict(split=split, family=family, messages=[dict(role='user', content=question), dict(role='assistant', content=answer)], **extra)


def saas_tasks():
    """Small contract-driven examples, not imaginary whole-app solutions.

    Variations teach identifiers/contracts. They are explicitly counted as
    template variations, not independent applications or expert solutions.
    """
    for split, entities in [('train', ['task','note','project','ticket','article','invoice','contact','board','comment','event','bookmark','document']),
                            ('validation', ['reminder','workspace'])]:
        for entity in entities:
            table = entity+'s'
            for variant in range(12):
                name = f'{entity}{variant}'
                fn = f'list_{name}'
                yield example(f'Write a Python SQLite function {fn}(db, user_id) that lists id and title from {table} for the authenticated owner_id, newest id first. Bind parameters. Return the rows.',
                    f'```python\ndef {fn}(db, user_id):\n    return db.execute("SELECT id, title FROM {table} WHERE owner_id = ? ORDER BY id DESC", (user_id,)).fetchall()\n```', split, 'saas_owned_read')
                fn = f'delete_{name}'
                yield example(f'Write Python {fn}(db, item_id, user_id). Delete from {table} only when both id and owner_id match. Use parameterized SQLite, commit, and return whether exactly one row was deleted.',
                    f'```python\ndef {fn}(db, item_id, user_id):\n    cursor = db.execute("DELETE FROM {table} WHERE id = ? AND owner_id = ?", (item_id, user_id))\n    db.commit()\n    return cursor.rowcount == 1\n```', split, 'saas_owned_delete')
                fn = f'create_{name}'
                yield example(f'Write Python {fn}(db, user_id, title) to insert owner_id and title into SQLite {table}. Reject non-string, empty, or over-120-character trimmed titles with ValueError. Bind values, commit and return lastrowid.',
                    f'```python\ndef {fn}(db, user_id, title):\n    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:\n        raise ValueError("Invalid title")\n    cursor = db.execute("INSERT INTO {table} (owner_id, title) VALUES (?, ?)", (user_id, title.strip()))\n    db.commit()\n    return cursor.lastrowid\n```', split, 'saas_owned_create')
                fn = f'update_{name}'
                yield example(f'Write Python {fn}(db, item_id, user_id, title). Validate a trimmed string title of length 1 to 120, raising ValueError otherwise. Update {table} by id AND owner_id, using bound SQLite values. Commit and return rowcount == 1.',
                    f'```python\ndef {fn}(db, item_id, user_id, title):\n    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 120:\n        raise ValueError("Invalid title")\n    cursor = db.execute("UPDATE {table} SET title = ? WHERE id = ? AND owner_id = ?", (title.strip(), item_id, user_id))\n    db.commit()\n    return cursor.rowcount == 1\n```', split, 'saas_owned_update')
                fn = f'summary_{name}'
                yield example(f'Write Python {fn}(db, user_id), returning a dashboard count for rows in {table} owned by this user. Use parameterized SQLite.',
                    f'```python\ndef {fn}(db, user_id):\n    return db.execute("SELECT COUNT(*) FROM {table} WHERE owner_id = ?", (user_id,)).fetchone()[0]\n```', split, 'saas_dashboard')
                fn = f'validate_{name}'
                yield example(f'Write JavaScript function {fn}(body). Return null unless body is a non-null object with a string title whose trimmed length is 1 to 120. Otherwise return only {{title: trimmedTitle}}. Ignore other fields.',
                    f'```javascript\nfunction {fn}(body) {{\n  if (!body || typeof body !== "object" || typeof body.title !== "string") return null;\n  const title = body.title.trim();\n  if (title.length < 1 || title.length > 120) return null;\n  return {{ title }};\n}}\n```', split, 'saas_validation')
                fn = f'render_{name}'
                yield example(f'Write JavaScript {fn}(container, items). Clear the container, create an li for each item title using textContent, and append it. Treat titles as text.',
                    f'```javascript\nfunction {fn}(container, items) {{\n  container.replaceChildren();\n  for (const item of items) {{\n    const li = document.createElement("li");\n    li.textContent = item.title;\n    container.appendChild(li);\n  }}\n}}\n```', split, 'saas_dom')
    for split, names in [('train', ['current_user','resolve_user','read_session','authenticated_user']), ('validation', ['session_principal'])]:
        for name in names:
            yield example(f'Write Python {name}(db, token, now) for SQLite sessions(token_hash,user_id,expires_at). Return None for missing token or expired session. Hash the bearer token with SHA-256 before lookup. Bind values and return user_id when expires_at > now.',
                f'```python\nimport hashlib\n\ndef {name}(db, token, now):\n    if not token:\n        return None\n    token_hash = hashlib.sha256(token.encode()).hexdigest()\n    row = db.execute("SELECT user_id FROM sessions WHERE token_hash = ? AND expires_at > ?", (token_hash, now)).fetchone()\n    return row[0] if row else None\n```', split, 'saas_session')


def chunks(text, tokenizer, limit=650):
    block = []; size = 0
    for line in text.splitlines(keepends=True):
        n = len(tokenizer.encode(line))
        if n > limit:
            if block: yield ''.join(block)
            block = []; size = 0
            continue
        if size+n > limit and block:
            yield ''.join(block); block = []; size = 0
        block.append(line); size += n
    if block: yield ''.join(block)


def build_instructions(root, documents, tokenizer):
    candidates = []; seen = {}; rng = random.Random(2207)
    for doc in documents:
        if doc['source'] == 'mdn':
            snippets = [(m[1].split()[0].lower(), m[2]) for m in re.finditer(r'^```([\w-]+)[^\n]*\n(.*?)^```', doc['text'], re.M|re.S)
                        if m[1].lower() in ('js','javascript','ts','typescript','python','html','css','sql','jsx','tsx')]
        else: snippets = [(LANGUAGES[Path(doc['path']).suffix], doc['text'])]
        for language, snippet in snippets:
            for index, block in enumerate(chunks(snippet, tokenizer)):
                key = identity(block)

                if key in seen or len(tokenizer.encode(block)) < 32: continue
                seen[key] = doc['split']
                lines = block.splitlines(keepends=True)
                if len(lines) < 3: continue
                cut = max(1, len(lines)//3)
                prefix, suffix = ''.join(lines[:cut]), ''.join(lines[cut:])
                prompt = (f'Continue this {language} code excerpt from {doc["path"]}. '
                          'Return only the next part in a code fence; do not repeat the prefix.\n'+f'```{language}\n{prefix}```')
                row = example(prompt, f'```{language}\n{suffix.rstrip()}\n```', doc['split'], 'web_code_continuation',
                              source=doc['source'], document=doc['id'], url=doc['url'], license=doc['license'], block=index)
                if len(conversation(row['messages'], tokenizer=tokenizer)[0]) <= 1000: candidates.append(row)
    candidates.extend(saas_tasks())
    replay = list(rows(ROOT/'data/cide-v1.jsonl'))
    candidates.extend(rng.sample([r for r in replay if r['split']=='train'], 3000))
    candidates.extend(rng.sample([r for r in replay if r['split']=='validation'], 16))
    unique = {}
    for row in candidates:
        key = identity(json.dumps(row['messages'], sort_keys=True))
        if key not in unique or row['split']=='validation': unique[key] = row
    candidates = list(unique.values())
    train = [r for r in candidates if r['split']=='train']; validation = []

    for family in sorted(set(r['family'] for r in candidates)):
        pool = [r for r in candidates if r['split']=='validation' and r['family']==family]
        validation.extend(rng.sample(pool, min(8 if family!='web_code_continuation' else 32, len(pool))))
    rng.shuffle(train); rng.shuffle(validation)
    output = root/'instructions.jsonl'
    output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in train+validation))
    (root/'instructions-report.json').write_text(json.dumps({
        'sha256':digest(output), 'examples':dict(Counter(r['split'] for r in train+validation)),
        'families':dict(Counter(r['family']+':'+r['split'] for r in train+validation)),
        'limitations':'Code continuations plus 8 small SaaS contract templates with identifier variations and v1 replay; not thousands of unique complete apps. Validation uses related template families and is not an independent capability test.'}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'data/cide-saas-v2')
    p.add_argument('--parent', type=Path, default=ROOT/'checkpoints/cide-v1-release.npz')
    a = p.parse_args(); root = a.root
    if (root/'manifest.json').exists(): raise FileExistsError('Prepared corpus exists; preserve it or select a new root.')
    model, _, _ = Model.load(a.parent); tokenizer = model.tokenizer; del model
    documents = collect(root, ROOT/'data/foundation')
    (root/'tokenizer.json').write_text(json.dumps(tokenizer.to_dict())+'\n')
    counts = Counter(); tokens = Counter(); offsets = {s:[] for s in ('train','validation')}
    handles = {s:(root/(s+'.bin')).open('wb') for s in offsets}
    try:
        for i, doc in enumerate(documents):

            repeats = 8 if doc['source']!='mdn' and doc['split']=='train' else 1
            encoded = np.asarray([BOS]+tokenizer.encode(doc['text'])+[EOS], dtype='<u2')
            split = doc['split']
            for _ in range(repeats):
                offsets[split].append([tokens[split], len(encoded), doc['id']])
                handles[split].write(encoded.tobytes()); tokens[split] += len(encoded)
            counts[split] += 1
            if i%500==0: print(json.dumps({'stage':'encoding', 'documents':i, 'tokens':dict(tokens)}), flush=True)
    finally:
        for f in handles.values(): f.close()
    for split, values in offsets.items(): (root/(split+'-documents.json')).write_text(json.dumps(values)+'\n')
    build_instructions(root, documents, tokenizer)
    manifest = dict(format='kelus-raw-corpus-v1', dtype='<u2', vocabulary=tokenizer.vocab_size,
        tokenizer_sha256=digest(root/'tokenizer.json'), tokens=dict(tokens), documents=dict(counts),
        documents_sha256=digest(root/'documents.jsonl'), sources_sha256=digest(root/'sources.json'),
        files={s:digest(root/(s+'.bin')) for s in handles}, parent_sha256=digest(a.parent),
        tokenizer_training='Unchanged own Cide tokenizer, no external weights or vocabulary.',
        weighting='Next.js and FastAPI training documents repeated 8x; MDN and all validation documents 1x.',
        packing='BOS/document/EOS, random 1024-token next-token windows.')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)
    print((root/'instructions-report.json').read_text(), flush=True)


if __name__ == '__main__': main()
