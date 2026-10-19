"""Prepare pinned public documents for next-token pretraining, without dataset libraries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time
import numpy as np
from .tokenizer import BPETokenizer, BOS, EOS

SOURCES = [
    {'name':'cpython','repo':'python/cpython','revision':'036d309cf542986a39eeba5ca805b2ec7619731d',
     'license':'Python license stack; PSF-2.0 and retained notices', 'license_files':['LICENSE','Doc/license.rst']},
    {'name':'mdn','repo':'mdn/content','revision':'a1aa79c09717761fc72c20bac94bb9c430ed487a',
     'license':'CC-BY-SA-2.5 prose; code licensing as described in LICENSE.md', 'license_files':['LICENSE.md']},
]
ARCHIVE_HASHES = {'cpython':'df87274f76d3dc95e2c80222b645ff6dfa3eecb65e80e2945f760f063d48753e',
                  'mdn':'682efd6d0ba293ceb8e084631d14c811d1bf682b222e0dfb3358692d7839be2f'}

def download(root):
    (root/'downloads').mkdir(parents=True,exist_ok=True)
    for source in SOURCES:
        target=root/'downloads'/(source['name']+'.tar.gz')
        if not target.exists():
            temp=target.with_suffix('.partial')
            try:
                subprocess.run(['curl','--fail','--location','--silent','--show-error','--max-time','180',
                    '--max-filesize','268435456',f"https://codeload.github.com/{source['repo']}/tar.gz/{source['revision']}",'-o',str(temp)],check=True)
                if digest(temp)!=ARCHIVE_HASHES[source['name']]:raise ValueError('Downloaded archive fingerprint mismatch')
                temp.replace(target)
            finally:temp.unlink(missing_ok=True)
        if digest(target)!=ARCHIVE_HASHES[source['name']]:raise ValueError('Cached archive fingerprint mismatch')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def identity(text):
    return hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode()).hexdigest()

def split_for(key):
    # Every copy of the same normalized document gets the same split.
    return 'validation' if int(key[:8],16)%32==0 else 'train'

def accepted(source,path):
    if source=='cpython':
        return (path.startswith('Lib/') and path.endswith('.py') and not path.startswith(('Lib/test/','Lib/idlelib/','Lib/tkinter/'))) or (path.startswith('Doc/') and path.endswith('.rst'))
    return path.startswith(('files/en-us/web/','files/en-us/learn_web_development/')) and path.endswith('/index.md')

def collect(root):
    docs=root/'documents.jsonl'; receipts=[];seen=set();counts=Counter();total=0
    if docs.exists():raise ValueError('Corpus already exists; choose a new directory to preserve provenance.')
    temporary=docs.with_suffix('.partial');(root/'licenses').mkdir(exist_ok=True,parents=True)
    try:
        with temporary.open('w',encoding='utf-8') as output:
            for source in SOURCES:
                archive=root/'downloads'/(source['name']+'.tar.gz')
                if digest(archive)!=ARCHIVE_HASHES[source['name']]:
                    raise ValueError('Archive fingerprint mismatch: '+source['name'])
                licenses=[]
                with tarfile.open(archive,'r:gz') as tar:
                    for member in tar:
                        if not member.isfile() or member.size>512000:continue
                        path=member.name.partition('/')[2]
                        if not path or '..' in Path(path).parts:continue
                        if path in source['license_files']:
                            name=source['name']+'-'+path.replace('/','_')
                            (root/'licenses'/name).write_bytes(tar.extractfile(member).read());licenses.append(name)
                        if not accepted(source['name'],path):continue
                        raw=tar.extractfile(member).read()
                        try:text=raw.decode('utf-8').replace('\r\n','\n')
                        except UnicodeDecodeError:continue
                        if len(text)<128 or '\0' in text:continue
                        key=identity(text)
                        if key in seen:continue
                        seen.add(key);split=split_for(key)
                        total+=len(raw)
                        if total>200_000_000:raise ValueError('Corpus exceeds 200 MB extraction budget')
                        record={'id':key,'source':source['name'],'path':path,'split':split,
                                'url':f"https://github.com/{source['repo']}/blob/{source['revision']}/{path}",
                                'license':source['license'],'text':text}
                        output.write(json.dumps(record,ensure_ascii=False)+'\n');counts[source['name']+':'+split]+=1
                if not licenses:raise ValueError('Missing upstream license for '+source['name'])
                receipts.append({**source,'archive_sha256':digest(archive),'licenses':licenses})
        temporary.replace(docs)
    except BaseException:
        temporary.unlink(missing_ok=True);raise
    manifest={'sources':receipts,'documents':dict(counts),'utf8_bytes':total,'documents_sha256':digest(docs),
              'split':'Normalized whole-document SHA256 modulo 32; exact normalized duplicates removed before split.',
              'limitations':'Related documents and near-duplicate snippets may cross splits. Validation loss is not an independent coding benchmark.'}
    (root/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)

def prepare(root,vocabulary=4096,sample_chars=500000):
    if (root/'manifest.json').exists():raise ValueError('Prepared dataset already exists; choose another directory.')
    source=json.loads((root/'sources.json').read_text());docs=root/'documents.jsonl'
    if digest(docs)!=source['documents_sha256']:raise ValueError('Document fingerprint changed')
    started=time.monotonic()
    candidates=sorted((r for r in rows(docs) if r['split']=='train'),key=lambda r:r['id'])
    samples=[];per_source=Counter();budget=sample_chars//len(SOURCES)
    for r in candidates:
        remaining=budget-per_source[r['source']]
        if remaining<=0:continue
        n=min(1500,remaining,len(r['text']));start=int(r['id'][8:16],16)%max(1,len(r['text'])-n+1)
        samples.append(r['text'][start:start+n]);per_source[r['source']]+=n
    del candidates
    print(json.dumps({'stage':'tokenizer','sample_chars':dict(per_source),'target_vocabulary':vocabulary}),flush=True)
    tokenizer=BPETokenizer.train(samples,vocabulary,version=2)
    (root/'tokenizer.json').write_text(json.dumps(tokenizer.to_dict())+'\n')
    totals=Counter();doc_counts=Counter();offsets={s:[] for s in ('train','validation')}
    handles={s:(root/(s+'.bin.partial')).open('wb') for s in offsets}
    try:
        for r in rows(docs):
            tokens=np.asarray([BOS]+tokenizer.encode(r['text'])+[EOS],dtype='<u2')
            split=r['split'];offsets[split].append([totals[split],len(tokens),r['id']])
            handles[split].write(tokens.tobytes());totals[split]+=len(tokens);doc_counts[split]+=1
            if sum(doc_counts.values())%1000==0:print(json.dumps({'stage':'encoding','documents':sum(doc_counts.values()),'tokens':dict(totals)}),flush=True)
    finally:
        for handle in handles.values():handle.close()
    for split in handles:
        if totals[split]<4096:raise ValueError('Not enough tokens in '+split)
        (root/(split+'.bin.partial')).replace(root/(split+'.bin'))
        (root/(split+'-documents.json')).write_text(json.dumps(offsets[split])+'\n')
    manifest={'format':'kelus-raw-corpus-v1','dtype':'<u2','vocabulary':tokenizer.vocab_size,'tokenizer_sha256':digest(root/'tokenizer.json'),
              'tokens':dict(totals),'documents':dict(doc_counts),'sources_sha256':digest(root/'sources.json'),
              'documents_sha256':source['documents_sha256'],'files':{s:digest(root/(s+'.bin')) for s in handles},
              'tokenizer_training':'Training documents only, deterministic balanced source sample; no external vocabulary.',
              'packing':'BOS/document/EOS concatenation within each split. Random contiguous windows, all next-token positions supervised.',
              'elapsed_seconds':round(time.monotonic()-started,1)}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/foundation'));p.add_argument('--vocabulary',type=int,default=4096);p.add_argument('--tokenizer-chars',type=int,default=500000);p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download:download(a.root)
    if not (a.root/'sources.json').exists():print(json.dumps(collect(a.root)),flush=True)
    prepare(a.root,a.vocabulary,a.tokenizer_chars)

if __name__=='__main__':main()
