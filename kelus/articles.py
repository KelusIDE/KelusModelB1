"""Collect attributed Wikipedia articles and prepare a separate raw-text corpus.

Whole pages stay in one split; existing tokenizer IDs are reused unchanged.
No generated question/answer pairs and no external model are involved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode
import numpy as np
from .corpus import digest, identity, rows
from .tokenizer import from_dict, BOS, EOS

API = 'https://en.wikipedia.org/w/api.php'
TRAIN = ['Percentage','Fraction','Ratio','Proportion (mathematics)','Arithmetic','Multiplication','Division (mathematics)',
         'Average','Probability','Geometry','Algebra','Algorithm','Data structure','Computer programming',
         'Computer science','Recursion (computer science)','Software testing','Database','World Wide Web',
         'Photosynthesis','Water cycle','Gravity','Solar System','Nutrition','Cooking','Pancake','Language','Scientific method']
VALIDATION = ['Decimal','Prime number','Binary search','Ecosystem','Atmosphere of Earth','Bread']
# Level-4 vital article lists (~10,000 core topics), sampled evenly so no one area dominates.
VITAL = ['People','History','Geography','Arts','Philosophy and religion','Everyday life','Society and social sciences',
         'Biology and health sciences','Physical sciences','Technology','Mathematics']

def request(attempts=4, **params):
    url=API+'?'+urlencode({'action':'query','format':'json','formatversion':2,'maxlag':5,**params})
    for attempt in range(attempts):
        result=subprocess.run(['/usr/bin/curl','--fail','--silent','--show-error','--proto','=https','--max-time','30',
            '--max-filesize','4000000','--user-agent','KaniResearch/0.1 (https://github.com/KelusIDE/KelusModelB1)',url],
            capture_output=True,timeout=35)
        if result.returncode==0 and b'"code":"maxlag"' not in result.stdout[:300]:break
        if attempt==attempts-1:raise ValueError('Wikipedia request failed: '+result.stderr.decode(errors='replace').strip()[:200])
        time.sleep(5*3**attempt)  # serial requests only; back off on 429, 5xx and maxlag
    if len(result.stdout)>4_000_000:raise ValueError('Article response exceeded budget')
    data=json.loads(result.stdout)
    if 'error' in data:raise ValueError(data['error']['info'])
    return data

def held_out(title):
    return int(hashlib.sha256(title.encode()).hexdigest(),16)%40==0


def vital_titles(count):
    """About `count` Level-4 vital articles, an even share from each area, in a fixed hashed order."""
    areas=[]
    for area in VITAL:
        titles,more=set(),{}
        while True:
            data=request(titles='Wikipedia:Vital articles/Level/4/'+area,redirects=1,prop='links',plnamespace=0,pllimit='max',**more)
            titles.update(link['title'] for page in data['query']['pages'] for link in page.get('links',[]))
            if 'continue' not in data:break
            more={'plcontinue':data['continue']['plcontinue']}
        areas.append(sorted(titles,key=lambda t:hashlib.sha256(t.encode()).hexdigest()))
    chosen,seen=[],set(TRAIN)|set(VALIDATION)
    for rank in range(max(map(len,areas))):
        for titles in areas:
            if rank<len(titles) and titles[rank] not in seen and len(chosen)<count:
                chosen.append(titles[rank]);seen.add(titles[rank])
    return chosen


def collect(root, tokenizer_path, replay_root=None, vital=0, replay=(80,8)):
    """With `vital`, unavailable or duplicate pages are skipped and logged rather than fatal."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Corpus already prepared; use a new directory.')
    rights=request(meta='siteinfo',siprop='rightsinfo')['query']['rightsinfo']
    if rights.get('url') not in ('https://creativecommons.org/licenses/by-sa/4.0/','https://creativecommons.org/licenses/by-sa/4.0/deed.en'):
        raise ValueError('Unexpected upstream license; inspect before preparing data: '+str(rights))
    (root/'raw').mkdir(exist_ok=True)
    docs=[];seen=set();page_ids=set();skipped=[]
    extra=vital_titles(vital) if vital else []
    plan=[('train',TRAIN),('validation',VALIDATION),('train',[t for t in extra if not held_out(t)]),('validation',[t for t in extra if held_out(t)])]
    for split,titles in plan:
        for title in titles:
            raw_path=root/'raw'/(identity(title)+'.json')
            if raw_path.exists():data=json.loads(raw_path.read_text())
            else:
                data=request(titles=title,redirects=1,prop='extracts|info|revisions',explaintext=1,exlimit=1,inprop='url',rvprop='ids|timestamp',curtimestamp=1)
                raw_path.write_text(json.dumps(data,ensure_ascii=False)+'\n')
            page=data['query']['pages'][0]
            problem=None
            if page.get('missing') or page.get('ns')!=0 or not page.get('extract'):problem='Missing article: '+title
            else:
                text=page['extract'].strip();key=identity(text);page_id=page['pageid']
                if len(text)<500 or len(text)>400000 or '\0' in text:problem='Unexpected article size: '+title
                elif key in seen or page_id in page_ids:problem='Duplicate article across requested pages: '+title
            if problem:
                if title in extra:skipped.append(problem);print(json.dumps({'skipped':problem}),flush=True);continue
                raise ValueError(problem)
            seen.add(key);page_ids.add(page_id)
            revision=page['revisions'][0]
            docs.append({'id':key,'source':'wikipedia','title':page['title'],'page_id':page_id,'revision':revision,
                'retrieved_at':data.get('curtimestamp'),'url':page['fullurl'],'revision_url':f'https://en.wikipedia.org/w/index.php?oldid={revision["revid"]}',
                'attribution':'Wikipedia contributors','license':rights,'split':split,'text':text})
            print(json.dumps({'article':page['title'],'split':split,'characters':len(text)}),flush=True)
    if replay_root:
        # Preserve source notices and original whole-document split for rehearsal.
        source=Path(replay_root)/'documents.jsonl'
        for split,limit in zip(('train','validation'),replay):
            chosen=sorted((r for r in rows(source) if r['split']==split),key=lambda r:r['id'])[:limit if limit>=0 else None]
            for row in chosen:
                if row['id'] not in seen: docs.append(row);seen.add(row['id'])
    docs.sort(key=lambda r:r['id'])
    document_path=root/'documents.jsonl'
    document_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in docs))
    (root/'tokenizer.json').write_bytes(Path(tokenizer_path).read_bytes())
    tokenizer=from_dict(json.loads((root/'tokenizer.json').read_text()))
    totals=Counter();counts=Counter();offsets={s:[] for s in ('train','validation')}
    for split in offsets:
        with (root/(split+'.bin')).open('wb') as stream:
            for row in docs:
                if row['split']!=split:continue
                ids=np.asarray([BOS]+tokenizer.encode(row['text'])+[EOS],dtype='<u2')
                offsets[split].append([totals[split],len(ids),row['id']]);stream.write(ids.tobytes())
                totals[split]+=len(ids);counts[split]+=1
        if totals[split]<4096:raise ValueError('Insufficient whole-document validation/training data')
        (root/(split+'-documents.json')).write_text(json.dumps(offsets[split])+'\n')
    sources={'wikipedia_license':rights,'article_attribution':'Wikipedia contributors; per-page URLs, revision IDs and timestamps retained in documents.jsonl',
        'replay_sources':str(replay_root) if replay_root else None,'replay_documents':list(replay) if replay_root else None,
        'vital_articles':{'requested':vital,'areas':VITAL if vital else [],'skipped':skipped},'documents_sha256':digest(document_path),
        'split':'Entire distinct articles; named validation pages never in training. Related facts can still overlap.',
        'purpose':'Raw next-token continued pretraining; not instruction tuning or proof of reasoning.'}
    (root/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    manifest={'format':'kelus-raw-corpus-v1','dtype':'<u2','vocabulary':tokenizer.vocab_size,'tokenizer_sha256':digest(root/'tokenizer.json'),
        'tokens':dict(totals),'documents':dict(counts),'sources_sha256':digest(root/'sources.json'),'documents_sha256':digest(document_path),
        'files':{s:digest(root/(s+'.bin')) for s in offsets},'tokenizer_training':'Unchanged existing Kani tokenizer; no new token IDs.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/articles-v1'))
    p.add_argument('--tokenizer',type=Path,default=Path('data/foundation/tokenizer.json'))
    p.add_argument('--replay',type=Path,default=Path('data/foundation'))
    p.add_argument('--vital',type=int,default=0,help='also collect about this many Level-4 vital articles (~1 in 40 held out whole)')
    p.add_argument('--replay-documents',type=int,nargs=2,default=[80,8],metavar=('TRAIN','VALIDATION'),help='-1 replays every document of that split')
    a=p.parse_args();print(json.dumps(collect(a.root,a.tokenizer,a.replay,a.vital,tuple(a.replay_documents)),indent=2))

if __name__=='__main__':main()
