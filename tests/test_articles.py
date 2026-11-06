import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from kelus.articles import collect
from kelus.pretrain import Corpus
from kelus.tokenizer import ByteTokenizer

class ArticleTests(unittest.TestCase):
    def test_whole_article_split_provenance_and_existing_token_ids(self):
        def response(**params):
            if 'meta' in params:return {'query':{'rightsinfo':{'url':'https://creativecommons.org/licenses/by-sa/4.0/deed.en','text':'CC BY-SA 4.0'}}}
            title=params['titles'];page_id=1 if title=='Training' else 2
            return {'curtimestamp':'2026-10-20T00:00:00Z','query':{'pages':[{'pageid':page_id,'ns':0,'title':title,
                'extract':(title+' article text. ')*400,'fullurl':'https://en.wikipedia.org/wiki/'+title,
                'revisions':[{'revid':page_id*10,'timestamp':'2026-10-19T00:00:00Z'}]}]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=root/'tokenizer.json';tokenizer.write_text(json.dumps(ByteTokenizer().to_dict()))
            with patch('kelus.articles.TRAIN',['Training']),patch('kelus.articles.VALIDATION',['Validation']),patch('kelus.articles.request',side_effect=response),contextlib.redirect_stdout(io.StringIO()):
                collect(root/'articles',tokenizer)
            corpus=Corpus(root/'articles')
            self.assertEqual(corpus.tokenizer.to_dict(),ByteTokenizer().to_dict())
            records=[json.loads(line) for line in (root/'articles/documents.jsonl').read_text().splitlines()]
            self.assertEqual({r['title']:r['split'] for r in records},{'Training':'train','Validation':'validation'})
            self.assertTrue(all(r['revision_url'].startswith('https://en.wikipedia.org/w/index.php?oldid=') for r in records))
            self.assertTrue(all(r['attribution']=='Wikipedia contributors' for r in records))
            with self.assertRaisesRegex(ValueError,'already prepared'):collect(root/'articles',tokenizer)

    def test_vital_articles_split_whole_pages_and_skip_unavailable_ones(self):
        from kelus.articles import held_out
        titles=[f'Topic {i}' for i in range(120)]
        def response(**params):
            if 'meta' in params:return {'query':{'rightsinfo':{'url':'https://creativecommons.org/licenses/by-sa/4.0/','text':'CC BY-SA 4.0'}}}
            if params.get('prop')=='links':
                area=params['titles'].rsplit('/',1)[1]
                return {'query':{'pages':[{'links':[{'title':t} for t in titles if (titles.index(t)%2==0)==(area=='A')]}]}}
            title=params['titles']
            if title=='Topic 7':return {'query':{'pages':[{'title':title,'missing':True}]}}
            page_id=1000+titles.index(title) if title in titles else len(title)
            return {'curtimestamp':'2026-10-20T00:00:00Z','query':{'pages':[{'pageid':page_id,'ns':0,'title':title,
                'extract':(title+' article text. ')*60,'fullurl':'https://en.wikipedia.org/wiki/'+title,
                'revisions':[{'revid':page_id*10,'timestamp':'2026-10-19T00:00:00Z'}]}]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=root/'tokenizer.json';tokenizer.write_text(json.dumps(ByteTokenizer().to_dict()))
            with patch('kelus.articles.TRAIN',['Training']),patch('kelus.articles.VALIDATION',['Validation']),patch('kelus.articles.VITAL',['A','B']),\
                 patch('kelus.articles.request',side_effect=response),contextlib.redirect_stdout(io.StringIO()):
                collect(root/'articles',tokenizer,vital=120)
            records={r['title']:r for r in map(json.loads,(root/'articles/documents.jsonl').read_text().splitlines())}
            sources=json.loads((root/'articles/sources.json').read_text())
        vital=[t for t in records if t.startswith('Topic')]
        self.assertEqual(len(vital),119)
        self.assertEqual(sources['vital_articles']['skipped'],['Missing article: Topic 7'])
        self.assertTrue(all(records[t]['split']==('validation' if held_out(t) else 'train') for t in vital))
