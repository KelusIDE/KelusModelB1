
import test from 'node:test';
import assert from 'node:assert/strict';
import { searchWords, questionNames, webAnswer } from '../../site/web.js';
test('Wikipedia search drops question words and opens names in the question as articles first', async () => {
  assert.equal(searchWords('when was Albert Einstein born'), 'Albert Einstein born');
  assert.equal(searchWords('what is the capital of japan'), 'capital japan');
  assert.deepEqual(questionNames('Who wrote The Lord of the Rings'), ['The Lord of the Rings']);
  assert.deepEqual(questionNames('what is a cat'), []);
  const asked = [];
  const fetcher = async url => {
    const params = new URL(url).searchParams; asked.push(params.get('titles') || params.get('gsrsearch'));
    const pages = params.get('titles') ? [{ ns: 0, index: 1, title: 'Albert Einstein', extract: 'Albert Einstein was a physicist.' }]
      : [{ ns: 0, index: 1, title: 'Einstein family', extract: 'The Einstein family.' }, { ns: 0, index: 2, title: 'Albert Einstein', extract: 'Albert Einstein was a physicist.' }];
    return { ok: true, text: async () => JSON.stringify({ query: { pages } }) };
  };
  const result = await webAnswer({ mode: 'search', query: 'when was Albert Einstein born' }, fetcher, undefined, { duckduckgo: false, proxy: false, key: undefined });
  assert.deepEqual(asked, ['Albert Einstein', 'Albert Einstein born']);
  assert.deepEqual(result.sources.map(s => s.title), ['Albert Einstein', 'Einstein family']);
});
