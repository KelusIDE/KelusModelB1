import test from 'node:test';
import assert from 'node:assert/strict';
import { READING_CHARACTERS, SEARCH_CALL, groundedQuery, maybeSearch, readingPrompt, searchedAnswer, selectSentences } from '../../site/reading.js';

test('search calls are recognized and held back while they might still be one', () => {
  assert.equal('SEARCH: when was Einstein born'.match(SEARCH_CALL)[1], 'when was Einstein born');
  for (const text of ['S', 'SEAR', 'SEARCH:', 'SEARCH: x']) assert.ok(maybeSearch(text), text);
  for (const text of ['Sure', 'Hello', 'A variable is']) assert.ok(!maybeSearch(text), text);
});

test('a searched answer reads the top results and lists them as sources', async () => {
  let asked;
  const webAnswer = async request => ({ engine: 'langsearch', text: 'results', sources: [
    { title: 'Albert Einstein', url: 'https://en.wikipedia.org/wiki/Albert_Einstein', text: 'Albert Einstein was a physicist. He was born on 14 March 1879 in Ulm.' },
    { title: 'Ulm', url: 'https://en.wikipedia.org/wiki/Ulm', text: 'Ulm is a city in Germany.' }], query: request.query });
  const generate = async function* (messages) { asked = messages; yield 'Albert Einstein was born on 14 March 1879. [1]'; };
  let text = '';
  for await (const piece of searchedAnswer('when was Einstein born', 'hey kani, when was Einstein born?', { generate, webAnswer })) text += piece;
  assert.equal(asked.length, 1);
  assert.equal(asked[0].content, readingPrompt(selectSentences([['Albert Einstein', 'Albert Einstein was a physicist. He was born on 14 March 1879 in Ulm.'],
    ['Ulm', 'Ulm is a city in Germany.']], 'hey kani, when was Einstein born?', 90), 'hey kani, when was Einstein born?'));
  assert.equal(text, 'Searching the web for “when was Einstein born”…\n\nAlbert Einstein was born on 14 March 1879. [1]\n\nSources:\n'
    + '[1] Albert Einstein — https://en.wikipedia.org/wiki/Albert_Einstein\n[2] Ulm — https://en.wikipedia.org/wiki/Ulm');
});

test('dense web text is trimmed until the reading prompt fits the model', async () => {
  let asked;
  const dense = Array.from({ length: 400 }, (_, i) => `https://example.com/einstein/${i}/born-1879-ulm-physicist`).join(' ');
  const webAnswer = async () => ({ engine: 'langsearch', text: 'results', sources: ['A', 'B', 'C'].map(t => ({ title: t, url: `https://${t}.example`, text: dense })) });
  const generate = async function* (messages) { asked = messages; yield 'Born in 1879. [1]'; };
  for await (const piece of searchedAnswer('einstein', 'when was Einstein born?', { generate, webAnswer })) void piece;
  assert.ok(asked[0].content.length <= READING_CHARACTERS, `${asked[0].content.length} characters`);
  assert.match(asked[0].content, /^Search results:\n\[1\] A: https:\/\/example\.com/);

  const counted = [];
  const fits = text => { counted.push(text.length); return text.length < 300; };
  for await (const piece of searchedAnswer('einstein', 'when was Einstein born?', { generate, webAnswer, fits })) void piece;
  assert.ok(counted.length > 1 && asked[0].content.length < 300, counted.join(' '));
});

test('a garbled search query falls back to the question; a faithful one is kept', () => {
  assert.equal(groundedQuery('when was Alsecre born', 'when was Albert Einstein born?'), 'when was Albert Einstein born');
  assert.equal(groundedQuery('when was Einstein born', 'hey kani, when was Einstein born?'), 'when was Einstein born');
  assert.equal(groundedQuery('capital of Japan', 'What is the capital of Japan?'), 'capital of Japan');
});

test('without readable sources the search results are shown as they are', async () => {
  let text = '';
  const webAnswer = async () => ({ engine: 'wikipedia', text: 'No Wikipedia article text found for that query.', sources: [] });
  for await (const piece of searchedAnswer('zzz', 'zzz', { generate: () => { throw new Error('should not run'); }, webAnswer })) text += piece;
  assert.equal(text, 'Searching the web for “zzz”…\n\nNo Wikipedia article text found for that query.');
});

test('a recipe search shows the recipe as the page gives it, with its link; other searches answer as before', async () => {
  const { searchedAnswer } = await import('../../site/reading.js');
  const page = 'Fluffy pancakes in 20 minutes. IngredientsDry ingredients:1 1/2 cups all-purpose flour 3 1/2 teaspoons baking powder 1 teaspoon salt '
    + 'Wet ingredients:1 1/4 cups milk 1 large egg 3 tablespoons butter, melted MethodWhisk the dry ingredients together. Beat the egg into the milk and butter. '
    + 'Stir the wet mixture into the dry until just combined. Cook on a hot griddle until bubbles form, then flip.';
  const sources = [{ title: 'Pancake history', url: 'https://example.org/history', text: 'Pancakes are very old. People ate them in ancient Greece.' },
    { title: 'Easy pancakes', url: 'https://cook.example.com/pancakes', text: page }];
  let generated = 0;
  const generate = async function* () { generated++; yield 'Pancakes are very old. [1]'; };
  const webAnswer = async () => ({ engine: 'web', sources });
  let text = '';
  for await (const piece of searchedAnswer('pancake recipe', 'How do I make pancakes?', { generate, webAnswer })) text += piece;
  assert.equal(generated, 0);
  assert.match(text, /^Searching the web for “pancake recipe”…\n\nHere's a recipe for pancake from cook\.example\.com \[2\]:\n\nIngredients:\nDry ingredients:\n- 1 1\/2 cups all-purpose flour\n- 3 1\/2 teaspoons baking powder\n- 1 teaspoon salt\nWet ingredients:\n- 1 1\/4 cups milk\n- 1 large egg\n- 3 tablespoons butter, melted\n\nSteps:\n1\. Whisk the dry ingredients together\.\n2\. Beat the egg into the milk and butter\.\n3\. Stir the wet mixture into the dry until just combined\.\n4\. Cook on a hot griddle until bubbles form, then flip\./);
  assert.match(text, /Sources:\n\[1\] Pancake history — https:\/\/example\.org\/history\n\[2\] Easy pancakes — https:\/\/cook\.example\.com\/pancakes$/);
  text = '';
  for await (const piece of searchedAnswer('pancake recipe', 'How do I make pancakes?', { generate, webAnswer: async () => ({ engine: 'web', sources: sources.slice(0, 1) }) })) text += piece;
  assert.equal(generated, 1);
  assert.match(text, /Pancakes are very old\. \[1\]/);
});

test('a web answer that is not about the question or not from the results is not shown', async () => {
  const { searchedAnswer, backed, NO_ANSWER } = await import('../../site/reading.js');
  const results = [['Charles Sanders Peirce', 'Charles Sanders Peirce was an American scientist, mathematician, logician, and philosopher.'],
    ['Marajó', 'Marajó is a large coastal island in Pará, Brazil.']];
  assert.ok(backed('what is Marajó?', 'Marajó is a large coastal island in Pará, Brazil. [2]', results));
  assert.ok(!backed('what is Marajó?', 'Marajó is an American scientist, mathematician, logician, and philosopher. [1]', results), 'cites a page about something else');
  assert.ok(!backed('who was Peirce?', 'A prime number is a natural number greater than 1. [1]', results), 'not about the question');
  assert.ok(!backed('who was Peirce?', 'Peirce was a British nobleman and politician who founded a famous bank. [1]', results), 'not from the results');
  assert.ok(!backed('what is Marajó?', 'Marajó is a large coastal island in Pará, Brazil. [7]', results), 'cites a result that does not exist');
  assert.ok(backed('how long will it take? how long does it take to learn guitar', 'Most people learn basic guitar chords in about three months. [1]',
    [['Learning guitar', 'Most people learn basic guitar chords in about three months of practice.']]), 'a follow-up names its subject in the search');
  const sources = [{ title: 'Charles Sanders Peirce', url: 'https://example.org/peirce', text: results[0][1] }];
  const generate = async function* () { yield 'Charles Sanders Peirce was a British nobleman and banker. [1]'; };
  let text = '';
  for await (const piece of searchedAnswer('Charles Sanders Peirce', 'who was Charles Sanders Peirce?', { generate, webAnswer: async () => ({ engine: 'web', sources }) })) text += piece;
  assert.ok(text.includes(NO_ANSWER) && !text.includes('nobleman'), text);
  assert.match(text, /Sources:\n\[1\] Charles Sanders Peirce — https:\/\/example\.org\/peirce$/);
});

test('recipes are read from pages in other languages and from pages that run headings and words together', async () => {
  const { recipeAnswer } = await import('../../site/recipes.js');
  const german = 'Pfannkuchen ZutatenFür den Teig: 250 g Mehl 3 Eier 500 ml Milch 1 Prise Salz 2 EL Zucker ZubereitungMehl und Milch verrühren. '
    + 'Eier und Salz unterrühren. In der Pfanne goldbraun backen.';
  assert.equal(recipeAnswer('Pfannkuchen recipe', [{ title: 'Pfannkuchen', url: 'https://www.example.de/pfannkuchen', text: german }]),
    "Here's a recipe for Pfannkuchen from example.de [1]:\n\nIngredients:\nFür den Teig:\n- 250 g Mehl\n- 3 Eier\n- 500 ml Milch\n- 1 Prise Salz\n- 2 EL Zucker\n\n"
    + 'Steps:\n1. Mehl und Milch verrühren.\n2. Eier und Salz unterrühren.\n3. In der Pfanne goldbraun backen.');
  const dutch = 'Best cookies 36 servings 30 minutes totale tijdIngrediënten ▢ 1 cup salted butter softened ▢ 2 large eggs ▢ 3 cups all-purpose flour ▢ 1⁄2 teaspoon baking powder ▢ Pinch of salt';
  const card = recipeAnswer('cookie recipe', [{ title: 'Cookies', url: 'https://umami.example/nl/recipe/1', text: dutch }]);
  assert.match(card, /- 1 cup salted butter softened\n- 2 large eggs\n- 3 cups all-purpose flour\n- 1⁄2 teaspoon baking powder\n- Pinch of salt\n\nThe steps are on the page \[1\]\./);
  assert.equal(recipeAnswer('cookie recipe', [{ title: 'History', url: 'https://example.org', text: 'Cookies were first baked in Persia in the 7th century.' }]), null);
});

test('a lookup written without the space after the colon is still a lookup', async () => {
  const { SEARCH_CALL, maybeSearch } = await import('../../site/reading.js');
  const { PLACES_CALL, maybePlaces } = await import('../../site/places.js');
  assert.equal('SEARCH:peafowl'.match(SEARCH_CALL)[1], 'peafowl');
  assert.equal('SEARCH: peafowl'.match(SEARCH_CALL)[1], 'peafowl');
  assert.ok(maybeSearch('SEARCH:p') && maybeSearch('SEAR') && !maybeSearch('Sea otters'));
  assert.deepEqual('PLACES:pizza @ here'.match(PLACES_CALL).slice(1), ['pizza', 'here']);
  assert.ok(maybePlaces('PLACES:pi') && !maybePlaces('Places to'));
  assert.equal('SEARCH: '.match(SEARCH_CALL), null);
});

test('a searched answer goes back into the chat as the answer alone', async () => {
  const { spokenAnswer } = await import('../../site/reading.js');
  assert.equal(spokenAnswer('Searching the web for “otters”…\n\nOtters are mammals. [1]\n\nSources:\n[1] Otter — https://example.org/otter'), 'Otters are mammals. [1]');
  assert.equal(spokenAnswer('Otters are mammals.'), 'Otters are mammals.');
});
