

import test from 'node:test';
import assert from 'node:assert/strict';
import { penalize, pickToken } from '../../../site/b1.js';
import { MODELS, Stopper, createServer, parseRequest, seeded } from '../server.mjs';

const SECRET = 'test-secret';
const server = createServer({ secret: SECRET, log: () => {}, fetcher: async () => { throw new Error('no network in tests'); } });
await new Promise(resolve => server.listen(0, resolve));
test.after(() => server.close());
const base = `http://127.0.0.1:${server.address().port}`;
const post = (body, { secret = SECRET } = {}) => fetch(base + '/v1/chat/completions', { method: 'POST',
  headers: { 'Content-Type': 'application/json', ...(secret && { Authorization: `Bearer ${secret}` }) }, body: JSON.stringify(body) });

test('Cide gets room for complete code and respects explicit output limits', () => {
  const request = { model: 'cide-2', messages: [{ role: 'user', content: 'Write a page.' }] };
  assert.equal(parseRequest(request).maxTokens, 2048);
  assert.equal(parseRequest({ ...request, max_tokens: 128 }).maxTokens, 128);
  assert.equal(parseRequest({ ...request, model: 'kani-seal-3' }).maxTokens, 512);
});
const ask = (content, extra = {}) => post({ model: 'kani-seal-3', messages: [{ role: 'user', content }], ...extra });

test('only the gateway, which sends the secret, gets answers', async () => {
  for (const secret of [null, 'wrong']) {
    const response = await post({ model: 'kani-seal-3', messages: [{ role: 'user', content: 'hi' }] }, { secret });
    assert.equal(response.status, 401);
    assert.equal((await response.json()).error.code, 'unauthorized');
  }
  assert.equal((await fetch(base + '/health')).status, 200);
});

test('a chat completion has the OpenAI shape, token counts and the source', async () => {
  const response = await ask('hello', { max_tokens: 24 });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.match(body.id, /^chatcmpl-/);
  assert.equal(body.object, 'chat.completion');
  assert.equal(body.model, 'kani-seal-3');
  assert.equal(body.choices[0].message.role, 'assistant');
  assert.ok(body.choices[0].message.content.length > 0);
  assert.ok(['stop', 'length'].includes(body.choices[0].finish_reason));
  assert.ok(body.usage.prompt_tokens > 0 && body.usage.completion_tokens > 0 && body.usage.completion_tokens <= 24);
  assert.equal(body.usage.total_tokens, body.usage.prompt_tokens + body.usage.completion_tokens);
  assert.equal(body.kani.source, 'model');
});

test('greedy replies repeat exactly', async () => {
  const first = await (await ask('what is a variable?', { max_tokens: 30 })).json();
  const second = await (await ask('what is a variable?', { max_tokens: 30 })).json();
  assert.equal(first.choices[0].message.content, second.choices[0].message.content);
});

test('streaming sends chunks, the finish reason with usage, then [DONE]', async () => {
  const response = await ask('tell me about dogs', { max_tokens: 20, stream: true });
  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type'), /text\/event-stream/);
  const events = (await response.text()).split('\n\n').filter(Boolean).map(line => line.replace(/^data: /, ''));
  assert.equal(events.at(-1), '[DONE]');
  const chunks = events.slice(0, -1).map(JSON.parse);
  assert.deepEqual(chunks[0].choices[0].delta, { role: 'assistant', content: '' });
  assert.equal(chunks[0].kani.source, 'model');
  assert.ok(chunks.every(c => c.object === 'chat.completion.chunk'));
  const last = chunks.at(-1);
  assert.ok(['stop', 'length'].includes(last.choices[0].finish_reason));
  assert.ok(last.usage.completion_tokens > 0);
  const text = chunks.map(c => c.choices[0].delta.content || '').join('');
  const whole = await (await ask('tell me about dogs', { max_tokens: 20 })).json();
  assert.equal(text, whole.choices[0].message.content);
});

test('arithmetic goes to the calculator, labeled', async () => {
  const body = await (await ask('what is 12 * 12?')).json();
  assert.equal(body.kani.source, 'local_calculator');
  assert.match(body.choices[0].message.content, /144/);
});

test('a stop string ends the reply before it', async () => {
  const whole = (await (await ask('tell me about dogs', { max_tokens: 40 })).json()).choices[0].message.content;
  const word = whole.split(' ').find((w, i) => i > 1 && w.length > 2);
  const body = await (await ask('tell me about dogs', { max_tokens: 40, stop: [' ' + word] })).json();
  assert.equal(body.choices[0].message.content, whole.slice(0, whole.indexOf(' ' + word)));
  assert.equal(body.choices[0].finish_reason, 'stop');
});

test('requests Kani cannot take are refused with a reason', async () => {
  const cases = [
    [{ model: 'gpt-4', messages: [{ role: 'user', content: 'hi' }] }, 404, /Unknown model/],
    [{ messages: [] }, 400, /messages/],
    [{ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'Hello!' }] }, 400, /last message/],
    [{ messages: [{ role: 'user', content: 'x'.repeat(8001) }] }, 400, /at most 8,000/],
    [{ messages: [{ role: 'user', content: 'hi' }], max_tokens: 5000 }, 400, /max_tokens/],
    [{ messages: [{ role: 'user', content: 'hi' }], temperature: 3 }, 400, /temperature/],
    [{ messages: [{ role: 'user', content: 'hi' }], tools: [] }, 400, /Function calling/],
    [{ messages: [{ role: 'user', content: 'hi' }], top_p: 0 }, 400, /"top_p" must be a number above 0 and at most 1/],
    [{ messages: [{ role: 'user', content: 'hi' }], top_k: 2.5 }, 400, /"top_k" must be a whole number from 1 to 200/],
    [{ messages: [{ role: 'user', content: 'hi' }], presence_penalty: 3 }, 400, /presence_penalty/],
    [{ messages: [{ role: 'user', content: 'hi' }], frequency_penalty: '1' }, 400, /frequency_penalty/],
    [{ messages: [{ role: 'user', content: 'hi' }], repetition_penalty: 0.5 }, 400, /repetition_penalty/],
    [{ messages: [{ role: 'user', content: 'hi' }], seed: 1.5 }, 400, /seed/],
    [{ messages: [{ role: 'user', content: 'hi' }], response_format: { type: 'json_object' } }, 400, /plain text only/],
    [{ messages: [{ role: 'user', content: 'hi' }], logprobs: true }, 400, /logprobs/],
    [{ messages: [{ role: 'user', content: 'hi' }], logit_bias: { 5: 10 } }, 400, /logit_bias/],
    [{ messages: [{ role: 'user', content: 'hi' }], stream_options: { include_usage: 'yes' } }, 400, /stream_options/],
    [{ messages: [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'x' } }] }] }, 400, /text only/],
  ];
  for (const [body, status, message] of cases) {
    const response = await post(body);
    assert.equal(response.status, status, JSON.stringify(body).slice(0, 80));
    assert.match((await response.json()).error.message, message);
  }
  const bad = await fetch(base + '/v1/chat/completions', { method: 'POST', headers: { Authorization: `Bearer ${SECRET}` }, body: '{' });
  assert.equal(bad.status, 400);
});

test('system messages are dropped, text parts are joined, and the default model is the first listed', () => {
  const parsed = parseRequest({ messages: [{ role: 'system', content: 'You are helpful.' }, { role: 'user', content: [{ type: 'text', text: 'hi' }, { type: 'text', text: 'there' }] }] });
  assert.equal(parsed.model.id, MODELS[0].id);
  assert.deepEqual(parsed.messages, [{ role: 'user', content: 'hi\nthere' }]);
  assert.equal(parsed.maxTokens, 512);
  assert.deepEqual(parsed.sampling, { temperature: 0, topP: 1, topK: 30, presencePenalty: 0, frequencyPenalty: 0, repetitionPenalty: 1 });
  assert.equal(parsed.seed, undefined);
  assert.equal(parsed.usage, false);
  assert.equal(parsed.tools, true);
  assert.equal(parseRequest({ messages: [{ role: 'user', content: 'a' }], kani_tools: false }).tools, false);
});

test('the stopper holds back a partial stop string and drops everything from a full one', () => {
  const stopper = new Stopper(['END']);
  assert.equal(stopper.push('abc E'), 'abc ');
  assert.equal(stopper.push('N'), '');
  assert.equal(stopper.push('Dx'), '');
  assert.equal(stopper.end(), '');
  const open = new Stopper(['END']);
  assert.equal(open.push('E'), '');
  assert.equal(open.push('x'), 'Ex');
  assert.equal(open.end(), '');
});

test('OpenAI fields Kani has no use for are accepted', () => {
  const parsed = parseRequest({ messages: [{ role: 'user', content: 'hi' }], user: 'customer-42', logprobs: false, logit_bias: {},
    response_format: { type: 'text' }, stream_options: { include_usage: true }, seed: null, top_p: null });
  assert.equal(parsed.usage, true);
  assert.equal(parsed.seed, undefined);
  assert.equal(parsed.sampling.topP, 1);
});

test('a seed makes sampled replies repeat, and different seeds vary them', async () => {
  const sample = async seed => (await (await ask('tell me a story', { max_tokens: 30, temperature: 1.2, seed })).json()).choices[0].message.content;
  const first = await sample(7);
  assert.equal(await sample(7), first);
  const others = new Set();
  for (const seed of [1, 2, 3, 4]) others.add(await sample(seed));
  assert.ok([...others].some(text => text !== first));
  assert.equal(seeded(7)(), seeded(7)());
});

test('penalties change a greedy reply that repeats words', async () => {

  const plain = (await (await ask('tell me about dogs', { max_tokens: 60 })).json()).choices[0].message.content;
  const penalized = (await (await ask('tell me about dogs', { max_tokens: 60, frequency_penalty: 2, presence_penalty: 2, repetition_penalty: 2 })).json()).choices[0].message.content;
  assert.notEqual(penalized, plain);
});

test('stream_options.include_usage sends usage in its own chunk with no choices', async () => {
  const response = await ask('hello', { max_tokens: 12, stream: true, stream_options: { include_usage: true } });
  const events = (await response.text()).split('\n\n').filter(Boolean).map(line => line.replace(/^data: /, ''));
  assert.equal(events.at(-1), '[DONE]');
  const [finish, usage] = events.slice(-3, -1).map(JSON.parse);
  assert.ok(['stop', 'length'].includes(finish.choices[0].finish_reason));
  assert.equal(finish.usage, undefined);
  assert.deepEqual(usage.choices, []);
  assert.ok(usage.usage.completion_tokens > 0);
});

test('top_p keeps the fewest likeliest tokens that hold that share; penalties lower used tokens', () => {

  const logits = () => {
    const values = new Float32Array(261).fill(-Infinity);
    values.set([0.6, 0.3, 0.08, 0.02].map(Math.log), 4);
    return values;
  };
  const picks = new Set();
  for (let i = 0; i < 200; i++) picks.add(pickToken(logits(), 1, 30, Math.random, 0.85));
  assert.deepEqual([...picks].sort(), [4, 5]);
  assert.equal(pickToken(logits(), 1, 30, () => 0.999, 0.5), 4);
  const used = Float32Array.from([2, -2, 1]);
  penalize(used, new Map([[0, 3], [1, 1]]), { presencePenalty: 0.5, frequencyPenalty: 0.1, repetitionPenalty: 2 });
  assert.deepEqual([...used].map(x => Math.round(x * 100) / 100), [0.2, -4.6, 1]);
});
