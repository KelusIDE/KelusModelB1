
import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_SYNCED_CHATS, fromRemote, mergeChats, toRemote } from '../src/sync.mjs';

const chat = (id, updated, extra = {}) => ({ id, title: `Chat ${id}`, updated, model: 'Kani Otter 4.2',
  messages: [{ id: `${id}-1`, role: 'user', content: 'hi' }, { id: `${id}-2`, role: 'assistant', content: 'hello', status: 'complete', source: 'model' }], ...extra });
const remote = (id, updated, extra = {}) => ({ id, title: `Chat ${id}`, updated, model: 'b1-v3', pending: false,
  messages: [{ role: 'user', content: 'from the web' }, { role: 'assistant', content: 'answer', source: 'model_web' }], ...extra });

test('chats are stored in the website format, within the account rules', () => {
  const stored = toRemote(chat('a', 1, {
    title: 'x'.repeat(200), model: 'm'.repeat(60),
    messages: [...Array.from({ length: 230 }, (_, i) => ({ id: String(i), role: i % 2 ? 'assistant' : 'user', content: `turn ${i}`, source: 'model' })),
      { id: 'p', role: 'assistant', content: 'Full Google results near 40.71,-74.00', saved: 'I looked up pizza near you.', source: 'places' },
      { id: 'e', role: 'assistant', content: '', status: 'failed' }],
  }));
  assert.equal(stored.title.length, 120);
  assert.equal(stored.model.length, 40);
  assert.equal(stored.messages.length, 200);
  assert.deepEqual(Object.keys(stored).sort(), ['messages', 'model', 'title']);
  assert.deepEqual(stored.messages.at(-1), { role: 'assistant', content: 'I looked up pizza near you.', source: 'places' });
  assert.deepEqual(stored.messages.at(-2), { role: 'assistant', content: 'turn 229' });
  const back = fromRemote(remote('w', 5));
  assert.equal(back.model, 'Kani Seal 3 · web');
  assert.deepEqual(back.messages.map(m => [m.role, m.content, m.status, m.source]), [['user', 'from the web', 'complete', undefined], ['assistant', 'answer', 'complete', 'model_web']]);
  assert.ok(back.messages.every(m => typeof m.id === 'string' && m.id.length === 32));
});

test('merging: newer copies win, new chats travel both ways, deletions elsewhere are followed', () => {
  const local = [chat('both-phone-newer', 50_000), chat('both-web-newer', 10_000), chat('phone-only', 40_000), chat('deleted-on-web', 30_000), chat('same', 20_000)];
  const list = [remote('both-phone-newer', 20_000), remote('both-web-newer', 60_000), remote('web-only', 70_000), remote('same', 20_500)];
  const { merged, upload, dropped } = mergeChats(local, list, new Set(['deleted-on-web', 'both-phone-newer', 'both-web-newer', 'same']));
  assert.deepEqual(merged.map(c => c.id), ['web-only', 'both-web-newer', 'both-phone-newer', 'phone-only', 'same']);
  assert.equal(merged[1].messages[0].content, 'from the web');
  assert.equal(merged.find(c => c.id === 'same').messages[0].content, 'hi');
  assert.deepEqual(upload.map(c => c.id), ['both-phone-newer', 'phone-only']);
  assert.deepEqual(dropped, ['deleted-on-web']);

  assert.equal(mergeChats([chat('a', 1_000)], [remote('a', 9_000, { pending: true })], new Set()).merged[0].messages[0].content, 'hi');
});

test('a full page of account chats leaves older ones out without deleting them on the phone', () => {
  const list = Array.from({ length: MAX_SYNCED_CHATS }, (_, i) => remote(`r${i}`, 100_000 + i));
  const { merged, dropped, upload } = mergeChats([chat('old', 5_000), chat('recent-gone', 200_000)], list, new Set(['old', 'recent-gone']));
  assert.deepEqual(dropped, ['recent-gone']);
  assert.deepEqual(upload.map(c => c.id), ['old']);
  assert.equal(merged.length, MAX_SYNCED_CHATS);
});
