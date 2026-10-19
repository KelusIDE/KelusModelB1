// Account end-to-end test for site/: two Chrome tabs against the Firebase emulators.
// Run with tests/run_site_e2e.sh, which serves site/ on :8791 and starts the emulators.
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const profile = mkdtempSync(join(tmpdir(), 'kelus-b1-e2e-'));
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--disable-gpu', '--remote-debugging-port=9334', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ready = false; for (let i = 0; i < 50 && !ready; i++) { await sleep(200); try { await fetch('http://127.0.0.1:9334/json/version'); ready = true; } catch {} }
const problems = [];
async function openTab(name) {
  const target = await (await fetch('http://127.0.0.1:9334/json/new?about:blank', { method: 'PUT' })).json();
  const ws = new WebSocket(target.webSocketDebuggerUrl); await new Promise(r => ws.addEventListener('open', r));
  let id = 0; const pending = new Map();
  const send = (method, params = {}) => new Promise(r => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
  ws.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Page.javascriptDialogOpening') send('Page.handleJavaScriptDialog', { accept: true });
    if (m.method === 'Runtime.exceptionThrown') problems.push(`${name} exception: ${m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text}`);
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') problems.push(`${name} console: ${m.params.entry.text}`);
  });
  await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable');
  const run = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); return r.result.result?.value; };
  const waitFor = async (expr, label, ms = 15000) => { for (let t = 0; t < ms; t += 100) { if (await run(expr)) return; await sleep(100); } throw new Error(`${name}: timed out waiting for ${label}`); };
  await send('Page.navigate', { url: 'http://127.0.0.1:8791/?emulators' });
  await waitFor("document.querySelector('#connection')?.textContent.includes('Running') && !document.querySelector('#account-button').disabled", 'model and accounts');
  return { run, waitFor, close: () => ws.close() };
}
async function say(tab, text) {
  await tab.run(`document.querySelector('#prompt').value = ${JSON.stringify(text)}; document.querySelector('#chat-form').requestSubmit();`);
  await sleep(150); await tab.waitFor("!document.querySelector('#send').disabled", 'reply');
}
const count = tab => tab.run("document.querySelectorAll('#messages .message').length");
const docs = async uid => (await (await fetch(`http://127.0.0.1:8080/v1/projects/kelus-ide/databases/(default)/documents/users/${uid}/b1Chats`, { headers: { Authorization: 'Bearer owner' } })).json()).documents || [];
const results = []; const check = (name, ok, detail = '') => results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
try {
  const a = await openTab('tab A');
  await a.run(`document.querySelector('#account-button').click(); document.querySelector('#email').value = 'e2e@example.com'; document.querySelector('#password').value = 'secret123'; document.querySelector('#email-form').requestSubmit(document.querySelector('[data-mode=create]'));`);
  await a.waitFor("document.querySelector('#account-button').textContent === 'Account'", 'sign-up');
  check('create account', true);
  const uid = (await (await fetch('http://127.0.0.1:9099/identitytoolkit.googleapis.com/v1/projects/kelus-ide/accounts:batchGet', { headers: { Authorization: 'Bearer owner' } })).json()).users?.[0]?.localId;
  await say(a, 'hello!'); await say(a, 'write a python function to reverse a string');
  await sleep(1500);
  let saved = await docs(uid);
  const fields = saved[0]?.fields;
  check('chat saved to account', saved.length === 1 && fields.messages.arrayValue.values.length === 4, `${saved.length} doc(s), title "${fields?.title?.stringValue}"`);
  check('chat listed in sidebar', await a.run("document.querySelectorAll('aside [data-chat-list] .chat-item').length") === 1);
  const b = await openTab('tab B');
  await b.waitFor("document.querySelector('#account-button').textContent === 'Account' && document.querySelectorAll('aside .chat-item').length === 1", 'session and chat list in second tab');
  check('second tab stays signed in and sees chat', true);
  await b.run("document.querySelector('aside .chat-item').click()");
  check('reopen saved chat', await count(b) === 4, `${await count(b)} messages`);
  await say(b, 'thanks');
  await a.waitFor("document.querySelectorAll('#messages .message').length === 6", 'live update in tab A', 10000);
  check('live update reaches other tab', true, await a.run("document.querySelector('#messages .message:last-child pre').textContent"));
  await a.run("document.querySelector('aside .chat-delete').click()");
  await b.waitFor("document.querySelectorAll('aside .chat-item').length === 0", 'deletion reaching tab B', 10000);
  saved = await docs(uid);
  check('delete chat everywhere', saved.length === 0 && await count(a) === 0);
  await say(a, 'hi'); await sleep(1200);
  await a.run("document.querySelector('#account-button').click(); document.querySelector('#sign-out').click();");
  await a.waitFor("document.querySelector('#account-button').textContent === 'Sign in'", 'sign-out');
  check('sign out clears screen', await count(a) === 0 && await a.run("!document.querySelector('#sidebar-signed-out').hidden"));
  await say(a, 'hello'); await sleep(800);
  check('signed-out chat is not saved', (await docs(uid)).length === 1, 'only the chat from before sign-out exists');
  a.close(); b.close();
} catch (error) { results.push('FAIL  ' + error.message); }
console.log(results.join('\n')); console.log(problems.length ? 'PAGE ERRORS:\n' + problems.join('\n') : 'no page errors');
chrome.kill(); process.exit(results.every(r => r.startsWith('PASS')) && !problems.length ? 0 : 1);
