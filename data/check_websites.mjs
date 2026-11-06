



import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const [file, count = '40'] = process.argv.slice(2);
// the last reply's page is checked, so chats with an edit (data/make_cide_followups.py) test the edited page
const rows = (await readFile(file, 'utf8')).trim().split('\n').map(line => JSON.parse(line)).filter(r => r.messages.at(-1).content.includes('```html\n'));

const sample = [];
for (const feature of new Set(rows.flatMap(r => r.features))) sample.push(...rows.filter(r => r.features.includes(feature)).slice(0, 2));
while (sample.length < Number(count)) sample.push(rows[Math.floor(Math.random() * rows.length)]);

const STEPS = {
  counter: `document.getElementById('increase').click(); return document.getElementById('count').textContent !== '0';`,
  todo: `const input = document.getElementById('todo-input'); input.value = 'Buy milk';
    const before = document.querySelectorAll('#todo-list li').length; document.getElementById('todo-form').requestSubmit(); const items = document.querySelectorAll('#todo-list li');
    items[0].querySelector('span').click(); const done = document.querySelector('#todo-list li').classList.contains('done');
    document.querySelector('#todo-list button').click(); return items.length === before + 1 && document.querySelectorAll('#todo-list li').length === before;`,
  tip: `return document.getElementById('tip-amount').textContent === '$7.50';`,
  converter: `const a = document.getElementById('first-unit'); a.value = '100'; a.dispatchEvent(new Event('input'));
    return document.getElementById('second-unit').value !== '';`,
  quiz: `for (let i = 0; i < 3; i++) document.querySelector('#answers button').click();
    return /You scored \\d out of 3/.test(document.getElementById('quiz-result').textContent);`,
  stopwatch: `document.getElementById('start').click(); await new Promise(r => setTimeout(r, 350));
    document.getElementById('stop').click(); return document.getElementById('time').textContent !== '00:00.0';`,
  contact: `const form = document.getElementById('contact-form'); form.requestSubmit();
    const complained = document.getElementById('form-error').textContent.length > 0;
    document.getElementById('contact-name').value = 'Sam'; document.getElementById('contact-email').value = 'sam@example.com';
    document.getElementById('contact-message').value = 'Hello there, this works.'; form.requestSubmit();
    return complained && !document.getElementById('form-success').hidden;`,
  dark_mode: `const before = document.body.classList.contains('dark'); document.getElementById('theme-toggle').click();
    return document.body.classList.contains('dark') !== before && localStorage.getItem('theme') === (before ? 'light' : 'dark');`,
  faq: `const q = document.querySelector('.faq-question'); q.click(); return !q.nextElementSibling.hidden;`,
  tabs: `document.querySelectorAll('.tab')[1].click(); return !document.getElementById('prices').hidden && document.getElementById('about').hidden;`,
  modal: `document.getElementById('open-modal').click(); const opened = !document.getElementById('modal').hidden;
    document.getElementById('close-modal').click(); return opened && document.getElementById('modal').hidden;`,
  slideshow: `document.getElementById('next').click(); return document.getElementById('caption').textContent.includes('(2/3)');`,
  quotes: `document.getElementById('new-quote').click(); return !document.getElementById('quote').textContent.startsWith('Click');`,
  guess: `const g = document.getElementById('guess-input'); g.value = '1'; document.getElementById('guess-form').requestSubmit();
    return document.getElementById('guess-hint').textContent.length > 0;`,
  cart: `document.querySelector('.add-item').click(); return document.getElementById('cart-total').textContent !== '$0.00';`,
  filter_list: `const s = document.getElementById('search'); s.value = 'zzz'; s.dispatchEvent(new Event('input'));
    return !document.getElementById('no-results').hidden;`,
  char_count: `const n = document.getElementById('note'); const before = Number(document.getElementById('chars-left').textContent);
    n.value = 'abc'; n.dispatchEvent(new Event('input')); return Number(document.getElementById('chars-left').textContent) === before - 3;`,
  login: `const f = document.getElementById('login-form'); f.requestSubmit(); const complained = document.getElementById('login-error').textContent.length > 0;
    const u = document.getElementById('login-user'); u.value = u.type === 'email' ? 'sam@example.com' : 'sam_lee';
    document.getElementById('login-password').value = 'secret123'; f.requestSubmit(); return complained && !document.getElementById('login-success').hidden;`,
  signup: `const f = document.getElementById('signup-form'); const v = (id, text) => { document.getElementById(id).value = text; };
    v('signup-name', 'Sam'); v('signup-email', 'sam@example.com'); v('signup-password', 'secret123'); v('signup-confirm', 'secret124'); f.requestSubmit();
    const complained = /match/.test(document.getElementById('signup-error').textContent); v('signup-confirm', 'secret123'); f.requestSubmit();
    return complained && !document.getElementById('signup-success').hidden;`,
  calculator: `const press = keys => keys.forEach(k => document.querySelector('#keys [data-key="' + k + '"]').click());
    press(['1', '2', '+', '3', '=']); const sum = document.getElementById('display').textContent === '15';
    press(['C', '7', '/', '2', '=']); const half = document.getElementById('display').textContent === '3.5';
    press(['C', '1', '/', '0', '=']); return sum && half && document.getElementById('display').textContent === 'Error';`,
  countdown: `document.getElementById('timer-minutes').value = '1'; document.getElementById('timer-start').click();
    await new Promise(r => setTimeout(r, 1300)); const ticking = document.getElementById('timer-display').textContent === '00:59';
    document.getElementById('timer-reset').click(); return ticking && document.getElementById('timer-display').textContent === '00:00';`,
  landing: `return !!document.getElementById('features') && document.querySelectorAll('.card').length > 0;`,
  hello: `return /hello/i.test(document.querySelector('h1').textContent);`,
  template: `return !!document.querySelector('header h1') && !!document.querySelector('main') && !!document.querySelector('footer');`,
  table: `return document.querySelectorAll('thead th').length > 1 && document.querySelectorAll('tbody tr').length > 1;`,
  profile: `return document.querySelectorAll('section').length >= 3 && !!document.querySelector('a[href^="mailto:"]');`,
  navbar: `const b = document.getElementById('menu-toggle'); b.click(); const opened = document.getElementById('menu').classList.contains('open');
    b.click(); return opened && !document.getElementById('menu').classList.contains('open') && b.getAttribute('aria-expanded') === 'false';`,
  button: `document.getElementById('message-button').click(); return document.getElementById('message').textContent.length > 0;`,
};

const target = await (await fetch('http://localhost:9444/json/new?about:blank', { method: 'PUT' })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve => { ws.onopen = resolve; });
let id = 0, errors = [];
const pending = new Map();
ws.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.id && pending.has(message.id)) { pending.get(message.id)(message); pending.delete(message.id); }
  if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
  if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') errors.push(message.params.args.map(a => a.value).join(' '));
};
const send = (method, params = {}) => new Promise(resolve => { const n = ++id; pending.set(n, resolve); ws.send(JSON.stringify({ id: n, method, params })); });
await send('Runtime.enable'); await send('Page.enable');

const folder = await mkdtemp(path.join(tmpdir(), 'kani-sites-'));
let passed = 0;
const failures = [];
for (const [index, row] of sample.entries()) {
  const html = row.messages.at(-1).content.split('```html\n')[1].split('\n```')[0];
  const page = path.join(folder, `site-${index}.html`);
  await writeFile(page, html);
  errors = [];
  await send('Page.navigate', { url: 'file://' + page });
  await new Promise(resolve => setTimeout(resolve, 400));
  const results = {};
  for (const feature of row.features) {
    const { result } = await send('Runtime.evaluate', { expression: `(async () => { ${STEPS[feature]} })()`, awaitPromise: true, returnByValue: true });
    results[feature] = result.result?.value === true;
  }
  const ok = Object.values(results).every(Boolean) && errors.length === 0;
  if (ok) passed++; else failures.push({ index, features: row.features, results, errors });
}
console.log(JSON.stringify({ checked: sample.length, working: passed, failures: failures.slice(0, 5) }, null, 1));
ws.close();
process.exit(failures.length ? 1 : 0);
