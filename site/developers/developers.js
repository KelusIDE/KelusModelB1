

import { watchUser, signInEmail, signInWith, signOutUser, friendlyError } from '../account.js';

const API = location.hostname === '127.0.0.1' && new URLSearchParams(location.search).get('api') || 'https://kani-api.chatgpt-ai-5-2o2.workers.dev';
const $ = id => document.getElementById(id);
let user = null;

const status = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
const when = time => time ? new Date(time).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : 'Never';

async function call(method, path, body) {
  const response = await fetch(API + path, { method, headers: { Authorization: `Bearer ${await user.getIdToken()}`, ...(body && { 'Content-Type': 'application/json' }) },
    body: body && JSON.stringify(body) });
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error?.message || `The key service answered ${response.status}.`);
  return data;
}

async function refresh() {
  const data = await call('GET', '/v1/keys');
  $('usage').textContent = `Today: ${data.today} of ${data.daily_limit} requests used (resets 00:00 UTC). Keys: ${data.keys.length} of ${data.keys_limit}.`;
  const rows = data.keys.map(key => {
    const row = document.createElement('tr');
    for (const text of [key.name, key.hint, when(key.created), when(key.last_used), key.requests.toLocaleString()]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.append(cell);
    }
    row.children[1].className = 'mono';
    const remove = document.createElement('button');
    remove.className = 'link danger';
    remove.textContent = 'Delete';
    remove.addEventListener('click', async () => {
      if (!confirm(`Delete “${key.name}” (${key.hint})? Apps using it stop working at once.`)) return;
      try { await call('DELETE', '/v1/keys/' + key.id); status('Key deleted.'); await refresh(); } catch (error) { status(error.message, true); }
    });
    const cell = document.createElement('td');
    cell.append(remove);
    row.append(cell);
    return row;
  });
  if (!rows.length) {
    const row = document.createElement('tr'), cell = document.createElement('td');
    cell.colSpan = 6;
    cell.className = 'empty';
    cell.textContent = 'No keys yet. Name one below and press Create key.';
    row.append(cell);
    rows.push(row);
  }
  $('key-rows').replaceChildren(...rows);
  $('create-form').hidden = data.keys.length >= data.keys_limit;
}

watchUser(async current => {
  user = current;
  $('signed-out').hidden = Boolean(user);
  $('signed-in').hidden = !user;
  $('new-key').hidden = true;
  status('');
  if (!user) return;
  $('who').textContent = 'Signed in as ' + (user.email || 'your Kelus account');
  try { await refresh(); } catch (error) { status(error.message, true); }
});

$('google').addEventListener('click', () => signInWith('google').catch(error => status(friendlyError(error), true)));
$('github').addEventListener('click', () => signInWith('github').catch(error => status(friendlyError(error), true)));
$('email-form').addEventListener('submit', event => {
  event.preventDefault();
  signInEmail($('email').value, $('password').value).catch(error => status(friendlyError(error), true));
});
$('sign-out').addEventListener('click', () => signOutUser());

$('create-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  try {
    const made = await call('POST', '/v1/keys', { name: $('key-name').value });
    $('new-key-value').textContent = made.key;
    $('new-key').hidden = false;
    $('key-name').value = '';
    status('Key created.');
    await refresh();
  } catch (error) {
    status(error.message, true);
  } finally {
    button.disabled = false;
  }
});

$('copy-key').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('new-key-value').textContent); status('Copied.'); }
  catch { status('Select the key and copy it by hand.', true); }
});

for (const tab of document.querySelectorAll('[data-tab]')) {
  tab.addEventListener('click', () => {
    for (const other of document.querySelectorAll('[data-tab]')) other.setAttribute('aria-selected', String(other === tab));
    for (const panel of document.querySelectorAll('[data-panel]')) panel.hidden = panel.dataset.panel !== tab.dataset.tab;
  });
}
