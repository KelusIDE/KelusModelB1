import { loadModel, prepareReply } from './b1.js';
import { attachSvgPreview, svgPrompt, svgRequest } from './svg.js';
import { webAnswer, webRequest } from './web.js';
import { mountAttachments, prepareAttachments } from './attachments.js';

const $ = selector => document.querySelector(selector);
const LABELS = { local_clock: 'LOCAL CLOCK', local_calculator: 'LOCAL CALCULATOR', web:'WEB · WIKIPEDIA', file_analysis:'LOCAL FILE INSPECTION', model: 'KANI SEAL 3' };
let webAborter;
let model, generating = false, stopped = false;
// The full conversation; the model only ever sees its last 15 messages.
let transcript = [], chatId = crypto.randomUUID(), savedChat = false;
let account = null, user = null, stopWatchingChats = null, savedChats = [];
let previews = [];
let privateChat = false;
function preview(body, reply) {
  const release = attachSvgPreview(body, reply);
  if (release) previews.push(release);
  return Boolean(release);
}

function message(role, text, source = 'model') {
  const article = document.createElement('article'); article.className = 'message ' + role;
  if (role === 'assistant' && source !== 'model') article.classList.add('helper');
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'YOU' : LABELS[source] || LABELS.model;
  const body = document.createElement('pre'); body.textContent = text;
  article.append(label, body); $('#messages').append(article); return body;
}
const scrollLatest = () => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
// Let the page paint between tokens. Background tabs never run animation frames,
// so yield through a message task there instead of pausing the reply.
const yieldChannel = new MessageChannel();
const nextFrame = () => new Promise(resolve => {
  if (!document.hidden) { requestAnimationFrame(() => resolve()); return; }
  yieldChannel.port1.onmessage = () => resolve(); yieldChannel.port2.postMessage(null);
});
const plain = messages => messages.map(({ role, content, source }) => source ? { role, content, source } : { role, content });

function showError(text) { $('#error').textContent = text; $('#error').hidden = false; }

const attachments = mountAttachments({form:$('#chat-form'), input:$('#prompt'), onError:showError});

function setBusy(busy) {
  attachments.setBusy(busy);
  generating = busy;
  $('#send').disabled = busy || !model; $('#new-chat').disabled = busy; $('#new-chat-mobile').disabled = busy; $('#stop').hidden = !busy;
}

function renderTranscript() {
  previews.forEach(release => release()); previews = [];
  $('#messages').replaceChildren();
  for (const item of transcript) {
    const body = message(item.role, item.content, item.source);
    if (item.role === 'assistant') preview(body, item.content);
  }
  $('#welcome').hidden = transcript.length > 0;
}

function startChat(messages = [], id = crypto.randomUUID(), saved = false) {
  if (generating) return;
  attachments.clear(); privateChat = false;
  transcript = plain(messages); chatId = id; savedChat = saved;
  renderTranscript(); $('#error').hidden = true; renderChatList();
}

async function persist() {
  if (privateChat || !account || !user || !transcript.length) return;
  // Firestore applies writes in order, so a later save can safely assume this one created the chat.
  const created = savedChat; savedChat = true;
  try {
    await account.saveChat(user.uid, chatId, transcript, { created });
  } catch (error) {
    savedChat = created;
    showError('This chat was not saved: ' + account.friendlyError(error));
  }
}

async function send(prompt) {
  if (generating || attachments.reading || !model || (!prompt.trim() && !attachments.files.length)) return;
  const attached = attachments.files.length > 0;
  prompt = prompt.trim() || 'Analyze these files.';
  let fileTurn;
  try { if (attached) fileTurn = prepareAttachments(prompt, attachments.files, model.tokenizer, model.config.context, model.config.context >= 1024 ? 768 : 240); }
  catch(error) { showError(error.message); return; }
  if (attached) privateChat = true;
  setBusy(true); stopped = false;
  $('#error').hidden = true; $('#welcome').hidden = true;
  const shown = prompt + (attached ? '\nAttached: ' + attachments.files.map(f=>f.name).join(', ') : '');
  message('user', shown); const output = message('assistant', '');
  $('#prompt').value = ''; transcript.push({ role: 'user', content: shown, excludeFromContext:attached });
  let reply = '', source = 'model';
  const started = performance.now();
  try {
    const temperature = Math.min(2, Math.max(0, Number($('#temperature').value) || 0));
    let context = transcript.filter(item=>!item.excludeFromContext).slice(-15).map(({ role, content }) => ({ role, content }));
    if (fileTurn) context = [{role:'user',content:fileTurn.text}];
    if (context[0]?.role !== 'user') context = context.slice(1);
    const drawing = attached ? null : svgRequest(context.at(-1).content);
    if (drawing) context = [{ role: 'user', content: svgPrompt(drawing) }];
    const maxTokens = model.config.context >= 1024 ? 768 : drawing ? Math.min(512, model.config.context) : 240;
    const online=attached ? null : webRequest(prompt);
    const { retained, omitted, tool } = online || fileTurn?.helper ? {retained:[],omitted:0,tool:null} : prepareReply(model, context, maxTokens);
    if(fileTurn?.helper) { reply=fileTurn.text; source='file_analysis'; output.parentElement.classList.add('helper'); } else if(online) {
      webAborter=new AbortController();
      output.textContent='Searching Wikipedia…';
      reply=(await webAnswer(online,fetch,webAborter.signal)).text;source='web';
      output.parentElement.classList.add('helper');
    } else if (tool) {
      [reply, source] = tool;
      output.parentElement.classList.add('helper');
    } else {
      let count = 0;
      for (const text of model.generate(retained, { maxTokens, temperature })) {
        reply += text; output.textContent = reply;
        if (++count % 4 === 0) await nextFrame();
        if (stopped) break;
      }
    }
    output.previousElementSibling.textContent = LABELS[source];
    output.textContent = reply || '(The model ended its reply without text. Try another wording.)';
    if (fileTurn && !fileTurn.helper) { const note=document.createElement('div'); note.className='attachment-receipt'; note.textContent=fileTurn.receipt; output.parentElement.append(note); }
    const rendered = source === 'model' && preview(output, reply);
    if (drawing && !rendered && !stopped) showError('The model did not produce a complete, supported SVG. Its response is shown above.');
    $('#generation-info').textContent = `${source==='web'?'Retrieved online':'Generated on this device'} in ${((performance.now() - started) / 1000).toFixed(2)}s`
      + (omitted ? ' · Older turns omitted to fit this model' : '') + (privateChat ? ' · Attachment chat stays local; not saved to your account.' : ' · Review and test generated code.');
    transcript.push({ role:'assistant', content:reply, source, excludeFromContext:attached });
    void persist();
  } catch (error) {
    showError(error.message);
    transcript.pop();
    if (!reply) output.textContent = '(No response generated.)';
  } finally {
    webAborter=null;
    setBusy(false); $('#prompt').focus(); scrollLatest();
  }
}

// ---- Saved chats

function relativeTime(timestamp) {
  if (!timestamp?.toDate) return 'saving…';
  const seconds = (Date.now() - timestamp.toDate().getTime()) / 1000;
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return timestamp.toDate().toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function renderChatList() {
  $('#sidebar-signed-out').hidden = Boolean(user);
  for (const list of document.querySelectorAll('[data-chat-list]')) {
    list.hidden = !user;
    list.replaceChildren();
    if (!user) continue;
    if (!savedChats.length) {
      const empty = document.createElement('li'); empty.className = 'chat-empty'; empty.textContent = 'No saved chats yet. Send a message to start one.';
      list.append(empty); continue;
    }
    for (const chat of savedChats) {
      const item = document.createElement('li'); item.setAttribute('aria-current', String(chat.id === chatId));
      const open = document.createElement('button'); open.className = 'chat-item';
      const title = document.createElement('span'); title.textContent = chat.title || 'Untitled chat';
      const when = document.createElement('small'); when.textContent = relativeTime(chat.updatedAt);
      open.append(title, when);
      open.addEventListener('click', () => { startChat(chat.messages || [], chat.id, true); $('#account-dialog').close(); });
      const remove = document.createElement('button'); remove.className = 'chat-delete'; remove.textContent = '×';
      remove.setAttribute('aria-label', `Delete chat: ${chat.title || 'Untitled chat'}`);
      remove.addEventListener('click', async () => {
        if (!confirm('Delete this chat from your account? This cannot be undone.')) return;
        try {
          await account.deleteChat(user.uid, chat.id);
          if (chat.id === chatId) startChat();
        } catch (error) { showError('Could not delete the chat: ' + account.friendlyError(error)); }
      });
      item.append(open, remove); list.append(item);
    }
  }
}

function onChats(chats) {
  savedChats = chats;
  const current = chats.find(chat => chat.id === chatId);
  // Another device or tab changed the open chat: show its latest version.
  if (current && !current.pending && !generating && !privateChat && JSON.stringify(plain(current.messages || [])) !== JSON.stringify(transcript)) {
    transcript = plain(current.messages || []); savedChat = true; renderTranscript();
  }
  renderChatList();
}

function onUser(next) {
  const wasSignedIn = Boolean(user);
  user = next;
  stopWatchingChats?.(); stopWatchingChats = null; savedChats = [];
  $('#account-button').textContent = user ? 'Account' : 'Sign in';
  $('#signed-out-view').hidden = Boolean(user); $('#signed-in-view').hidden = !user;
  $('#account-email').textContent = user ? user.email || user.displayName || 'your account' : '';
  if (user) {
    stopWatchingChats = account.watchChats(user.uid, onChats, error => showError('Saved chats are unavailable: ' + account.friendlyError(error)));
    if (transcript.length && !savedChat) void persist();
  } else if (wasSignedIn) {
    startChat(); // Do not leave an account's chat on screen after signing out.
  }
  renderChatList();
}

// ---- Account dialog

const dialog = $('#account-dialog');
function authStatus(text, error = false) { $('#auth-status').textContent = text; $('#auth-status').classList.toggle('error', error); }
async function authAction(action, success = '') {
  authStatus('Working…');
  try { await action(); authStatus(success); if (!success) dialog.close(); } catch (error) { authStatus(account.friendlyError(error), true); }
}

document.querySelectorAll('[data-open-account]').forEach(button => button.addEventListener('click', () => {
  if (!account) return;
  authStatus(''); dialog.showModal();
}));
$('#close-account').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
$('#email-form').addEventListener('submit', event => {
  event.preventDefault();
  const email = $('#email').value.trim(), password = $('#password').value;
  const create = event.submitter?.dataset.mode === 'create';
  void authAction(() => create ? account.createAccount(email, password) : account.signInEmail(email, password));
});
$('#forgot').addEventListener('click', () => {
  const email = $('#email').value.trim();
  if (!email) { authStatus('Enter your email first.', true); return; }
  void authAction(() => account.resetPassword(email), 'If an account exists for that email, a reset link is on its way.');
});
document.querySelectorAll('[data-provider]').forEach(button => button.addEventListener('click', () => {
  void authAction(() => account.signInWith(button.dataset.provider));
}));
$('#sign-out').addEventListener('click', () => void authAction(() => account.signOutUser()));
$('#dialog-new-chat').addEventListener('click', () => { startChat(); dialog.close(); $('#prompt').focus(); });

// ---- Chat controls

$('#chat-form').addEventListener('submit', event => { event.preventDefault(); void send($('#prompt').value); });
$('#prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#chat-form').requestSubmit(); }
});
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { void send(button.dataset.prompt); }));
$('#stop').addEventListener('click', () => { stopped = true; webAborter?.abort(); });
for (const button of [$('#new-chat'), $('#new-chat-mobile')]) button.addEventListener('click', () => { startChat(); $('#prompt').focus(); });
$('#reload').addEventListener('click', () => location.reload());

// A redeploy with new weights shows a banner instead of silently changing a running chat.
async function checkForUpdate() {
  if (!model) return;
  try {
    const latest = await (await fetch('model/b1.json', { cache: 'no-store' })).json();
    if (latest.weights_sha256 !== model.manifest.weights_sha256) $('#update-banner').hidden = false;
  } catch { /* Offline: check again later. */ }
}
setInterval(checkForUpdate, 10 * 60 * 1000);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') void checkForUpdate(); });

loadModel('model/').then(loaded => {
  model = loaded;
  const { training, config } = model.manifest;
  $('#parameters').textContent = model.parameterCount.toLocaleString();
  $('#steps').textContent = (training.steps || 0).toLocaleString();
  $('#context').textContent = `${config.context} BPE tokens`;
  $('#connection').textContent = '● Running in your browser';
  setBusy(false);
}).catch(error => {
  $('#connection').textContent = 'Not loaded';
  showError(error.message);
});

// Accounts load separately, so chatting works even if Firebase is unreachable.
import('./account.js').then(module => {
  account = module;
  module.watchUser(onUser);
  $('#account-button').disabled = false;
}).catch(() => {
  $('#account-button').textContent = 'Accounts unavailable';
});
