import { Tokenizer } from './b1.js';
import { mountAttachments, prepareAttachments } from './attachments.js';
import { attachSvgPreview, svgPrompt, svgRequest } from './svg.js';
import { webAnswer, webRequest } from './web.js';
const $ = selector => document.querySelector(selector);
let history = [], generating = false, aborter;
let previews = [];
let replyBudget = 300, fileTokenizer, fileContext;
function showError(text) { $('#error').textContent=text; $('#error').hidden=false; }
const attachments = mountAttachments({form:$('#chat-form'),input:$('#prompt'),onError:showError});
function message(role, text) {
  const article = document.createElement('article'); article.className = 'message ' + role;
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'YOU' : 'KANI SEAL 3';
  const body = document.createElement('pre'); body.textContent = text;
  article.append(label, body); $('#messages').append(article); return body;
}
function scrollLatest() { window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); }
async function send(prompt) {
  if (generating || attachments.reading || (!prompt.trim() && !attachments.files.length)) return;
  const attached=attachments.files.length>0; prompt=prompt.trim()||'Analyze these files.';
  let fileTurn;
  try { if(attached) fileTurn=prepareAttachments(prompt,attachments.files,fileTokenizer,fileContext,replyBudget); }
  catch(error){showError(error.message);return;}
  generating = true; attachments.setBusy(true); $('#send').disabled = true; $('#new-chat').disabled = true;
  $('#error').hidden = true; $('#welcome').hidden = true;
  message('user', prompt+(attached?'\nAttached: '+attachments.files.map(f=>f.name).join(', '):'')); const output = message('assistant', '');
  $('#prompt').value = ''; history.push({ role: 'user', content: prompt });
  let reply = '', done = false;
  aborter = new AbortController();
  try {
    const online=attached?null:webRequest(prompt);
    if(fileTurn?.helper){reply=fileTurn.text;output.textContent=reply;output.previousElementSibling.textContent='LOCAL FILE INSPECTION';history.pop();$('#generation-info').textContent='Files inspected locally · no model inference';return;}
    if(online) {
      output.previousElementSibling.textContent='WEB · WIKIPEDIA';output.textContent='Searching Wikipedia…';
      reply=(await webAnswer(online,fetch,aborter.signal)).text;output.textContent=reply;
      history.pop();
      $('#generation-info').textContent='Retrieved online from Wikipedia · only this lookup was sent';
      return;
    }
    const drawing = attached?null:svgRequest(prompt), messages = fileTurn ? [{role:'user',content:fileTurn.text}] : drawing ? [{role:'user',content:svgPrompt(drawing)}] : history.slice(-15).map(item => ({ ...item }));
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: aborter.signal,
      body: JSON.stringify({ messages, max_tokens: drawing ? 512 : replyBudget, temperature: Number($('#temperature').value) }) });
    if (!response.ok) throw new Error((await response.json()).error || 'The model could not answer.');
    const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = '';
    while (true) {
      const result = await reader.read();
      pending += decoder.decode(result.value, { stream: !result.done });
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.source) output.previousElementSibling.textContent = event.source === 'local_clock' ? 'LOCAL CLOCK' : event.source === 'local_calculator' ? 'LOCAL CALCULATOR' : 'KANI SEAL 3';
        if (event.error) throw new Error(event.error);
        if (event.text) { reply += event.text; output.textContent = reply; }
        if (event.done) { done = true; $('#generation-info').textContent = `Generated locally in ${event.seconds}s${event.history_messages_omitted ? ' · Older turns omitted to fit this model' : ''} · Review and test generated code.`; }
      }
      if (result.done) break;
    }
    if (!done) throw new Error('Generation ended before completion.');
    if (!reply) output.textContent = '(The model ended its reply without text. Try another prompt or continue training.)';
    if(fileTurn){const note=document.createElement('div');note.className='attachment-receipt';note.textContent=fileTurn.receipt;output.parentElement.append(note);}
    const release = attachSvgPreview(output, reply);
    if (release) previews.push(release);
    else if (drawing) { $('#error').textContent = 'The model did not produce a complete, supported SVG. Its response is shown above.'; $('#error').hidden = false; }
    if(attached)history.pop();else history.push({ role: 'assistant', content: reply }); history = history.slice(-16);
  } catch (error) {
    $('#error').textContent = error.message; $('#error').hidden = false;
    history.pop();
    if (!reply) output.textContent = '(No response generated.)';
  } finally {
    generating = false; attachments.setBusy(false); $('#send').disabled = false; $('#new-chat').disabled = false; $('#prompt').focus(); scrollLatest();
  }
}
$('#chat-form').addEventListener('submit', event => { event.preventDefault(); void send($('#prompt').value); });
$('#prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#chat-form').requestSubmit(); } });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { void send(button.dataset.prompt); }));
$('#new-chat').addEventListener('click', () => { if (generating) return; attachments.clear(); history = []; previews.forEach(release => release()); previews = []; $('#messages').replaceChildren(); $('#welcome').hidden = false; $('#error').hidden = true; $('#prompt').focus(); });
fetch('/api/info').then(response => { if (!response.ok) throw new Error('Model server unavailable.'); return response.json(); }).then(info => {
  replyBudget = info.config.context >= 1024 ? 512 : 300;
  fileTokenizer=new Tokenizer(info.tokenizer_spec);fileContext=info.config.context;
  $('#parameters').textContent = info.parameters.toLocaleString(); $('#steps').textContent = (info.training.steps || 0).toLocaleString();
  $('#context').textContent = info.config.context + (info.tokenizer === 'bpe' ? ' BPE tokens' : ' byte tokens'); $('#connection').textContent = '● Running locally';
}).catch(error => { $('#connection').textContent = 'Offline'; $('#error').textContent = error.message; $('#error').hidden = false; });
