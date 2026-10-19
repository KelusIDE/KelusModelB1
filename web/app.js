const $ = selector => document.querySelector(selector);
let history = [], generating = false, aborter;
function message(role, text) {
  const article = document.createElement('article'); article.className = 'message ' + role;
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'YOU' : 'KELUS / B1';
  const body = document.createElement('pre'); body.textContent = text;
  article.append(label, body); $('#messages').append(article); return body;
}
function scrollLatest() { window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }); }
async function send(prompt) {
  if (generating || !prompt.trim()) return;
  generating = true; $('#send').disabled = true; $('#new-chat').disabled = true;
  $('#error').hidden = true; $('#welcome').hidden = true;
  message('user', prompt); const output = message('assistant', '');
  $('#prompt').value = ''; history.push({ role: 'user', content: prompt });
  let reply = '', done = false;
  aborter = new AbortController();
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: aborter.signal,
      body: JSON.stringify({ messages: history.slice(-15), max_tokens: 300, temperature: Number($('#temperature').value) }) });
    if (!response.ok) throw new Error((await response.json()).error || 'The model could not answer.');
    const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = '';
    while (true) {
      const result = await reader.read();
      pending += decoder.decode(result.value, { stream: !result.done });
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.source) output.previousElementSibling.textContent = event.source === 'local_clock' ? 'LOCAL CLOCK' : event.source === 'local_calculator' ? 'LOCAL CALCULATOR' : 'KELUS / B1';
        if (event.error) throw new Error(event.error);
        if (event.text) { reply += event.text; output.textContent = reply; }
        if (event.done) { done = true; $('#generation-info').textContent = `Generated locally in ${event.seconds}s${event.history_messages_omitted ? ' · Older turns omitted; B1 uses one prior exchange' : ''} · Review and test generated code.`; }
      }
      if (result.done) break;
    }
    if (!done) throw new Error('Generation ended before completion.');
    if (!reply) output.textContent = '(The model ended its reply without text. Try another prompt or continue training.)';
    history.push({ role: 'assistant', content: reply }); history = history.slice(-16);
  } catch (error) {
    $('#error').textContent = error.message; $('#error').hidden = false;
    history.pop();
    if (!reply) output.textContent = '(No response generated.)';
  } finally {
    generating = false; $('#send').disabled = false; $('#new-chat').disabled = false; $('#prompt').focus(); scrollLatest();
  }
}
$('#chat-form').addEventListener('submit', event => { event.preventDefault(); void send($('#prompt').value); });
$('#prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#chat-form').requestSubmit(); } });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { void send(button.dataset.prompt); }));
$('#new-chat').addEventListener('click', () => { if (generating) return; history = []; $('#messages').replaceChildren(); $('#welcome').hidden = false; $('#error').hidden = true; $('#prompt').focus(); });
fetch('/api/info').then(response => { if (!response.ok) throw new Error('Model server unavailable.'); return response.json(); }).then(info => {
  $('#parameters').textContent = info.parameters.toLocaleString(); $('#steps').textContent = (info.training.steps || 0).toLocaleString();
  $('#context').textContent = info.config.context + (info.tokenizer === 'bpe' ? ' BPE tokens' : ' byte tokens'); $('#connection').textContent = '● Running locally';
}).catch(error => { $('#connection').textContent = 'Offline'; $('#error').textContent = error.message; $('#error').hidden = false; });
