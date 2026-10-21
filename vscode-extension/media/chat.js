// Chat webview. Talks to extension.js with postMessage; never runs model output.
(async () => {
  'use strict';
  const vscode = acquireVsCodeApi();
  const NAME = document.body.dataset.name || 'Kani Seal 3';
  const BRAND = NAME;
  const VERSION = document.body.dataset.codename || 'Kani B2:10F';
  let mountAttachments, prepareAttachments, Tokenizer;
  try { ({mountAttachments,prepareAttachments}=await import(document.body.dataset.attachments)); ({Tokenizer}=await import(document.body.dataset.engine)); } catch { /* The setup banner explains a missing repository. */ }
  const SVG = 'http://www.w3.org/2000/svg';
  const WORDMARK = [
    '██╗  ██╗ █████╗ ███╗   ██╗██╗',
    '██║ ██╔╝██╔══██╗████╗  ██║██║',
    '█████╔╝ ███████║██╔██╗ ██║██║',
    '██╔═██╗ ██╔══██║██║╚██╗██║██║',
    '██║  ██╗██║  ██║██║ ╚████║██║',
    '╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝',
  ].join('\n');
  const SUGGESTIONS = [
    ['chat', 'what can you do?'],
    ['concept', 'explain recursion'],
    ['code', 'write a python function that reverses a string'],
    ['math', 'what is 2 + 2?'],
  ];
  const SOURCES = { local_calculator: 'calculator', local_clock: 'clock', file_analysis:'file inspection' };
  const ICONS = {
    plus: 'M8 3v10M3 8h10',
    send: 'M8 13V3M3.5 7.5 8 3l4.5 4.5',
    stop: 'M5 5h6v6H5z',
    restart: 'M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3',
    gear: 'M6.9 1.5h2.2l.3 1.7 1.2.5 1.4-1 1.6 1.6-1 1.4.5 1.2 1.7.3v2.2l-1.7.3-.5 1.2 1 1.4-1.6 1.6-1.4-1-1.2.5-.3 1.7H6.9l-.3-1.7-1.2-.5-1.4 1-1.6-1.6 1-1.4-.5-1.2-1.7-.3V6.9l1.7-.3.5-1.2-1-1.4 1.6-1.6 1.4 1 1.2-.5zM8 6a2 2 0 1 0 0 4 2 2 0 0 0 0-4z',
    expand: 'M9.5 2.5h4v4M13.5 2.5 8.5 7.5M6.5 13.5h-4v-4M2.5 13.5l5-5',
    copy: 'M5.5 5.5h8v8h-8zM10.5 5.5v-3h-8v8h3',
    insert: 'M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10',
  };
  const KEYWORDS = {
    python: new Set('and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield'.split(' ')),
    javascript: new Set('async await break case catch class const continue default delete do else export extends finally for function if import in instanceof let new of return static super switch this throw try typeof var void while yield'.split(' ')),
  };
  const CONSTANTS = new Set('True False None true false null undefined NaN Infinity self'.split(' '));
  const ALIASES = { py: 'python', js: 'javascript', ts: 'javascript', typescript: 'javascript', jsx: 'javascript' };

  let messages = (vscode.getState() || {}).messages || [];
  let status = { state: 'starting' };
  let busy = null;
  let live = null; // { message, content } while a reply streams
  let pending = false;

  for (const message of messages) {
    if (message.streaming) {
      delete message.streaming;
      message.note = message.note || 'Interrupted when the view reloaded.';
    }
  }

  function el(tag, attributes = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) {
      if (key === 'class') node.className = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else if (value !== false && value != null) node.setAttribute(key, value === true ? '' : value);
    }
    node.append(...children.filter(child => child != null));
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS(SVG, 'path');
    path.setAttribute('d', ICONS[name]);
    if (name === 'stop') path.setAttribute('fill', 'currentColor');
    svg.append(path);
    return svg;
  }

  const iconButton = (name, title, onclick) => el('button', { class: 'icon-button', title, 'aria-label': title, onclick }, icon(name));

  // Layout
  const header = el('header', { class: 'bar' },
    el('img', { class: 'logo', src: document.body.dataset.icon, alt: '' }),
    el('span', { class: 'brand' }, BRAND),
    VERSION ? el('span', { class: 'version' }, VERSION) : null,
    el('span', { class: 'spacer' }),
    iconButton('plus', 'New chat', reset),
    iconButton('expand', 'Open in new tab', () => vscode.postMessage({ type: 'openTab' })),
    iconButton('restart', 'Restart model', () => vscode.postMessage({ type: 'restart' })),
    iconButton('gear', 'Settings', () => vscode.postMessage({ type: 'settings' })));
  const log = el('main', { id: 'log', 'aria-live': 'polite' });
  const bannerText = el('div', { class: 'banner-text' });
  const banner = el('div', { class: 'banner', role: 'alert', hidden: true }, bannerText,
    el('div', { class: 'actions' },
      el('button', { class: 'button', onclick: () => vscode.postMessage({ type: 'settings' }) }, 'Open Settings'),
      el('button', { class: 'button secondary', onclick: () => vscode.postMessage({ type: 'restart' }) }, 'Restart')));
  const input = el('textarea', { rows: 2, placeholder: `Ask ${BRAND} anything…`, 'aria-label': `Message ${BRAND}` });
  const sendButton = el('button', { class: 'send', onclick: () => (busy ? stop() : submit()) }, icon('send'));
  const where = el('span', { class: 'chip status' }, el('span', { class: 'dot starting' }), 'Local');
  const model = el('button', { class: 'chip model', title: 'Change the model in Settings', onclick: () => vscode.postMessage({ type: 'settings' }) }, 'starting…');
  const composer = el('footer', { class: 'composer' },
    el('div', { class: 'box' }, input, el('div', { class: 'toolbar' }, where, model, el('span', { class: 'spacer' }), sendButton)),
    el('div', { class: 'hint' }, el('kbd', {}, '⏎'), ' send  ', el('kbd', {}, '⇧⏎'), ' new line  ', el('kbd', {}, 'esc'), ' stop'));
  document.body.append(el('div', { id: 'app' }, header, log, banner, composer));
  const fileError=el('div',{class:'attachment-error',role:'alert',hidden:true}); composer.prepend(fileError);
  const attachments=mountAttachments?.({form:composer.querySelector('.box'),input,onError:text=>{fileError.textContent=text;fileError.hidden=false;},onChange:()=>controls()});

  // Rendering
  function inline(text) {
    const fragment = document.createDocumentFragment();
    text.split(/(`[^`\n]+`)/).forEach((part, i) => {
      fragment.append(i % 2 ? el('code', {}, part.slice(1, -1)) : part);
    });
    return fragment;
  }

  function highlight(code, language) {
    const keywords = KEYWORDS[language] || new Set();
    const comment = language === 'python' ? '#.*' : '\\/\\/.*|\\/\\*[\\s\\S]*?\\*\\/';
    const token = new RegExp(`(${comment})|("(?:\\\\.|[^"\\\\\\n])*"|'(?:\\\\.|[^'\\\\\\n])*'|\`(?:\\\\.|[^\`\\\\])*\`)|(\\b\\d+(?:\\.\\d+)?\\b)|([A-Za-z_$][\\w$]*)`, 'g');
    const fragment = document.createDocumentFragment();
    let last = 0;
    let match;
    while ((match = token.exec(code))) {
      fragment.append(code.slice(last, match.index));
      const [text, isComment, isString, isNumber, word] = match;
      let kind = isComment ? 'comment' : isString ? 'string' : isNumber ? 'number' : null;
      if (word) kind = keywords.has(word) ? 'keyword' : CONSTANTS.has(word) ? 'constant' : /^\s*\(/.test(code.slice(token.lastIndex)) ? 'function' : null;
      fragment.append(kind ? el('span', { class: 'tok-' + kind }, text) : text);
      last = token.lastIndex;
    }
    fragment.append(code.slice(last));
    return fragment;
  }

  function codeBlock(language, code) {
    const lang = ALIASES[language] || language || 'text';
    const action = (name, title) => el('button', { title, onclick: () => vscode.postMessage({ type: name, code, language: lang }) },
      icon(name), name === 'copy' ? 'Copy' : 'Insert');
    return el('div', { class: 'code' },
      el('div', { class: 'code-head' }, el('span', { class: `lang lang-${lang}` }, lang), el('span', { class: 'spacer' }),
        action('copy', 'Copy to clipboard'), action('insert', 'Insert at the cursor in the last active editor')),
      el('pre', {}, el('code', {}, highlight(code, lang))));
  }

  function renderContent(container, text, streaming) {
    container.replaceChildren();
    const fence = /```([\w+#.-]*)[^\n]*\n?([\s\S]*?)(?:```|$)/g;
    let last = 0;
    let match;
    const prose = chunk => {
      for (const paragraph of chunk.split(/\n{2,}/)) {
        if (paragraph.trim()) container.append(el('p', {}, inline(paragraph.replace(/^\n+|\n+$/g, ''))));
      }
    };
    while ((match = fence.exec(text)) && match[0]) {
      prose(text.slice(last, match.index));
      container.append(codeBlock(match[1].toLowerCase(), match[2].replace(/\n$/, '')));
      last = fence.lastIndex;
    }
    prose(text.slice(last));
    if (streaming) {
      if (!text) container.append(el('div', { class: 'thinking' }, `${BRAND} is thinking…`));
      else (container.lastElementChild && container.lastElementChild.tagName === 'P' ? container.lastElementChild : container).append(el('span', { class: 'caret' }));
    }
  }

  function turn(message) {
    if (message.role === 'user') return el('div', { class: 'turn user' }, el('div', { class: 'bubble' }, message.content));
    const content = el('div', { class: 'content' });
    renderContent(content, message.content, message.streaming);
    const helper = message.source && message.source !== 'model';
    const head = el('div', { class: 'speaker' }, el('span', { class: 'mark' }), el('span', { class: 'name' }, BRAND),
      helper ? el('span', { class: 'pill helper' }, `${SOURCES[message.source] || message.source} · local helper, not the model`) : null);
    const notes = [message.omitted ? `Earlier turns were left out to fit the loaded model’s context.` : null, message.note]
      .filter(Boolean).map(text => el('div', { class: 'note' }, text));
    const error = message.error ? el('div', { class: 'note error' }, message.error) : null;
    const node = el('div', { class: 'turn assistant' }, head, content, ...notes, error);
    if (message.streaming) live = { message, content };
    return node;
  }

  function empty() {
    return el('div', { class: 'empty' },
      el('img', {class:'empty-logo', src:document.body.dataset.icon, alt:'Kani Seal 3 logo'}),
      el('pre', { class: 'wordmark', 'aria-label': BRAND }, WORDMARK),
      VERSION ? el('div', { class: 'tagline' }, el('span', { class: 'version' }, VERSION), 'trained from scratch · runs on this computer') : null,
      el('p', {}, 'Knows practiced chat replies, 20 coding concepts, and short Python and JavaScript snippets.'),
      el('div', { class: 'suggestions' }, ...SUGGESTIONS.map(([kind, text]) =>
        el('button', { class: 'suggestion', onclick: () => submit(text) }, el('span', { class: `tag tag-${kind}` }, kind), el('span', {}, text)))));
  }

  function render() {
    live = null;
    log.replaceChildren(...(messages.length ? messages.map(turn) : [empty()]));
    scroll(true);
    controls();
  }

  function scroll(force) {
    if (force || log.scrollHeight - log.scrollTop - log.clientHeight < 80) log.scrollTop = log.scrollHeight;
  }

  function update() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      if (!live) return;
      renderContent(live.content, live.message.content, live.message.streaming);
      scroll(false);
    });
  }

  function controls() {
    const ready = status.state === 'ready';
    sendButton.replaceChildren(icon(busy ? 'stop' : 'send'));
    sendButton.classList.toggle('stop', Boolean(busy));
    sendButton.title = busy ? 'Stop (Escape)' : 'Send (Enter)';
    sendButton.setAttribute('aria-label', busy ? 'Stop' : 'Send');
    sendButton.disabled = !busy && (!ready || attachments?.reading || (!input.value.trim() && !attachments?.files.length));
    input.placeholder = ready ? `Ask ${BRAND} anything…` : status.state === 'error' ? `${BRAND} is not running` : `Starting ${BRAND}…`;
    for (const button of log.querySelectorAll('.suggestion')) button.disabled = !ready || Boolean(busy);
  }

  function showStatus() {
    where.firstChild.className = 'dot ' + status.state;
    where.title = { ready: 'Running on this computer', starting: 'Starting the local model', error: 'Not running' }[status.state] || '';
    if (status.state === 'ready' && status.info) {
      const info = status.info;
      model.textContent = `${info.checkpoint} · ${Math.round(info.parameters / 1000)}K`;
      model.title = `${info.name} · ${info.parameters.toLocaleString()} parameters · ${info.steps.toLocaleString()} training steps. Change the model in Settings.`;
    } else {
      model.textContent = status.state === 'error' ? 'not running' : 'starting…';
    }
    banner.hidden = status.state !== 'error';
    const [problem, command] = (status.error || '').split(' Run: ');
    bannerText.replaceChildren(problem, ...(command ? [' Run: ', el('code', {}, command)] : []));
    controls();
  }

  // Conversation
  function history() {
    const turns = [];
    for (const message of messages) {
      if (message.excludeFromContext) continue;
      if (message.role === 'user') {
        if (turns.length && turns[turns.length - 1].role === 'user') turns.pop(); // its reply failed
        turns.push({ role: 'user', content: message.content });
      } else if (message.content && turns.length && turns[turns.length - 1].role === 'user') {
        turns.push({ role: 'assistant', content: message.content });
      }
    }
    return turns;
  }

  function submit(text) {
    let prompt = (text ?? input.value).trim();
    if ((!prompt && !attachments?.files.length) || busy || attachments?.reading || status.state !== 'ready') return;
    prompt ||= 'Analyze these files.';
    const attached=Boolean(attachments?.files.length);
    let fileTurn;
    try { if(attached) fileTurn=prepareAttachments(prompt,attachments.files,new Tokenizer(status.info.tokenizer_spec),status.info.config.context,512); }
    catch(error){fileError.textContent=error.message;fileError.hidden=false;return;}
    fileError.hidden=true;
    input.value = '';
    resize();
    busy = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now() + Math.random());
    messages.push({ role:'user', content:prompt+(attached?'\nAttached: '+attachments.files.map(f=>f.name).join(', '):''), excludeFromContext:attached });
    if(fileTurn?.helper){messages.push({role:'assistant',content:fileTurn.text,source:'file_analysis',excludeFromContext:true});busy=null;save();render();return;}
    const conversation = fileTurn?[{role:'user',content:fileTurn.text}]:history();
    attachments?.setBusy(true);
    messages.push({ role:'assistant', content:'', streaming:true, id:busy, excludeFromContext:attached, note:fileTurn?.receipt });
    render();
    vscode.postMessage({ type: 'send', id: busy, messages: conversation });
  }

  function stop() {
    if (busy) vscode.postMessage({ type: 'cancel', id: busy });
  }

  function reset() {
    stop();
    busy = null;
    messages = [];
    attachments?.clear(); attachments?.setBusy(false);fileError.hidden=true;
    save();
    render();
    input.focus();
  }

  function save() {
    vscode.setState({ messages: messages.map(({ id, ...message }) => message) });
  }

  function finish(message, changes) {
    Object.assign(message, changes);
    delete message.streaming;
    busy = null; attachments?.setBusy(false);
    save();
    render();
  }

  window.addEventListener('message', ({ data }) => {
    if (data.type === 'status') {
      status = data;
      showStatus();
      return;
    }
    if (data.type === 'reset') return reset();
    const message = messages.find(m => m.id === data.id);
    if (!message) return;
    if (data.type === 'start') {
      message.source = data.source;
      message.omitted = data.omitted;
      render();
    } else if (data.type === 'token') {
      message.content += data.text;
      update();
    } else if (data.type === 'done') {
      finish(message, data.cancelled ? { note: 'Stopped.' } : {});
    } else if (data.type === 'error') {
      finish(message, { error: data.message });
    }
  });

  // Composer
  function resize() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
  }
  input.addEventListener('input', () => { resize(); controls(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    } else if (event.key === 'Escape' && busy) {
      stop();
    }
  });

  render();
  showStatus();
  vscode.postMessage({ type: 'ready' });
})();
