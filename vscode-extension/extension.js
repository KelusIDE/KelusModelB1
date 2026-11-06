'use strict';
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const vscode = require('vscode');
const { Bridge } = require('./bridge');

const LANGUAGES = { py: 'python', python: 'python', js: 'javascript', javascript: 'javascript', ts: 'typescript',
  typescript: 'typescript', html: 'html', css: 'css', json: 'json', sh: 'shellscript', bash: 'shellscript', zsh: 'shellscript' };

function settings() {
  return vscode.workspace.getConfiguration('kani');
}

function repository() {
  const folders = (vscode.workspace.workspaceFolders || []).map(folder => folder.uri.fsPath);
  const candidates = [settings().get('repositoryPath'), process.env.KELUS_B1_PATH, ...folders].filter(Boolean);
  return candidates.find(dir => fs.existsSync(path.join(dir, 'kelus', 'cli.py')));
}

function python() {
  return settings().get('pythonPath') || process.env.KELUS_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
}

class Host {
  constructor(context) {
    this.context = context;
    this.name = context.extension.packageJSON.displayName;
    this.views = new Set();
    this.owners = new Map();
    this.searches = new Map();
    this.holds = new Map();
    this.bridge = null;
    this.lastEditor = vscode.window.activeTextEditor;
    context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(editor => { if (editor) this.lastEditor = editor; }));
  }

  builds() {
    const cwd = repository();
    if (!cwd) return [];
    const builds = JSON.parse(fs.readFileSync(path.join(cwd, 'kani-builds.json'), 'utf8'));
    return builds.filter(b => fs.existsSync(path.join(cwd, 'checkpoints', b.checkpoint + '.npz')));
  }

  selectedBuild() {
    return this.builds().find(b => b.id === settings().get('build', 'latest'));
  }

  ensureBridge() {
    if (this.bridge) return;
    const cwd = repository();
    if (!cwd) {
      this.status = { state: 'error', error: 'Open the KelusModelB1 folder, or set "Kani: Repository Path" in Settings.', setup: true };
      return this.broadcast({ type: 'status', ...this.status });
    }
    const build = this.selectedBuild();
    this.bridge = new Bridge({ python: python(), cwd, model: build ? build.checkpoint : settings().get('model') });
    this.bridge.on('status', status => {
      this.status = status;
      this.broadcast({ type: 'status', ...status, setup: status.state === 'error' });
    });
    this.bridge.on('event', event => {
      const view = this.owners.get(event.id);
      if (!view || this.hold(view, event)) return;
      if (event.type === 'done' || event.type === 'error') this.owners.delete(event.id);
      view.postMessage(event);
    });
    this.bridge.start();
  }

  restart() {
    if (this.bridge) this.bridge.stop();
    this.bridge = null;
    for (const [id, view] of this.owners) view.postMessage({ type: 'error', id, message: 'The model restarted.' });
    this.owners.clear();
    this.ensureBridge();
  }

  broadcast(message) {
    for (const view of this.views) view.postMessage(message);
  }

  attach(webview, owner) {
    const media = vscode.Uri.joinPath(this.context.extensionUri, 'media');
    const shared = vscode.Uri.file(path.join(repository() || path.dirname(this.context.extensionPath), 'site'));
    webview.options = { enableScripts: true, localResourceRoots: [media, shared] };
    webview.html = this.html(webview, media, shared);
    this.views.add(webview);
    owner.onDidDispose(() => this.views.delete(webview));
    webview.onDidReceiveMessage(message => this.receive(webview, message));
    this.ensureBridge();
  }

  receive(webview, message) {
    if (message.type === 'ready') {
      webview.postMessage({ type: 'builds', builds: this.builds(), selected: settings().get('build', 'latest') });
      if (this.status) webview.postMessage({ type: 'status', ...this.status, setup: this.status.state === 'error' });
    } else if (message.type === 'selectBuild') {
      if (this.builds().some(b => b.id === message.build)) {
        settings().update('build', message.build, vscode.ConfigurationTarget.Global).then(() => {
          this.broadcast({ type: 'reset' });
          this.broadcast({ type: 'builds', builds: this.builds(), selected: message.build });
        }, error => vscode.window.showErrorMessage(error.message));
      }
    } else if (message.type === 'send') {
      this.send(webview, message);
    } else if (message.type === 'cancel') {
      if (this.searches.has(message.id)) this.searches.get(message.id).abort();
      else if (this.bridge) this.bridge.cancel(message.id);
    } else if (message.type === 'insert') {
      this.insert(message.code, message.language);
    } else if (message.type === 'copy') {
      vscode.env.clipboard.writeText(message.code);
      vscode.window.setStatusBarMessage('Copied Kani code', 2000);
    } else if (message.type === 'settings') {
      vscode.commands.executeCommand('workbench.action.openSettings', 'kani.');
    } else if (message.type === 'restart') {
      this.restart();
    } else if (message.type === 'openTab') {
      this.openTab();
    }
  }

  async send(webview, message) {
    const last = message.messages[message.messages.length - 1];
    const cwd = repository();

    const web = cwd && last && last.role === 'user' ? await import(pathToFileURL(path.join(cwd, 'site', 'web.js')).href).catch(() => null) : null;
    const request = web && web.webRequest(last.content);
    if (request) {
      const controller = new AbortController();
      this.searches.set(message.id, controller);
      webview.postMessage({ type: 'start', id: message.id, source: 'web', omitted: 0 });
      try {
        const found = await web.webAnswer(request, fetch, controller.signal);
        webview.postMessage({ type: 'token', id: message.id, text: found.text });
        webview.postMessage({ type: 'done', id: message.id, cancelled: false });
      } catch (error) {
        webview.postMessage({ type: 'error', id: message.id, message: error.message });
      } finally {
        this.searches.delete(message.id);
      }
      return;
    }
    const options = { maxTokens: settings().get('maxTokens'), cideMaxTokens: settings().get('cideMaxTokens'), temperature: settings().get('temperature'), imageModel: this.selectedBuild()?.imageModel || 'painter' };
    const reading = cwd ? await import(pathToFileURL(path.join(cwd, 'site', 'reading.js')).href).catch(() => null) : null;
    const places = cwd ? await import(pathToFileURL(path.join(cwd, 'site', 'places.js')).href).catch(() => null) : null;
    if (this.selectedBuild()?.id !== 'cide' && reading && web && last && last.role === 'user') this.holds.set(message.id, { text: '', question: last.content, reading, web, places });
    if (this.bridge && this.bridge.chat(message.id, message.messages.slice(-15), options)) this.owners.set(message.id, webview);
    else webview.postMessage({ type: 'error', id: message.id, message: 'Kani is not running. Check the message above, then Restart.' });
  }






  hold(view, event) {
    const hold = this.holds.get(event.id);
    if (!hold) return false;
    if (event.type === 'start' && event.source !== 'model') this.holds.delete(event.id);
    if (event.type === 'error') this.holds.delete(event.id);
    if (event.type !== 'token' && event.type !== 'done') return false;
    if (event.type === 'token') {
      hold.text += event.text;
      if (hold.reading.maybeSearch(hold.text) || hold.places?.maybePlaces(hold.text)) return true;
      this.holds.delete(event.id);
      view.postMessage({ ...event, text: hold.text });
      return true;
    }
    this.holds.delete(event.id);
    const place = hold.places && hold.text.trim().match(hold.places.PLACES_CALL);
    if (place) {
      this.placed(view, event.id, place, hold.places);
      return true;
    }
    const call = hold.text.trim().match(hold.reading.SEARCH_CALL);
    if (!call) {
      if (hold.text) view.postMessage({ type: 'token', id: event.id, text: hold.text });
      return false;
    }
    this.searched(view, event.id, call[1], hold);
    return true;
  }

  async searched(view, id, query, { question, reading, web }) {
    const controller = new AbortController();
    this.searches.set(id, controller);
    view.postMessage({ type: 'start', id, source: 'web_answer', omitted: 0 });
    const generate = messages => this.generate(messages);
    try {
      for await (const text of reading.searchedAnswer(query, question, { generate, webAnswer: web.webAnswer, signal: controller.signal })) {
        if (controller.signal.aborted) break;
        view.postMessage({ type: 'token', id, text });
      }
      view.postMessage({ type: 'done', id, cancelled: controller.signal.aborted });
    } catch (error) {
      view.postMessage({ type: 'error', id, message: error.message });
    } finally {
      this.searches.delete(id);
      this.owners.delete(id);
    }
  }


  async placed(view, id, call, places) {
    const controller = new AbortController();
    this.searches.set(id, controller);
    view.postMessage({ type: 'start', id, source: 'places', omitted: 0 });
    try {
      for await (const text of places.placesAnswer(call, { locate: async () => null, signal: controller.signal })) {
        if (controller.signal.aborted) break;
        view.postMessage({ type: 'token', id, text });
      }
      view.postMessage({ type: 'done', id, cancelled: controller.signal.aborted });
    } catch (error) {
      view.postMessage({ type: 'error', id, message: error.message });
    } finally {
      this.searches.delete(id);
      this.owners.delete(id);
    }
  }


  generate(messages) {
    const bridge = this.bridge, id = crypto.randomUUID();
    return {
      async *[Symbol.asyncIterator]() {
        const queue = [];
        let wake = null, finished = false, failure = null;
        const listener = event => {
          if (event.id !== id) return;
          if (event.type === 'token') queue.push(event.text);
          else if (event.type === 'done') finished = true;
          else if (event.type === 'error') { failure = new Error(event.message); finished = true; }
          if (wake) wake();
        };
        bridge.on('event', listener);
        try {
          if (!bridge.chat(id, messages, { maxTokens: 200, temperature: 0, reading: true })) throw new Error('Kani is not running.');
          for (;;) {
            while (queue.length) yield queue.shift();
            if (finished) break;
            await new Promise(resolve => { wake = resolve; });
            wake = null;
          }
          if (failure) throw failure;
        } finally {
          bridge.off('event', listener);
        }
      },
    };
  }

  async insert(code, language) {
    const editor = this.lastEditor && !this.lastEditor.document.isClosed ? this.lastEditor : null;
    if (!editor) {
      const document = await vscode.workspace.openTextDocument({ content: code, language: LANGUAGES[language] || 'plaintext' });
      return vscode.window.showTextDocument(document);
    }

    await editor.edit(builder => editor.selections.forEach(selection => builder.replace(selection, code)));
    vscode.window.showTextDocument(editor.document, editor.viewColumn);
  }

  html(webview, media, shared) {
    const nonce = crypto.randomBytes(16).toString('base64');
    const uri = name => webview.asWebviewUri(vscode.Uri.joinPath(media, name));
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} blob:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}' ${webview.cspSource};">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="${uri('chat.css')}">
<title>${this.name}</title>
</head>
<body data-icon="${uri('icon.png')}" data-name="${this.name}" data-codename="Kani B2:10F" data-attachments="${webview.asWebviewUri(vscode.Uri.joinPath(shared, 'attachments.js'))}" data-engine="${webview.asWebviewUri(vscode.Uri.joinPath(shared, 'b1.js'))}">
<script type="module" nonce="${nonce}" src="${uri('chat.js')}"></script>
</body>
</html>`;
  }

  openTab() {
    const panel = vscode.window.createWebviewPanel('kani.chatTab', this.name, vscode.ViewColumn.Beside, { retainContextWhenHidden: true });
    panel.iconPath = vscode.Uri.joinPath(this.context.extensionUri, 'media', 'activity.svg');
    this.attach(panel.webview, panel);
  }

  dispose() {
    if (this.bridge) this.bridge.stop();
  }
}

function activate(context) {
  const host = new Host(context);
  let sidebar = null;
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('kani.chat', {
      resolveWebviewView(view) {
        sidebar = view.webview;
        host.attach(view.webview, view);
      },
    }, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.commands.registerCommand('kani.newChat', () => sidebar && sidebar.postMessage({ type: 'reset' })),
    vscode.commands.registerCommand('kani.openInTab', () => host.openTab()),
    vscode.commands.registerCommand('kani.restart', () => host.restart()),
    vscode.workspace.onDidChangeConfiguration(event => { if (event.affectsConfiguration('kani')) host.restart(); }),
    vscode.workspace.onDidChangeWorkspaceFolders(() => { if (!host.bridge) host.restart(); }),
    { dispose: () => host.dispose() },
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
