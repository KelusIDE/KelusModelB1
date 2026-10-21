'use strict';
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
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
    this.owners = new Map(); // request id -> webview that asked
    this.bridge = null;
    this.lastEditor = vscode.window.activeTextEditor;
    context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(editor => { if (editor) this.lastEditor = editor; }));
  }

  ensureBridge() {
    if (this.bridge) return;
    const cwd = repository();
    if (!cwd) {
      this.status = { state: 'error', error: 'Open the KelusModelB1 folder, or set "Kani: Repository Path" in Settings.', setup: true };
      return this.broadcast({ type: 'status', ...this.status });
    }
    this.bridge = new Bridge({ python: python(), cwd, model: settings().get('model') });
    this.bridge.on('status', status => {
      this.status = status;
      this.broadcast({ type: 'status', ...status, setup: status.state === 'error' });
    });
    this.bridge.on('event', event => {
      const view = this.owners.get(event.id);
      if (!view) return;
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
      if (this.status) webview.postMessage({ type: 'status', ...this.status, setup: this.status.state === 'error' });
    } else if (message.type === 'send') {
      const options = { maxTokens: settings().get('maxTokens'), temperature: settings().get('temperature') };
      if (this.bridge && this.bridge.chat(message.id, message.messages.slice(-15), options)) this.owners.set(message.id, webview);
      else webview.postMessage({ type: 'error', id: message.id, message: 'Kani is not running. Check the message above, then Restart.' });
    } else if (message.type === 'cancel') {
      if (this.bridge) this.bridge.cancel(message.id);
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

  async insert(code, language) {
    const editor = this.lastEditor && !this.lastEditor.document.isClosed ? this.lastEditor : null;
    if (!editor) {
      const document = await vscode.workspace.openTextDocument({ content: code, language: LANGUAGES[language] || 'plaintext' });
      return vscode.window.showTextDocument(document);
    }
    // One edit call is one undo step. Inserted code is never saved or run.
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
