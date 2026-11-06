


const ESC = '\x1b[';
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

export const CRAB = [
  '  ▟▙        ▟▙',
  ' ▐▌▐▌      ▐▌▐▌',
  '  ▜▛▄▄████▄▄▜▛',
  '    ██ ▀  ▀ ██',
  '    ▀████████▀',
  '    ▞ ▞ ▞▚ ▚ ▚',
];
export const WORDMARK = [
  '██╗  ██╗ █████╗ ███╗   ██╗██╗',
  '██║ ██╔╝██╔══██╗████╗  ██║██║',
  '█████╔╝ ███████║██╔██╗ ██║██║',
  '██╔═██╗ ██╔══██║██║╚██╗██║██║',
  '██║  ██╗██║  ██║██║ ╚████║██║',
  '╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝',
];
export const SPINNER = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';

const DARK = {
  mint: [92, 201, 154], teal: [63, 182, 201], coral: [255, 122, 89], gold: [229, 180, 84], blue: [106, 168, 240],
  muted: [128, 138, 133], red: [241, 76, 76],
  keyword: [197, 134, 192], string: [206, 145, 120], number: [181, 206, 168], constant: [86, 156, 214], function: [220, 220, 170], comment: [106, 153, 85],
};
const LIGHT = {
  mint: [31, 138, 91], teal: [27, 127, 154], coral: [217, 72, 31], gold: [154, 103, 0], blue: [11, 99, 201],
  muted: [105, 113, 109], red: [205, 49, 49],
  keyword: [175, 0, 219], string: [163, 21, 21], number: [9, 134, 88], constant: [0, 0, 255], function: [121, 94, 38], comment: [0, 128, 0],
};
const LANGUAGE_COLORS = { python: 'blue', javascript: 'gold', html: 'coral', css: 'teal' };
const KEYWORDS = {
  python: new Set('and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield'.split(' ')),
  javascript: new Set('async await break case catch class const continue default delete do else export extends finally for function if import in instanceof let new of return static super switch this throw try typeof var void while yield'.split(' ')),
};
const CONSTANTS = new Set('True False None true false null undefined NaN Infinity self'.split(' '));
const ALIASES = { py: 'python', js: 'javascript', ts: 'javascript', typescript: 'javascript', jsx: 'javascript' };


export function colorMode(env = process.env, stream = process.stdout) {
  if ('NO_COLOR' in env) return 0;
  if (env.FORCE_COLOR) return 3;
  if (!stream.isTTY) return 0;
  if (/truecolor|24bit/i.test(env.COLORTERM || '') || ['vscode', 'iTerm.app', 'WezTerm', 'ghostty'].includes(env.TERM_PROGRAM)) return 3;
  return 2;
}


export function lightBackground(env = process.env) {
  const parts = (env.COLORFGBG || '').split(';');
  return parts.length > 1 && ['7', '15'].includes(parts[parts.length - 1]);
}

export function width(text) {
  return [...text.replace(ANSI, '')].length;
}

export function theme({ mode = colorMode(), light = lightBackground() } = {}) {
  const palette = light ? LIGHT : DARK;
  const to256 = ([r, g, b]) => 16 + 36 * Math.round(r / 51) + 6 * Math.round(g / 51) + Math.round(b / 51);
  const fg = rgb => (mode === 3 ? `${ESC}38;2;${rgb.join(';')}m` : `${ESC}38;5;${to256(rgb)}m`);
  const mix = (a, b, f) => a.map((value, i) => Math.round(value + (b[i] - value) * f));
  const t = {
    mode,
    color: (name, text) => (mode && text ? fg(palette[name]) + text + `${ESC}39m` : text),
    bold: text => (mode && text ? `${ESC}1m${text}${ESC}22m` : text),

    gradient(text, stops = ['mint', 'teal', 'coral'], span = width(text)) {
      if (!mode) return text;
      const colors = stops.map(name => palette[name]);
      return [...text].map((char, i) => {
        if (char === ' ') return char;
        const position = span > 1 ? (i / (span - 1)) * (colors.length - 1) : 0;
        const index = Math.min(Math.floor(position), colors.length - 2);
        return fg(mix(colors[index], colors[index + 1], position - index)) + char;
      }).join('') + `${ESC}39m`;
    },
  };
  return t;
}

export function fit(text, size) {
  const chars = [...text];
  return chars.length <= size ? text : chars.slice(0, Math.max(0, size - 1)).join('') + '…';
}

export function banner(t, columns) {
  if (columns >= 54) return CRAB.map((line, i) => '  ' + t.color('coral', line.padEnd(16)) + '  ' + t.gradient(WORDMARK[i])).join('\n');
  if (columns >= 33) return WORDMARK.map(line => '  ' + t.gradient(line)).join('\n');
  return '  ' + t.gradient('KANI');
}


export function box(t, lines, size, title = '') {
  const inner = size - 4;
  const head = title ? `─ ${title} ` : '';
  const top = t.gradient('╭' + head.replace(title, '\0') + '─'.repeat(Math.max(0, size - 2 - width(head))) + '╮', undefined, size)
    .replace('\0', t.bold(title));
  const bottom = t.gradient('╰' + '─'.repeat(size - 2) + '╯');
  const body = lines.map(line => t.color('mint', '│') + ' ' + line + ' '.repeat(Math.max(0, inner - width(line))) + ' ' + t.color('coral', '│'));
  return [top, ...body, bottom].join('\n');
}

export function highlight(t, code, language) {
  const keywords = KEYWORDS[language] || new Set();
  const comment = language === 'python' ? '#.*' : '\\/\\/.*|\\/\\*.*?\\*\\/';
  const token = new RegExp(`(${comment})|("(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|\`(?:\\\\.|[^\`\\\\])*\`)|(\\b\\d+(?:\\.\\d+)?\\b)|([A-Za-z_$][\\w$]*)`, 'g');
  return code.replace(token, (text, isComment, isString, isNumber, word, offset) => {
    let kind = isComment ? 'comment' : isString ? 'string' : isNumber ? 'number' : null;
    if (word) kind = keywords.has(word) ? 'keyword' : CONSTANTS.has(word) ? 'constant' : /^\s*\(/.test(code.slice(offset + text.length)) ? 'function' : null;
    return kind ? t.color(kind, text) : text;
  });
}





export class Renderer {
  constructor(t, columns, indent = 2) {
    Object.assign(this, { t, columns: Math.max(24, columns), pad: ' '.repeat(indent), indent });
    Object.assign(this, { mode: 'prose', line: '', word: '', held: false, inline: false, col: indent, language: '' });
  }

  begin() { return this.pad; }

  push(text) {
    let out = '';
    for (const char of text) out += this.char(char);
    return out;
  }

  char(char) {
    if (char === '\n') return this.newline();
    this.line += char;
    if (this.mode === 'code') return '';
    if (this.line === '`') this.held = true;
    return this.held ? '' : this.prose(char);
  }

  prose(char) {
    if (char !== ' ') {
      this.word += char;
      return '';
    }
    const out = this.flush();
    if (this.col >= this.columns) return out + this.lineBreak();
    if (this.col === this.indent) return out;
    this.col += 1;
    return out + ' ';
  }

  flush() {
    if (!this.word) return '';
    const parts = this.word.split('`');
    const size = [...parts.join('')].length;
    let out = '';
    if (this.col + size > this.columns && this.col > this.indent) out += this.lineBreak();
    parts.forEach((part, i) => {
      if (i > 0) this.inline = !this.inline;
      out += this.inline ? this.t.color('teal', part) : part;
    });
    this.col += size;
    this.word = '';
    return out;
  }

  lineBreak() {
    this.col = this.indent;
    return '\n' + this.pad;
  }

  newline() {
    const line = this.line;
    this.line = '';
    if (this.mode === 'code') {
      if (/^\s*```/.test(line)) {
        this.mode = 'prose';
        return this.pad + this.t.color('muted', '╰─') + this.lineBreak();
      }
      return this.codeLine(line) + '\n';
    }
    if (this.held) {
      this.held = false;
      const fence = line.match(/^```\s*([\w+#.-]*)/);
      if (fence) return this.open(fence[1]);
      return this.replay(line) + this.lineBreak();
    }
    return this.flush() + this.lineBreak();
  }

  replay(line) {
    let out = '';
    for (const char of line) out += this.prose(char);
    return out + this.flush();
  }

  open(language) {
    this.mode = 'code';
    this.language = ALIASES[language.toLowerCase()] || language.toLowerCase();
    const label = this.language || 'code';
    const rule = '─'.repeat(Math.max(2, Math.min(this.columns, 60) - this.indent - label.length - 4));
    const start = this.col > this.indent ? this.flush() + this.lineBreak() : this.flush();
    this.col = this.indent;
    return start + this.t.color('muted', '╭─ ') + this.t.color(LANGUAGE_COLORS[this.language] || 'teal', this.t.bold(label)) +
      ' ' + this.t.color('muted', rule) + '\n';
  }

  codeLine(line) {
    return this.pad + this.t.color('muted', '│ ') + highlight(this.t, line, this.language);
  }


  end() {
    const line = this.line;
    this.line = '';
    if (this.mode === 'code') {
      this.mode = 'prose';
      const body = line && !/^\s*```/.test(line) ? this.codeLine(line) + '\n' : '';
      return body + this.pad + this.t.color('muted', '╰─');
    }
    if (this.held) {
      this.held = false;
      return /^```/.test(line) ? '' : this.replay(line);
    }
    return this.flush();
  }
}





export function inputArea(t, { text, cursor, size, placeholder, status, suggestions = [] }) {
  const room = size - 6;
  const chars = [...text];
  const rows = Math.max(1, Math.floor(chars.length / room) + 1);
  const lines = [];
  for (let row = 0; row < rows; row++) {
    const chunk = chars.slice(row * room, (row + 1) * room).join('');
    const prompt = row === 0 ? t.color('mint', '›') + ' ' : '  ';
    const shown = !text && row === 0 ? t.color('muted', fit(placeholder, room)) : chunk;
    lines.push(prompt + shown);
  }
  const area = box(t, lines, size).split('\n');
  const keys = [['⏎', 'send'], ['/', 'commands'], ['↑', 'history'], ['esc', 'stop'], ['ctrl+c', 'quit']];
  let left = '  ' + keys.map(([key, action]) => t.color('teal', key) + ' ' + t.color('muted', action)).join(t.color('muted', ' · '));
  if (width(left) + width(status) + 2 > size) left = '  ' + t.color('muted', '/ for commands');
  const gap = Math.max(1, size - width(left) - width(status));
  area.push(width(left) + width(status) < size ? left + ' '.repeat(gap) + status : left);
  for (const [name, description] of suggestions) area.push('    ' + t.color('coral', name.padEnd(10)) + t.color('muted', description));
  return { lines: area, row: 1 + Math.floor(cursor / room), column: 4 + (cursor % room) };
}
