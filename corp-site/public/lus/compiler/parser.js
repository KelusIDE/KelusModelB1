// Builds the syntax tree from tokens (recursive descent).
//
//   program   := (fn | typedef)*
//   typedef   := 'type' Name '{' (field ':' type)* '}'     fields end at ',' or a line break
//   fn        := 'fn' (Name '.')? name ('(' params ')')? ('->' type)? block      Name. makes a method, with `self`
//   type      := int | float | bool | str | Name | '[' type ']'
//   block     := '{' statement* '}'            statements end at a line break, ';' or '}'
//   statement := ('let' | 'var') name (':' type)? '=' expr
//              | place ('=' | '+=' | '-=' | '*=' | '/=' | '%=') expr       place := name, then any .field or [index]
//              | 'if' expr block ('else' (if | block))?
//              | 'while' expr block | 'repeat' expr block | 'for' name 'in' expr ('..' expr)? block
//              | 'break' | 'continue' | 'return' expr? | 'say' expr | expr
//   postfix   := primary ('.' name ('(' args ')')? | '[' expr ']')*
//   args      := ((name ':')? expr),*
import { KelError } from './lexer.js';

const LEVELS = [['||'], ['&&'], ['==', '!=', '<', '<=', '>', '>='], ['+', '-'], ['*', '/', '%']];
const COMPARISON = 2;
const COMPOUND = { '+=': '+', '-=': '-', '*=': '*', '/=': '/', '%=': '%' };
const BLOCK_HINT = "Kel uses { } for blocks, like: if x > 3 { ... }";

export function parse(tokens) {
  return parser(tokens).program();
}

function parser(tokens) {
  const typeUses = [];
  let pos = 0;
  const peek = (offset = 0) => tokens[Math.min(pos + offset, tokens.length - 1)];
  const at = (type) => peek().type === type;
  const next = () => tokens[pos++];
  const fail = (message, token = peek()) => { throw new KelError(message, token.line, token.col); };
  const expect = (type, message) => (at(type) ? next() : fail(message));
  const skipBreaks = () => { while (at('newline') || at(';')) next(); };

  /** A type as text: 'int', 'Point', '[int]', '[[str]]'. The checker makes sure that each one exists. */
  function type() {
    if (at('[')) {
      next();
      const inner = type();
      expect(']', "Expected ']' to finish the list type, like: [int]");
      return `[${inner}]`;
    }
    const token = expect('name', 'Expected a type: int, float, bool, str, a list like [int], or one of your own types');
    typeUses.push(token);
    return token.value;
  }

  function typedef() {
    const start = next();
    const name = expect('name', 'Give the type a name, like: type Point { x: float, y: float }');
    expect('{', "Expected '{' to start the type, like: type Point { x: float, y: float }");
    const fields = [];
    const skip = () => { while (at('newline') || at(',') || at(';')) next(); };
    skip();
    while (!at('}')) {
      if (at('eof')) fail("This type is missing its closing '}'", start);
      const field = expect('name', 'Expected a field name, like: x: float');
      expect(':', `Give '${field.value}' a type, like: ${field.value}: int`);
      fields.push({ name: field.value, type: type(), line: field.line, col: field.col });
      skip();
    }
    next();
    return { name: name.value, fields, line: name.line, col: name.col };
  }

  function fn() {
    const start = next();
    let name = expect('name', 'Give the function a name, like: fn main { }');
    let receiver = null;
    if (at('.')) { // fn Point.length -> float { ... }
      next();
      receiver = name;
      name = expect('name', `Give the method a name, like: fn ${receiver.value}.show { }`);
    }
    const params = receiver ? [{ name: 'self', type: receiver.value, line: receiver.line, col: receiver.col }] : [];
    if (receiver) typeUses.push(receiver);
    const first = params.length;
    if (at('(')) {
      next();
      while (!at(')')) {
        if (params.length > first) expect(',', "Separate parameters with ','");
        const param = expect('name', 'Expected a parameter name');
        expect(':', `Give '${param.value}' a type, like: ${param.value}: int`);
        params.push({ name: param.value, type: type(), line: param.line, col: param.col });
      }
      next();
    }
    let returns = 'void';
    if (at('->')) { next(); returns = type(); }
    return { name: receiver ? `${receiver.value}.${name.value}` : name.value, method: receiver ? name.value : null, receiver: receiver?.value ?? null,
      params, returns, body: block('function'), line: start.line, col: start.col };
  }

  function block(what) {
    while (at('newline')) next();
    if (at(':')) fail(BLOCK_HINT);
    const open = expect('{', `Expected '{' to start the ${what}`);
    const body = [];
    skipBreaks();
    while (!at('}')) {
      if (at('eof')) fail("This block is missing its closing '}'", open);
      body.push(statement());
      if (!at('}')) {
        if (at(':')) fail(BLOCK_HINT);
        if (!at('newline') && !at(';')) fail('Expected the end of the line here');
      }
      skipBreaks();
    }
    next();
    return body;
  }

  function statement() {
    const token = peek();
    const place = { line: token.line, col: token.col };
    if (at('let') || at('var')) {
      next();
      const name = expect('name', `Expected a name after '${token.type}'`);
      let declared = null;
      if (at(':')) { next(); declared = type(); }
      expect('=', `Give '${name.value}' a starting value, like: ${token.type} ${name.value} = 0`);
      return { kind: 'let', mutable: token.type === 'var', name: name.value, declared, value: expression(), ...place };
    }
    if (at('if')) return ifStatement();
    if (at('while')) { next(); return { kind: 'while', condition: expression(), body: block('while loop'), ...place }; }
    if (at('repeat')) { next(); return { kind: 'repeat', count: expression(), body: block('repeat loop'), ...place }; }
    if (at('for')) {
      next();
      const name = expect('name', 'Name the counter, like: for i in 1..10 { }');
      expect('in', "Expected 'in', like: for i in 1..10 { }");
      const from = expression();
      if (at('..')) { next(); return { kind: 'for', name: name.value, from, to: expression(), body: block('for loop'), ...place }; }
      return { kind: 'forEach', name: name.value, list: from, body: block('for loop'), ...place };
    }
    if (at('return')) {
      next();
      const value = at('newline') || at(';') || at('}') ? null : expression();
      return { kind: 'return', value, ...place };
    }
    if (at('break') || at('continue')) { next(); return { kind: token.type, ...place }; }
    if (at('say')) { next(); return { kind: 'say', value: expression(), ...place }; }
    if (at('name') && (peek(1).type === '=' || peek(1).type in COMPOUND)) {
      const name = next();
      const op = next().type;
      const value = expression();
      const target = { kind: 'name', name: name.value, line: name.line, col: name.col };
      return { kind: 'assign', name: name.value, value: op === '=' ? value : { kind: 'binary', op: COMPOUND[op], left: target, right: value, ...place }, ...place };
    }
    const value = expression();
    if (at('=') || peek().type in COMPOUND) { // xs[0] = 1, p.x += 2
      if (value.kind !== 'field' && value.kind !== 'index') fail('Only a variable, a field or a list item can be set');
      const op = next().type;
      const right = expression();
      return { kind: 'set', target: value, value: op === '=' ? right : { kind: 'binary', op: COMPOUND[op], left: value, right, compound: true, ...place }, ...place };
    }
    return { kind: 'expr', value, ...place };
  }

  function ifStatement() {
    const token = next();
    const condition = expression();
    const then = block('if block');
    let otherwise = [];
    let ahead = 0;
    while (peek(ahead).type === 'newline') ahead++; // allow `}` and `else` on separate lines
    if (peek(ahead).type === 'else') {
      pos += ahead + 1;
      otherwise = at('if') ? [ifStatement()] : block('else block');
    }
    return { kind: 'if', condition, then, otherwise, line: token.line, col: token.col };
  }

  function expression(level = 0) {
    if (level === LEVELS.length) return unary();
    let left = expression(level + 1);
    while (LEVELS[level].includes(peek().type)) {
      const op = next();
      const right = expression(level + 1);
      left = { kind: 'binary', op: op.type, left, right, line: op.line, col: op.col };
      if (level === COMPARISON && LEVELS[level].includes(peek().type)) fail('Compare two values at a time, like: a < b && b < c');
    }
    return left;
  }

  function unary() {
    if (at('-') || at('!')) {
      const op = next();
      return { kind: 'unary', op: op.type, value: unary(), line: op.line, col: op.col };
    }
    return postfix();
  }

  function postfix() {
    let value = primary();
    for (;;) {
      const token = peek();
      const place = { line: token.line, col: token.col };
      if (at('.')) {
        next();
        const name = expect('name', 'Expected a name after the dot, like: point.x or list.add(1)');
        if (at('(')) {
          const { args, labels } = arguments_();
          value = { kind: 'method', object: value, name: name.value, args, labels, line: name.line, col: name.col };
        } else value = { kind: 'field', object: value, name: name.value, line: name.line, col: name.col };
      } else if (at('[')) {
        next();
        const index = expression();
        expect(']', "Expected ']' here, like: list[0]");
        value = { kind: 'index', object: value, index, ...place };
      } else return value;
    }
  }

  /** ( a, b ) or ( x: a, y: b ). `labels[i]` is null when a value has no name. */
  function arguments_() {
    next();
    const args = [], labels = [];
    while (!at(')')) {
      if (args.length) expect(',', "Separate values with ','");
      if (at('name') && peek(1).type === ':') { labels.push(next().value); next(); } else labels.push(null);
      args.push(expression());
    }
    next();
    return { args, labels };
  }

  function primary() {
    const token = peek();
    const place = { line: token.line, col: token.col };
    if (at('int') || at('float')) { next(); return { kind: token.type, value: token.value, ...place }; }
    if (at('string')) {
      next();
      if (token.parts.length === 1 && typeof token.parts[0] === 'string') return { kind: 'string', value: token.parts[0], ...place };
      const parts = token.parts.map(part => (typeof part === 'string' ? { kind: 'string', value: part, ...place } : parser(part.tokens).templatePart()));
      return { kind: 'template', parts, ...place };
    }
    if (at('true') || at('false')) { next(); return { kind: 'bool', value: token.type === 'true', ...place }; }
    if (at('name')) {
      next();
      if (!at('(')) return { kind: 'name', name: token.value, ...place };
      const { args, labels } = arguments_();
      return { kind: 'call', name: token.value, args, labels, ...place };
    }
    if (at('[')) {
      next();
      const items = [];
      while (!at(']')) {
        if (items.length) expect(',', "Separate list items with ','");
        if (at(']')) break; // a comma after the last item is fine
        items.push(expression());
      }
      next();
      return { kind: 'list', items, ...place };
    }
    if (at('(')) {
      next();
      const inner = expression();
      expect(')', "Expected ')' here");
      return inner;
    }
    return fail('Expected a value here, like a number, a name or "text"');
  }

  return {
    program() {
      const functions = [], types = [];
      skipBreaks();
      while (!at('eof')) {
        if (at('name') && peek().value === 'type') types.push(typedef());
        else if (at('fn')) functions.push(fn());
        else fail('Code must be inside a function. Start with: fn main { }');
        skipBreaks();
      }
      return { functions, types, typeUses };
    },
    /** One {expression} inside "text". */
    templatePart() {
      const value = expression();
      if (!at('eof')) fail("Only one value fits inside { } in text");
      return value;
    }
  };
}
