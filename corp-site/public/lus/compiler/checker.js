// Checks types before any code runs, so mistakes show up with a clear message instead of a crash.
// It also annotates the tree for the code generator: every expression gets `type`,
// every variable gets a `local` slot, and every function gets its `locals` list.
import { KelError } from './lexer.js';

const MAX_INT = 2n ** 63n - 1n;
const BUILTINS = new Set(['int', 'float', 'str', 'len']);
const SIMPLE = new Set(['int', 'float', 'bool', 'str']);
const isList = (type) => type.startsWith('[');
const itemOf = (type) => type.slice(1, -1);
const isShared = (type) => isList(type) || !SIMPLE.has(type); // lists and your own types live on the heap
// Methods that change a list, so its variable must be var.
const CHANGES_LIST = new Set(['add', 'pop', 'remove', 'sort', 'reverse']);
const TEXT_METHODS = {
  upper: [[], 'str'], lower: [[], 'str'], trim: [[], 'str'], contains: [['str'], 'bool'], starts_with: [['str'], 'bool'], ends_with: [['str'], 'bool'],
  replace: [['str', 'str'], 'str'], split: [['str'], '[str]'], slice: [['int', 'int'], 'str']
};

export function check(program) {
  const functions = new Map();
  const types = new Map();
  for (const type of program.types) {
    if (!/^[A-Z]/.test(type.name)) throw new KelError(`Type names start with a capital letter, like ${type.name[0].toUpperCase()}${type.name.slice(1)}`, type.line, type.col);
    if (types.has(type.name)) throw new KelError(`There are two types named '${type.name}'`, type.line, type.col);
    if (!type.fields.length) throw new KelError(`Give ${type.name} at least one field, like: type ${type.name} { x: int }`, type.line, type.col);
    const seen = new Set();
    for (const field of type.fields) {
      if (seen.has(field.name)) throw new KelError(`${type.name} has two fields named '${field.name}'`, field.line, field.col);
      seen.add(field.name);
    }
    types.set(type.name, type);
  }
  program.typeTable = types;
  for (const use of program.typeUses) {
    if (!SIMPLE.has(use.value) && !types.has(use.value)) {
      throw new KelError(`Unknown type '${use.value}'. Kel has int, float, bool, str, lists like [int], and types you make with: type ${use.value} { ... }`, use.line, use.col);
    }
  }
  // A type can't hold itself directly (there would be no way to build the first one), but it can hold a list of itself.
  const holds = (name, seen = []) => {
    if (seen.includes(name)) throw new KelError(`${name} can't hold itself. A list works: children: [${name}]`, types.get(name).line, types.get(name).col);
    for (const field of types.get(name).fields) if (types.has(field.type)) holds(field.type, [...seen, name]);
  };
  for (const name of types.keys()) holds(name);
  for (const fn of program.functions) {
    if (fn.receiver && !types.has(fn.receiver)) throw new KelError(`Methods go on your own types, and ${fn.receiver} isn't one. Make a plain function instead: fn ${fn.method}(value: ${fn.receiver})`, fn.line, fn.col);
    if (BUILTINS.has(fn.name)) throw new KelError(`'${fn.name}' is built into Kel. Pick another name`, fn.line, fn.col);
    if (types.has(fn.name)) throw new KelError(`'${fn.name}' is already a type. Pick another name`, fn.line, fn.col);
    if (functions.has(fn.name)) throw new KelError(`There are two functions named '${fn.name}'`, fn.line, fn.col);
    functions.set(fn.name, fn);
  }
  const main = functions.get('main');
  if (!main) throw new KelError('Every Kel program needs a main function. Add: fn main { }', 1, 1);
  if (main.params.length || main.returns !== 'void') throw new KelError('main takes nothing and returns nothing: fn main { }', main.line, main.col);
  for (const fn of program.functions) checkFunction(fn, functions, types);
  return program;
}

function checkFunction(fn, functions, types) {
  let loops = 0;
  const fail = (message, node) => { throw new KelError(message, node.line, node.col); };
  const scopes = [new Map()];
  fn.locals = [];
  const slot = (type) => fn.locals.push(type) - 1;
  const declare = (name, type, mutable, node, parameter = false) => {
    if (scopes.at(-1).has(name)) fail(`'${name}' already exists here. Pick another name, or change it with: ${name} = ...`, node);
    const variable = { type, mutable, parameter, local: slot(type) };
    scopes.at(-1).set(name, variable);
    return variable;
  };
  const lookup = (name, node) => {
    for (let i = scopes.length - 1; i >= 0; i--) if (scopes[i].has(name)) return scopes[i].get(name);
    return fail(`There is no variable named '${name}'. Make one with: let ${name} = ...`, node);
  };
  for (const param of fn.params) declare(param.name, param.type, false, param, true);

  /** Kel lists and types act like values: storing one that already has a name stores a copy, so changing one never changes another. */
  const owned = (node) => (isShared(node.type) && ['name', 'field', 'index'].includes(node.kind) ? { kind: 'copy', value: node, type: node.type, line: node.line, col: node.col } : node);
  const stored = (node, expected) => { expr(node, expected); return owned(node); };

  /** The variable under p.x or list[0][1] must be var for anything inside it to change. */
  function changing(node, what) {
    let root = node;
    while (root.kind === 'field' || root.kind === 'index') root = root.object;
    if (root.kind !== 'name') fail(`${what} must be on a variable, not on a value that was just made`, node);
    const variable = lookup(root.name, root);
    if (variable.parameter) fail(`'${root.name}' came into this function, so it can't change here. Make your own copy: var copy = ${root.name}`, node);
    if (!variable.mutable) fail(`'${root.name}' was made with let, so it can't change. Use var ${root.name} = ... if it needs to change`, node);
  }

  function expr(node, expected = null) {
    switch (node.kind) {
      case 'int':
        if (BigInt(node.value) > MAX_INT) fail('This number is too big for an int', node);
        return (node.type = 'int');
      case 'float': return (node.type = 'float');
      case 'bool': return (node.type = 'bool');
      case 'string': return (node.type = 'str');
      case 'template':
        for (const part of node.parts) if (expr(part) === 'void') fail('This has no value to put in the text', part);
        return (node.type = 'str');
      case 'name': {
        const variable = lookup(node.name, node);
        node.local = variable.local;
        return (node.type = variable.type);
      }
      case 'unary': {
        const type = expr(node.value);
        if (node.op === '!' && type !== 'bool') fail(`'!' flips true/false values, not ${type}`, node);
        if (node.op === '-' && type !== 'int' && type !== 'float') fail(`'-' works on numbers, not ${type}`, node);
        return (node.type = type);
      }
      case 'binary': return (node.type = binary(node));
      case 'call': return (node.type = call(node));
      case 'list': return (node.type = listLiteral(node, expected));
      case 'index': return (node.type = index(node));
      case 'field': return (node.type = field(node));
      case 'method': return (node.type = method(node));
    }
    return fail('Kel does not understand this expression', node);
  }

  function listLiteral(node, expected) {
    if (!node.items.length) {
      if (expected && isList(expected)) return expected;
      fail('Say what this empty list will hold, like: var scores: [int] = []', node);
    }
    const wanted = expected && isList(expected) ? itemOf(expected) : null;
    const first = expr(node.items[0], wanted);
    if (first === 'void') fail('This has no value to put in a list', node.items[0]);
    node.items[0] = owned(node.items[0]);
    for (let i = 1; i < node.items.length; i++) {
      const type = expr(node.items[i], first);
      if (type !== first) fail(`A list holds one type, but this list has ${first} and ${type}`, node.items[i]);
      node.items[i] = owned(node.items[i]);
    }
    return `[${first}]`;
  }

  function index(node) {
    const type = expr(node.object);
    if (!isList(type) && type !== 'str') fail(`Only lists and text can use [ ], not ${type}`, node);
    const position = expr(node.index);
    if (position !== 'int') fail(`Positions in [ ] are whole numbers (int), but this is ${position}`, node.index);
    return isList(type) ? itemOf(type) : 'str';
  }

  function field(node) {
    const type = expr(node.object);
    if (isList(type) || type === 'str') {
      if (node.name === 'len' || node.name === 'length') fail(`Use len(...) for the length, like: len(items)`, node);
      fail(`${type} has no field '${node.name}'`, node);
    }
    if (SIMPLE.has(type) || type === 'void') fail(`${type} has no field '${node.name}'`, node);
    const known = types.get(type).fields;
    const found = known.findIndex(item => item.name === node.name);
    if (found < 0) {
      if (functions.has(`${type}.${node.name}`)) fail(`'${node.name}' is a method, so call it: ${node.name}()`, node);
      fail(`${type} has no field '${node.name}'. It has: ${known.map(item => item.name).join(', ')}`, node);
    }
    node.offset = found;
    return known[found].type;
  }

  function method(node) {
    const type = expr(node.object);
    if (node.labels.some(Boolean) && (isList(type) || type === 'str')) fail(`.${node.name}() takes plain values, without names`, node);
    const count = (n) => { if (node.args.length !== n) fail(`.${node.name}() takes ${n} value${n === 1 ? '' : 's'}, but got ${node.args.length}`, node); };
    const wants = (i, wanted) => {
      const got = expr(node.args[i], wanted);
      if (got !== wanted) fail(`.${node.name}() needs ${wanted} here, but got ${got}`, node.args[i]);
    };
    if (isList(type)) {
      const item = itemOf(type);
      node.item = item;
      if (CHANGES_LIST.has(node.name)) changing(node.object, `.${node.name}()`);
      switch (node.name) {
        case 'add': count(1); wants(0, item); node.args[0] = owned(node.args[0]); return 'void';
        case 'pop': count(0); return item;
        case 'remove': count(1); wants(0, 'int'); return 'void';
        case 'reverse': count(0); return 'void';
        case 'sort':
          count(0);
          if (!['int', 'float', 'str'].includes(item)) fail(`.sort() works on lists of int, float or str, not ${item}`, node);
          return 'void';
        case 'contains': case 'index_of':
          count(1);
          if (!SIMPLE.has(item)) fail(`.${node.name}() works on lists of int, float, bool or str, not ${item}`, node);
          wants(0, item);
          return node.name === 'contains' ? 'bool' : 'int';
        case 'join':
          count(1);
          if (item !== 'str') fail(`.join() works on a list of text, not ${item}`, node);
          wants(0, 'str');
          return 'str';
      }
      fail(`Lists have add, pop, remove, sort, reverse, contains, index_of and join, but not '${node.name}'`, node);
    }
    if (type === 'str') {
      const known = TEXT_METHODS[node.name];
      if (!known) fail(`Text has ${Object.keys(TEXT_METHODS).join(', ')}, but not '${node.name}'`, node);
      count(known[0].length);
      known[0].forEach((wanted, i) => wants(i, wanted));
      return known[1];
    }
    if (SIMPLE.has(type) || type === 'void') fail(`${type} has no method '${node.name}'`, node);
    const target = functions.get(`${type}.${node.name}`);
    if (!target) {
      if (types.get(type).fields.some(item => item.name === node.name)) fail(`'${node.name}' is a field, so leave off the ( )`, node);
      fail(`${type} has no method '${node.name}'. Add one with: fn ${type}.${node.name} { }`, node);
    }
    node.target = target.name;
    passValues(node, target.params.slice(1), `.${node.name}()`);
    return target.returns;
  }

  /** Checks call values against parameters. Named values may come in any order; they end up in the parameters' order. */
  function passValues(node, params, what) {
    if (node.args.length !== params.length) fail(`${what} takes ${params.length} value${params.length === 1 ? '' : 's'}, but got ${node.args.length}`, node);
    if (node.labels.some(Boolean)) {
      const ordered = new Array(params.length);
      node.args.forEach((arg, i) => {
        const position = node.labels[i] ? params.findIndex(param => param.name === node.labels[i]) : i;
        if (position < 0) fail(`${what} has no value named '${node.labels[i]}'. It has: ${params.map(param => param.name).join(', ')}`, arg);
        if (ordered[position]) fail(`'${params[position].name}' of ${what} was given twice`, arg);
        ordered[position] = arg;
      });
      node.args = ordered;
    }
    params.forEach((param, i) => {
      const got = expr(node.args[i], param.type);
      if (got !== param.type) fail(`'${param.name}' of ${what} needs ${param.type}, but got ${got}`, node.args[i]);
    });
  }

  function binary(node) {
    if (node.compound && (node.left.kind === 'field' || node.left.kind === 'index')) {
      // list[f()] += 1 reads and writes the same place, so f() would run twice.
      const calls = (item) => item && typeof item === 'object' && (item.kind === 'call' || item.kind === 'method' || Object.values(item).some(value => (Array.isArray(value) ? value.some(calls) : calls(value))));
      if (calls(node.left)) fail('Put that value in a variable first, then change it', node);
    }
    const left = expr(node.left);
    const right = expr(node.right);
    const op = node.op;
    if (op === '&&' || op === '||') {
      if (left !== 'bool' || right !== 'bool') fail(`'${op}' joins two true/false values`, node);
      return 'bool';
    }
    if (left !== right) {
      let hint = '';
      if ((left === 'int' && right === 'float') || (left === 'float' && right === 'int')) hint = ' Convert one side with float(x) or int(x)';
      if (op === '+' && (left === 'str' || right === 'str')) hint = ' To mix text and values, use a template: "Score: {score}"';
      fail(`Can't use '${op}' between ${left} and ${right}.${hint}`, node);
    }
    if (isShared(left)) {
      if (op === '+' && isList(left)) fail('Lists do not join with +. Add items one at a time with list.add(x)', node);
      if (op === '==' || op === '!=') fail(`'${op}' doesn't work on ${isList(left) ? 'lists' : left}. Compare the parts one at a time`, node);
      fail(`'${op}' doesn't work on ${isList(left) ? 'lists' : left}`, node);
    }
    if (op === '==' || op === '!=') return 'bool';
    if (left === 'str') {
      if (op === '+') return 'str';
      fail(`'${op}' doesn't work on text`, node);
    }
    if (left === 'bool') fail(`'${op}' works on numbers, not true/false values`, node);
    if (op === '%' && left === 'float') fail("'%' works on int only", node);
    return ['<', '<=', '>', '>='].includes(op) ? 'bool' : left;
  }

  function call(node) {
    if (node.name === 'print' && !functions.has('print')) fail('Kel shows things with say, like: say x', node);
    const shape = types.get(node.name);
    if (shape) { // Point(x: 1.0, y: 2.0)
      passValues(node, shape.fields, `${node.name}()`);
      node.args = node.args.map(owned);
      node.builds = shape.name;
      return shape.name;
    }
    const target = functions.get(node.name);
    if (target) {
      passValues(node, target.params, `${node.name}()`);
      return target.returns;
    }
    return builtin(node);
  }

  /** Your own function of the same name wins, except for int, float, str and len. */
  function builtin(node) {
    if (node.labels.some(Boolean)) fail(`${node.name}() takes plain values, without names`, node);
    const known = ['int', 'float', 'str', 'len', 'ask', 'abs', 'min', 'max', 'sqrt', 'pow', 'floor', 'ceil', 'round', 'random'];
    if (!known.includes(node.name)) fail(`There is no function named '${node.name}'`, node);
    const list = node.args.map(arg => expr(arg));
    const count = (n) => { if (list.length !== n) fail(`${node.name}() takes ${n} value${n === 1 ? '' : 's'}, but got ${list.length}`, node); };
    const number = (i) => { if (list[i] !== 'int' && list[i] !== 'float') fail(`${node.name}() works on numbers, not ${list[i]}`, node.args[i]); };
    const decimal = (i) => { if (list[i] !== 'float') fail(`${node.name}() takes a float${list[i] === 'int' ? '. Convert with float(x)' : `, not ${list[i]}`}`, node.args[i]); };
    const whole = (i) => { if (list[i] !== 'int') fail(`${node.name}() takes whole numbers (int), not ${list[i]}`, node.args[i]); };
    const same = () => { if (list[0] !== list[1]) fail(`${node.name}() needs two of the same type, but got ${list[0]} and ${list[1]}. Convert one with float(x) or int(x)`, node); };
    switch (node.name) {
      case 'int': case 'float':
        count(1);
        if (list[0] !== 'int' && list[0] !== 'float' && list[0] !== 'str') fail(`${node.name}() converts numbers and text, not ${list[0]}`, node);
        node.builtin = `${list[0]}_to_${node.name}`;
        return node.name;
      case 'str':
        count(1);
        if (list[0] === 'void') fail('This has no value to turn into text', node.args[0]);
        node.builtin = 'to_str';
        return 'str';
      case 'len':
        count(1);
        if (list[0] !== 'str' && !isList(list[0])) fail(`len() measures text and lists, not ${list[0]}`, node);
        node.builtin = list[0] === 'str' ? 'text_len' : 'list_len';
        return 'int';
      case 'ask':
        count(1);
        if (list[0] !== 'str') fail(`ask() shows a question as text, like: ask("Your name?")`, node.args[0]);
        node.builtin = 'ask';
        return 'str';
      case 'abs': count(1); number(0); node.builtin = `abs_${list[0]}`; return list[0];
      case 'min': case 'max': count(2); number(0); number(1); same(); node.builtin = `${node.name}_${list[0]}`; return list[0];
      case 'sqrt': count(1); decimal(0); node.builtin = 'sqrt'; return 'float';
      case 'pow': count(2); number(0); number(1); same(); node.builtin = `pow_${list[0]}`; return list[0];
      case 'floor': case 'ceil': case 'round': count(1); decimal(0); node.builtin = node.name; return 'int';
      case 'random': count(2); whole(0); whole(1); node.builtin = 'random_int'; return 'int';
    }
    return fail(`There is no function named '${node.name}'`, node);
  }

  function condition(node, what) {
    const type = expr(node);
    if (type !== 'bool') fail(`${what} needs a true/false value, but this is ${type}. Did you mean something like x == 0?`, node);
  }
  function whole(node, what) {
    const type = expr(node);
    if (type !== 'int') fail(`${what} needs a whole number (int), but this is ${type}`, node);
  }

  function statements(list, setup) {
    scopes.push(new Map());
    setup?.();
    for (const node of list) statement(node);
    scopes.pop();
  }

  function statement(node) {
    switch (node.kind) {
      case 'let': {
        const type = expr(node.value, node.declared);
        if (type === 'void') fail('This has no value to store', node.value);
        if (node.declared && node.declared !== type) fail(`'${node.name}' is declared as ${node.declared}, but the value is ${type}`, node);
        node.value = owned(node.value);
        node.local = declare(node.name, type, node.mutable, node).local;
        return;
      }
      case 'assign': {
        const variable = lookup(node.name, node);
        if (variable.parameter) fail(`'${node.name}' came into this function, so it can't change here. Make your own copy: var copy = ${node.name}`, node);
        if (!variable.mutable) fail(`'${node.name}' was made with let, so it can't change. Use var ${node.name} = ... if it needs to change`, node);
        const type = expr(node.value, variable.type);
        if (type !== variable.type) fail(`'${node.name}' holds ${variable.type}, so it can't be set to ${type}`, node);
        node.value = owned(node.value);
        node.local = variable.local;
        return;
      }
      case 'set': {
        const place = expr(node.target);
        if (node.target.kind === 'index' && node.target.object.type === 'str') fail("Text can't be changed one letter at a time. Build new text with +", node);
        changing(node.target, 'This change');
        const type = expr(node.value, place);
        if (type !== place) fail(`This holds ${place}, so it can't be set to ${type}`, node);
        node.value = owned(node.value);
        return;
      }
      case 'say':
        if (expr(node.value) === 'void') fail('This has no value to say', node.value);
        return;
      case 'if':
        condition(node.condition, 'if');
        statements(node.then);
        statements(node.otherwise);
        return;
      case 'break': case 'continue':
        if (!loops) fail(`${node.kind} only works inside a loop`, node);
        return;
      case 'while':
        condition(node.condition, 'while');
        loops++; statements(node.body); loops--;
        return;
      case 'repeat':
        whole(node.count, 'repeat');
        node.counter = slot('int');
        node.limit = slot('int');
        loops++; statements(node.body); loops--;
        return;
      case 'forEach': {
        const type = expr(node.list);
        if (!isList(type) && type !== 'str') fail(`for needs a list, text or a range like 1..10, but this is ${type}`, node.list);
        const item = isList(type) ? itemOf(type) : 'str';
        loops++;
        statements(node.body, () => { node.local = declare(node.name, item, false, node).local; });
        loops--;
        return;
      }
      case 'for':
        whole(node.from, 'for');
        whole(node.to, 'for');
        node.limit = slot('int');
        loops++;
        statements(node.body, () => { node.local = declare(node.name, 'int', false, node).local; });
        loops--;
        return;
      case 'return': {
        if (fn.returns === 'void') {
          if (node.value) fail(`${fn.name}() doesn't give back a value. Add -> type after its name to return one`, node);
          return;
        }
        if (!node.value) fail(`${fn.name}() must return ${fn.returns}`, node);
        const type = expr(node.value, fn.returns);
        if (type !== fn.returns) fail(`${fn.name}() returns ${fn.returns}, but this is ${type}`, node.value);
        node.value = owned(node.value);
        return;
      }
      case 'expr':
        expr(node.value);
        return;
    }
  }

  statements(fn.body);
  if (fn.returns !== 'void' && !alwaysReturns(fn.body)) {
    throw new KelError(`${fn.name}() can reach its end without returning ${fn.returns}. Add a return at the end`, fn.line, fn.col);
  }
}

function alwaysReturns(list) {
  const last = list.at(-1);
  if (!last) return false;
  if (last.kind === 'return') return true;
  return last.kind === 'if' && alwaysReturns(last.then) && alwaysReturns(last.otherwise);
}
