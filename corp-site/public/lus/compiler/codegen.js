// Writes the checked tree as a WebAssembly binary module (no external tools).
// Types: int -> i64, float -> f64, bool -> i32, str -> i64 packing (address << 32 | length),
// lists and your own types -> i32 address.
// Text lives in the module's memory: literals in a data segment, new text on a heap the host
// fills (from the exported `heap_base`). Showing values and building text are imported from the host.
// A list on the heap is [length, capacity, address of items], 4 bytes each; every list item and every field of a type takes 8 bytes.

const VALTYPE = { int: 0x7e, float: 0x7c, bool: 0x7f, str: 0x7e, shared: 0x7f };
const valtype = (type) => VALTYPE[type] ?? VALTYPE.shared;
const IMPORTS = [
  ['say_int', ['int'], 'void'], ['say_float', ['float'], 'void'], ['say_bool', ['bool'], 'void'], ['say_str', ['str'], 'void'],
  ['concat', ['str', 'str'], 'str'], ['str_eq', ['str', 'str'], 'bool'],
  ['int_to_str', ['int'], 'str'], ['float_to_str', ['float'], 'str'], ['bool_to_str', ['bool'], 'str'],
  ['alloc', ['shared'], 'shared'], ['index_error', ['int', 'shared'], 'void'], ['empty_error', [], 'void'],
  ['show_shared', ['shared', 'str'], 'str'], ['copy_shared', ['shared', 'str'], 'shared'],
  ['list_remove', ['shared', 'int'], 'void'], ['list_contains', ['shared', 'int', 'shared'], 'bool'], ['list_index_of', ['shared', 'int', 'shared'], 'int'],
  ['list_reverse', ['shared'], 'void'], ['list_sort', ['shared', 'shared'], 'void'], ['list_join', ['shared', 'str'], 'str'],
  ['text_len', ['str'], 'int'], ['text_at', ['str', 'int'], 'str'], ['text_slice', ['str', 'int', 'int'], 'str'],
  ['text_upper', ['str'], 'str'], ['text_lower', ['str'], 'str'], ['text_trim', ['str'], 'str'],
  ['text_contains', ['str', 'str'], 'bool'], ['text_starts_with', ['str', 'str'], 'bool'], ['text_ends_with', ['str', 'str'], 'bool'],
  ['text_replace', ['str', 'str', 'str'], 'str'], ['text_split', ['str', 'str'], 'shared'],
  ['ask', ['str'], 'str'], ['str_to_int', ['str'], 'int'], ['str_to_float', ['str'], 'float'],
  ['pow_int', ['int', 'int'], 'int'], ['pow_float', ['float', 'float'], 'float'], ['random_int', ['int', 'int'], 'int']
];
const call = (name) => [0x10, ...u32(IMPORTS.findIndex(([n]) => n === name))];
const OPS = {
  int: { '+': 0x7c, '-': 0x7d, '*': 0x7e, '/': 0x7f, '%': 0x81, '==': 0x51, '!=': 0x52, '<': 0x53, '>': 0x55, '<=': 0x57, '>=': 0x59 },
  float: { '+': 0xa0, '-': 0xa1, '*': 0xa2, '/': 0xa3, '==': 0x61, '!=': 0x62, '<': 0x63, '>': 0x64, '<=': 0x65, '>=': 0x66 },
  bool: { '==': 0x46, '!=': 0x47 }
};
const CONVERSIONS = { int_to_int: [], float_to_float: [], int_to_float: [0xb9], float_to_int: [0xfc, 0x06] }; // i64.trunc_sat_f64_s: never traps
const EMPTY = 0x40;
const SIMPLE = new Set(['int', 'float', 'bool', 'str']);
const KIND = { int: 0, float: 1, bool: 2, str: 3 }; // how the host compares and sorts items
const load = (type, offset = 0) => (type === 'int' || type === 'str' ? [0x29, 3, ...u32(offset)] : type === 'float' ? [0x2b, 3, ...u32(offset)] : [0x28, 2, ...u32(offset)]);
const save = (type, offset = 0) => (type === 'int' || type === 'str' ? [0x37, 3, ...u32(offset)] : type === 'float' ? [0x39, 3, ...u32(offset)] : [0x36, 2, ...u32(offset)]);
const get = (local) => [0x20, ...u32(local)];
const set = (local) => [0x21, ...u32(local)];
const i32 = (value) => [0x41, ...s64(BigInt(value))];

// ---- binary encoding helpers ----
function u32(n) {
  const out = [];
  do {
    let byte = n & 0x7f;
    n >>>= 7;
    if (n) byte |= 0x80;
    out.push(byte);
  } while (n);
  return out;
}
function s64(value) {
  const out = [];
  for (;;) {
    const byte = Number(value & 0x7fn);
    value >>= 7n;
    const done = (value === 0n && !(byte & 0x40)) || (value === -1n && byte & 0x40);
    out.push(done ? byte : byte | 0x80);
    if (done) return out;
  }
}
function f64(value) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value, true);
  return [...new Uint8Array(view.buffer)];
}
const utf8 = (text) => [...new TextEncoder().encode(text)];
const name = (text) => { const bytes = utf8(text); return [...u32(bytes.length), ...bytes]; };
const vector = (items) => [...u32(items.length), ...items.flat()];
const section = (id, bytes) => [id, ...u32(bytes.length), ...bytes];

export function generate(program) {
  const signatures = [];
  const signature = (params, result) => {
    const bytes = [0x60, ...vector(params.map(type => [valtype(type)])), ...vector(result === 'void' ? [] : [[valtype(result)]])];
    const key = bytes.join(',');
    let index = signatures.findIndex(existing => existing.join(',') === key);
    if (index < 0) index = signatures.push(bytes) - 1;
    return index;
  };
  const importTypes = IMPORTS.map(([, params, result]) => signature(params, result));
  const functionIndex = new Map(program.functions.map((fn, i) => [fn.name, IMPORTS.length + i]));
  const functionTypes = program.functions.map(fn => signature(fn.params.map(p => p.type), fn.returns));
  // Small list helpers written directly in WebAssembly, so that reading and adding items never leave the module.
  const firstHelper = IMPORTS.length + program.functions.length;
  const HELPERS = { list_slot: firstHelper, list_push: firstHelper + 1, list_pop: firstHelper + 2 };
  const helper = (name) => [0x10, ...u32(HELPERS[name])];
  const helperTypes = [signature(['shared', 'int'], 'shared'), signature(['shared'], 'shared'), signature(['shared'], 'shared')];
  const helperBodies = [
    // list_slot(list, index): the address of an item; a wrong index (negative ones too, as unsigned) stops the program with a clear message
    [...vector([]), ...get(1), ...get(0), 0x28, 2, 0, 0xad, 0x5a, 0x04, EMPTY, ...get(1), ...get(0), 0x28, 2, 0, ...call('index_error'), 0x00, 0x0b,
      ...get(0), 0x28, 2, 8, ...get(1), 0xa7, ...i32(3), 0x74, 0x6a, 0x0b],
    // list_push(list): room for one more item, growing the list when it is full; returns the new item's address
    [...vector([[3, VALTYPE.shared]]), ...get(0), 0x28, 2, 0, ...set(1), ...get(0), 0x28, 2, 4, ...set(2),
      ...get(1), ...get(2), 0x4e, 0x04, EMPTY,
      ...get(2), ...i32(1), 0x74, ...i32(4), ...get(2), 0x1b + 0, ...set(2),
      ...get(2), ...i32(3), 0x74, ...call('alloc'), ...set(3),
      ...get(3), ...get(0), 0x28, 2, 8, ...get(1), ...i32(3), 0x74, 0xfc, 0x0a, 0, 0,
      ...get(0), ...get(3), 0x36, 2, 8, ...get(0), ...get(2), 0x36, 2, 4, 0x0b,
      ...get(0), ...get(1), ...i32(1), 0x6a, 0x36, 2, 0,
      ...get(0), 0x28, 2, 8, ...get(1), ...i32(3), 0x74, 0x6a, 0x0b],
    // list_pop(list): takes off the last item and returns its address
    [...vector([[1, VALTYPE.shared]]), ...get(0), 0x28, 2, 0, ...i32(1), 0x6b, ...set(1), ...get(1), ...i32(0), 0x48, 0x04, EMPTY, ...call('empty_error'), 0x00, 0x0b,
      ...get(0), ...get(1), 0x36, 2, 0, ...get(0), 0x28, 2, 8, ...get(1), ...i32(3), 0x74, 0x6a, 0x0b]
  ];

  // Text literals go into a data segment; identical texts share one copy.
  const data = [];
  const strings = new Map();
  const DATA_START = 8;
  const addString = (text) => {
    if (!strings.has(text)) { strings.set(text, DATA_START + data.length); data.push(...utf8(text)); }
    return { address: strings.get(text), length: utf8(text).length };
  };

  // What the host needs to show or copy a list or a type: JSON text in the data segment.
  const describe = (type) => {
    const shapes = {};
    const visit = (item) => {
      if (item.startsWith('[')) return visit(item.slice(1, -1));
      if (SIMPLE.has(item) || shapes[item]) return;
      shapes[item] = program.typeTable.get(item).fields.map(field => [field.name, field.type]);
      shapes[item].forEach(([, inner]) => visit(inner));
    };
    visit(type);
    const { address, length } = addString(JSON.stringify({ t: type, s: shapes }));
    return [0x42, ...s64((BigInt(address) << 32n) | BigInt(length))];
  };
  const toText = (type) => (SIMPLE.has(type) ? (type === 'str' ? [] : call(`${type}_to_str`)) : [...describe(type), ...call('show_shared')]);
  // The value of a simple item as the 8 bytes the host compares.
  const raw = (type) => (type === 'float' ? [0xbd] : type === 'bool' ? [0xad] : []);

  let current = null; // the function being written; it may need extra locals for in-between values
  const temporary = (type) => current.locals.push(type) - 1;
  let depth = 0; // how many blocks are open, so that break and continue can name theirs
  const loops = [];

  function expr(node, code) {
    switch (node.kind) {
      case 'int': code.push(0x42, ...s64(BigInt(node.value))); return;
      case 'float': code.push(0x44, ...f64(Number(node.value))); return;
      case 'bool': code.push(0x41, node.value ? 1 : 0); return;
      case 'string': {
        const { address, length } = addString(node.value);
        code.push(0x42, ...s64((BigInt(address) << 32n) | BigInt(length)));
        return;
      }
      case 'template': // "a{b}c" -> concat(concat("a", text(b)), "c")
        node.parts.forEach((part, i) => {
          expr(part, code);
          code.push(...toText(part.type));
          if (i > 0) code.push(...call('concat'));
        });
        return;
      case 'name': code.push(0x20, ...u32(node.local)); return;
      case 'unary':
        if (node.op === '!') { expr(node.value, code); code.push(0x45); return; } // i32.eqz
        if (node.type === 'float') { expr(node.value, code); code.push(0x9a); return; } // f64.neg
        code.push(0x42, 0); expr(node.value, code); code.push(0x7d); return; // 0 - x
      case 'binary':
        if (node.op === '&&' || node.op === '||') { // short-circuit: the right side runs only when needed
          expr(node.left, code);
          code.push(0x04, VALTYPE.bool);
          if (node.op === '&&') { expr(node.right, code); code.push(0x05, 0x41, 0); }
          else { code.push(0x41, 1, 0x05); expr(node.right, code); }
          code.push(0x0b);
          return;
        }
        expr(node.left, code);
        expr(node.right, code);
        if (node.left.type === 'str') code.push(...call(node.op === '+' ? 'concat' : 'str_eq'), ...(node.op === '!=' ? [0x45] : []));
        else code.push(OPS[node.left.type][node.op]);
        return;
      case 'copy':
        expr(node.value, code);
        code.push(...describe(node.type), ...call('copy_shared'));
        return;
      case 'list': { // made with room for its items, then filled
        const list = temporary('shared');
        code.push(...i32(16), ...call('alloc'), ...set(list), ...get(list), ...i32(Math.max(node.items.length, 4) * 8), ...call('alloc'), 0x36, 2, 8,
          ...get(list), ...i32(Math.max(node.items.length, 4)), 0x36, 2, 4);
        for (const item of node.items) {
          const value = temporary(item.type);
          expr(item, code);
          code.push(...set(value), ...get(list), ...helper('list_push'), ...get(value), ...save(item.type));
        }
        code.push(...get(list));
        return;
      }
      case 'index':
        expr(node.object, code);
        expr(node.index, code);
        if (node.object.type === 'str') code.push(...call('text_at'));
        else code.push(...helper('list_slot'), ...load(node.type));
        return;
      case 'field':
        expr(node.object, code);
        code.push(...load(node.type, node.offset * 8));
        return;
      case 'method': return method(node, code);
      case 'call':
        if (node.builds) { // Point(x: 1.0, y: 2.0)
          const made = temporary('shared');
          code.push(...i32(node.args.length * 8), ...call('alloc'), ...set(made));
          node.args.forEach((arg, i) => {
            const value = temporary(arg.type);
            expr(arg, code);
            code.push(...set(value), ...get(made), ...get(value), ...save(arg.type, i * 8));
          });
          code.push(...get(made));
          return;
        }
        for (const arg of node.args) expr(arg, code);
        if (node.builtin) return builtin(node, code);
        code.push(0x10, ...u32(functionIndex.get(node.name)));
        return;
    }
  }

  function builtin(node, code) {
    const name = node.builtin;
    if (name in CONVERSIONS) return void code.push(...CONVERSIONS[name]);
    const pair = (type, op) => { // leaves the value that wins: a, b, a op b, select
      const b = temporary(type), a = temporary(type);
      code.push(...set(b), ...set(a), ...get(a), ...get(b), ...get(a), ...get(b), op, 0x1b);
    };
    switch (name) {
      case 'to_str': code.push(...toText(node.args[0].type)); return;
      case 'text_len': case 'ask': case 'str_to_int': case 'str_to_float': case 'pow_int': case 'pow_float': case 'random_int': code.push(...call(name)); return;
      case 'list_len': code.push(0x28, 2, 0, 0xad); return; // i32.load, i64.extend_i32_u
      case 'abs_int': { const value = temporary('int'); code.push(...set(value), 0x42, 0, ...get(value), 0x7d, ...get(value), ...get(value), 0x42, 0, 0x53, 0x1b); return; }
      case 'abs_float': code.push(0x99); return;
      case 'min_int': return pair('int', 0x53);
      case 'max_int': return pair('int', 0x55);
      case 'min_float': code.push(0xa4); return;
      case 'max_float': code.push(0xa5); return;
      case 'sqrt': code.push(0x9f); return;
      case 'floor': code.push(0x9c, 0xfc, 0x06); return;
      case 'ceil': code.push(0x9b, 0xfc, 0x06); return;
      case 'round': code.push(0x44, ...f64(0.5), 0xa0, 0x9c, 0xfc, 0x06); return; // 2.5 -> 3
    }
  }

  function method(node, code) {
    expr(node.object, code);
    const type = node.object.type;
    if (type === 'str') {
      for (const arg of node.args) expr(arg, code);
      code.push(...call(`text_${node.name}`));
      return;
    }
    if (!type.startsWith('[')) { // your own type: the object goes first, as self
      for (const arg of node.args) expr(arg, code);
      code.push(0x10, ...u32(functionIndex.get(node.target)));
      return;
    }
    const item = node.item;
    switch (node.name) {
      case 'add': { // the value comes first: working it out may change this same list
        const list = temporary('shared'), value = temporary(item);
        code.push(...set(list));
        expr(node.args[0], code);
        code.push(...set(value), ...get(list), ...helper('list_push'), ...get(value), ...save(item));
        return;
      }
      case 'pop': code.push(...helper('list_pop'), ...load(item)); return;
      case 'remove': expr(node.args[0], code); code.push(...call('list_remove')); return;
      case 'reverse': code.push(...call('list_reverse')); return;
      case 'sort': code.push(...i32(KIND[item]), ...call('list_sort')); return;
      case 'contains': case 'index_of':
        expr(node.args[0], code);
        code.push(...raw(item), ...i32(KIND[item]), ...call(`list_${node.name}`));
        return;
      case 'join': expr(node.args[0], code); code.push(...call('list_join')); return;
    }
  }

  /** Statements in a block of their own, so that depth stays right for break and continue. */
  function open(code, opcode) { code.push(opcode, EMPTY); depth++; }
  function close(code) { code.push(0x0b); depth--; }
  /** block { loop { if done break; block { body }; step; continue } }: continue leaves the inner block, break the outer one. */
  function loopWith(code, done, body, step) {
    open(code, 0x02);
    const outer = depth;
    open(code, 0x03);
    const start = depth;
    code.push(...done, 0x0d, 1);
    open(code, 0x02);
    loops.push({ breakTo: outer, continueTo: depth });
    for (const inner of body) statement(inner, code);
    loops.pop();
    close(code);
    code.push(...step, 0x0c, ...u32(depth - start));
    close(code);
    close(code);
  }

  function statement(node, code) {
    switch (node.kind) {
      case 'let':
      case 'assign': expr(node.value, code); code.push(0x21, ...u32(node.local)); return;
      case 'set': { // the value comes first: working it out may change the same list
        const value = temporary(node.value.type);
        expr(node.value, code);
        code.push(...set(value));
        expr(node.target.object, code);
        if (node.target.kind === 'index') {
          expr(node.target.index, code);
          code.push(...helper('list_slot'), ...get(value), ...save(node.value.type));
        } else code.push(...get(value), ...save(node.value.type, node.target.offset * 8));
        return;
      }
      case 'break': code.push(0x0c, ...u32(depth - loops.at(-1).breakTo)); return;
      case 'continue': code.push(0x0c, ...u32(depth - loops.at(-1).continueTo)); return;
      case 'expr': expr(node.value, code); if (node.value.type !== 'void') code.push(0x1a); return; // drop unused values
      case 'say':
        expr(node.value, code);
        if (SIMPLE.has(node.value.type)) code.push(...call(`say_${node.value.type}`));
        else code.push(...toText(node.value.type), ...call('say_str'));
        return;
      case 'return': if (node.value) expr(node.value, code); code.push(0x0f); return;
      case 'if':
        expr(node.condition, code);
        open(code, 0x04);
        for (const inner of node.then) statement(inner, code);
        if (node.otherwise.length) { code.push(0x05); for (const inner of node.otherwise) statement(inner, code); }
        close(code);
        return;
      case 'while': {
        const done = [];
        expr(node.condition, done);
        done.push(0x45);
        loopWith(code, done, node.body, []);
        return;
      }
      case 'repeat': // counter = 0; limit = count; while counter < limit { body; counter += 1 }
        expr(node.count, code);
        code.push(0x21, ...u32(node.limit), 0x42, 0, 0x21, ...u32(node.counter));
        loopWith(code, [...get(node.counter), ...get(node.limit), 0x59], node.body, addOne(node.counter)); // i64.ge_s
        return;
      case 'for': // i = from; limit = to; while i <= limit { body; i += 1 }
        expr(node.from, code);
        code.push(0x21, ...u32(node.local));
        expr(node.to, code);
        code.push(0x21, ...u32(node.limit));
        loopWith(code, [...get(node.local), ...get(node.limit), 0x55], node.body, addOne(node.local)); // i64.gt_s
        return;
      case 'forEach': {
        const position = temporary('int'), limit = temporary('int');
        if (node.list.type === 'str') { // limit = len(text); while position < limit { x = text[position]; body; position += 1 }
          const text = temporary('str');
          expr(node.list, code);
          code.push(...set(text), ...get(text), ...call('text_len'), ...set(limit));
          const done = [...get(position), ...get(limit), 0x59];
          const body = [{ kind: 'bytes', code: [...get(text), ...get(position), ...call('text_at'), ...set(node.local)] }, ...node.body];
          code.push(0x42, 0, ...set(position));
          loopWith(code, done, body, addOne(position));
          return;
        }
        // Goes over the items there were at the start; if the list has shrunk since, it stops at the new end.
        const list = temporary('shared');
        const item = node.list.type.slice(1, -1);
        expr(node.list, code);
        code.push(...set(list), ...get(list), 0x28, 2, 0, 0xad, ...set(limit), 0x42, 0, ...set(position));
        const done = [...get(position), ...get(limit), 0x59, ...get(position), ...get(list), 0x28, 2, 0, 0xad, 0x59, 0x72];
        const body = [{ kind: 'bytes', code: [...get(list), ...get(position), ...helper('list_slot'), ...load(item), ...set(node.local)] }, ...node.body];
        loopWith(code, done, body, addOne(position));
        return;
      }
      case 'bytes': code.push(...node.code); return;
    }
  }
  const addOne = (local) => [...get(local), 0x42, 1, 0x7c, ...set(local)];

  const bodies = program.functions.map(fn => {
    const code = [];
    current = fn;
    for (const node of fn.body) statement(node, code);
    if (fn.returns !== 'void') code.push(0x00); // the checker guarantees a return before this point
    code.push(0x0b);
    const locals = fn.locals.slice(fn.params.length).map(type => [1, valtype(type)]);
    const body = [...vector(locals), ...code];
    return [...u32(body.length), ...body];
  });

  const heapBase = Math.ceil((DATA_START + data.length) / 8) * 8;
  const pages = Math.max(1, Math.ceil(heapBase / 65536));
  const module = [
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...section(1, vector(signatures)),
    ...section(2, vector(IMPORTS.map(([field], i) => [...name('env'), ...name(field), 0x00, ...u32(importTypes[i])]))),
    ...section(3, vector([...functionTypes, ...helperTypes].map(index => u32(index)))),
    ...section(5, vector([[0x00, ...u32(pages)]])),
    ...section(6, vector([[VALTYPE.bool, 0x00, 0x41, ...s64(BigInt(heapBase)), 0x0b]])), // const i32 heap_base
    ...section(7, vector([[...name('main'), 0x00, ...u32(functionIndex.get('main'))], [...name('memory'), 0x02, 0], [...name('heap_base'), 0x03, 0]])),
    ...section(10, vector([...bodies, ...helperBodies.map(body => [...u32(body.length), ...body])])),
    ...(data.length ? section(11, vector([[0x00, 0x41, ...s64(BigInt(DATA_START)), 0x0b, ...vector(data.map(byte => [byte]))]])) : [])
  ];
  return new Uint8Array(module);
}
