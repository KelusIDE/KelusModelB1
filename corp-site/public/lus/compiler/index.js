// Kel: clean { } syntax, typed like Rust, compiles to WebAssembly.
// The compiler has no dependencies, so the same code runs in Node and in the browser.
import { lex, KelError } from './lexer.js';
import { parse } from './parser.js';
import { check } from './checker.js';
import { generate } from './codegen.js';

export { KelError };

/** Kel source text -> WebAssembly bytes. Throws KelError (with line and col) for mistakes. */
export function compile(source) {
  return generate(check(parse(lex(source))));
}

/** Runs compiled bytes. `write` receives each line the program says; `read` answers ask("question") with text. */
export async function run(bytes, write = console.log, read = (question) => globalThis.prompt?.(question) ?? '') {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let memory;
  let heap;
  const view = (packed) => new Uint8Array(memory.buffer, Number(packed >> 32n), Number(packed & 0xffffffffn));
  const data = () => new DataView(memory.buffer);
  const text = (packed) => decoder.decode(view(packed));
  // New text and lists go on a simple heap after the program's own text; memory grows when it runs out.
  const alloc = (size) => {
    const start = heap, end = heap + size;
    if (end > memory.buffer.byteLength) memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
    heap = Math.ceil(end / 8) * 8;
    return start;
  };
  const store = (bytes) => {
    const start = alloc(bytes.length);
    new Uint8Array(memory.buffer).set(bytes, start);
    return (BigInt(start) << 32n) | BigInt(bytes.length);
  };
  const storeText = (value) => store(encoder.encode(value));
  // Letters, not bytes: len("안녕") is 2. The latest text is kept, so that looping over it stays fast.
  let lastPacked = null, lastLetters = [];
  const letters = (packed) => {
    if (packed !== lastPacked) { lastPacked = packed; lastLetters = [...text(packed)]; }
    return lastLetters;
  };
  const formatFloat = (value) => (Number.isInteger(value) ? value.toFixed(1) : String(value));
  // A list is [length, capacity, address of items] on the heap; each item, and each field of a type, takes 8 bytes.
  const listLength = (list) => data().getInt32(list, true);
  const listItems = (list) => data().getInt32(list + 8, true);
  const newList = (capacity) => {
    const list = alloc(16);
    const items = alloc(capacity * 8);
    data().setInt32(list, 0, true);
    data().setInt32(list + 4, capacity, true);
    data().setInt32(list + 8, items, true);
    return list;
  };
  // What a slot holds, for showing and copying. Descriptions come from the compiler as JSON: { t: type, s: { Name: [[field, type]] } }.
  const descriptions = new Map();
  const described = (packed) => {
    if (!descriptions.has(packed)) descriptions.set(packed, JSON.parse(text(packed)));
    return descriptions.get(packed);
  };
  const readSlot = (address, type) => {
    if (type === 'int' || type === 'str') return data().getBigInt64(address, true);
    if (type === 'float') return data().getFloat64(address, true);
    return data().getInt32(address, true); // bool, list or your own type
  };
  const show = (value, type, shapes, inner = false) => {
    if (type === 'int') return String(value);
    if (type === 'float') return formatFloat(value);
    if (type === 'bool') return value ? 'true' : 'false';
    if (type === 'str') return inner ? JSON.stringify(text(value)) : text(value);
    if (type.startsWith('[')) {
      const item = type.slice(1, -1), parts = [], start = listItems(value);
      for (let i = 0; i < listLength(value); i++) parts.push(show(readSlot(start + i * 8, item), item, shapes, true));
      return `[${parts.join(', ')}]`;
    }
    return `${type}(${shapes[type].map(([name, field], i) => `${name}: ${show(readSlot(value + i * 8, field), field, shapes, true)}`).join(', ')})`;
  };
  const shared = (type) => type.startsWith('[') || !['int', 'float', 'bool', 'str'].includes(type);
  const copy = (value, type, shapes) => {
    if (type.startsWith('[')) {
      const item = type.slice(1, -1), length = listLength(value);
      const list = newList(Math.max(length, 4));
      data().setInt32(list, length, true);
      for (let i = 0; i < length; i++) {
        const from = listItems(value) + i * 8, to = listItems(list) + i * 8;
        if (shared(item)) data().setInt32(to, copy(readSlot(from, item), item, shapes), true);
        else data().setBigInt64(to, data().getBigInt64(from, true), true);
      }
      return list;
    }
    const fields = shapes[type], made = alloc(fields.length * 8);
    fields.forEach(([, field], i) => {
      if (shared(field)) data().setInt32(made + i * 8, copy(readSlot(value + i * 8, field), field, shapes), true);
      else data().setBigInt64(made + i * 8, data().getBigInt64(value + i * 8, true), true);
    });
    return made;
  };
  const slotMatches = (address, raw, kind) => {
    if (kind === 1) return data().getFloat64(address, true) === new Float64Array(new BigInt64Array([raw]).buffer)[0];
    if (kind === 3) return text(data().getBigInt64(address, true)) === text(raw);
    return data().getBigInt64(address, true) === raw;
  };
  const indexOf = (list, raw, kind) => {
    const start = listItems(list);
    for (let i = 0; i < listLength(list); i++) if (slotMatches(start + i * 8, raw, kind)) return i;
    return -1;
  };
  const env = {
    say_int: (value) => write(String(value)),
    say_float: (value) => write(formatFloat(value)),
    say_bool: (value) => write(value ? 'true' : 'false'),
    say_str: (value) => write(text(value)),
    concat: (a, b) => {
      const left = view(a), right = view(b);
      const joined = new Uint8Array(left.length + right.length);
      joined.set(left);
      joined.set(right, left.length);
      return store(joined);
    },
    str_eq: (a, b) => {
      const left = view(a), right = view(b);
      return left.length === right.length && left.every((byte, i) => byte === right[i]) ? 1 : 0;
    },
    int_to_str: (value) => storeText(String(value)),
    float_to_str: (value) => storeText(formatFloat(value)),
    bool_to_str: (value) => storeText(value ? 'true' : 'false'),
    alloc,
    index_error: (index, length) => { throw new Error(`There is no item ${index} in this list: it has ${length} item${length === 1 ? '' : 's'}, numbered from 0`); },
    empty_error: () => { throw new Error("pop() can't take an item from an empty list"); },
    show_shared: (value, description) => { const { t, s } = described(description); return storeText(show(value, t, s)); },
    copy_shared: (value, description) => { const { t, s } = described(description); return copy(value, t, s); },
    list_remove: (list, index) => {
      const length = listLength(list);
      if (index < 0n || index >= BigInt(length)) env.index_error(index, length);
      const start = listItems(list), at = start + Number(index) * 8;
      new Uint8Array(memory.buffer).copyWithin(at, at + 8, start + length * 8);
      data().setInt32(list, length - 1, true);
    },
    list_contains: (list, raw, kind) => (indexOf(list, raw, kind) >= 0 ? 1 : 0),
    list_index_of: (list, raw, kind) => BigInt(indexOf(list, raw, kind)),
    list_reverse: (list) => {
      const start = listItems(list), length = listLength(list);
      for (let i = 0, j = length - 1; i < j; i++, j--) {
        const a = data().getBigInt64(start + i * 8, true);
        data().setBigInt64(start + i * 8, data().getBigInt64(start + j * 8, true), true);
        data().setBigInt64(start + j * 8, a, true);
      }
    },
    list_sort: (list, kind) => {
      const start = listItems(list), length = listLength(list), slots = [];
      for (let i = 0; i < length; i++) slots.push(data().getBigInt64(start + i * 8, true));
      const key = kind === 1 ? (raw) => new Float64Array(new BigInt64Array([raw]).buffer)[0] : kind === 3 ? text : (raw) => raw;
      const keyed = slots.map(raw => [key(raw), raw]).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
      keyed.forEach(([, raw], i) => data().setBigInt64(start + i * 8, raw, true));
    },
    list_join: (list, separator) => {
      const start = listItems(list), parts = [];
      for (let i = 0; i < listLength(list); i++) parts.push(text(data().getBigInt64(start + i * 8, true)));
      return storeText(parts.join(text(separator)));
    },
    text_len: (value) => BigInt(letters(value).length),
    text_at: (value, index) => {
      const all = letters(value);
      if (index < 0n || index >= BigInt(all.length)) throw new Error(`There is no letter ${index} in this text: it has ${all.length} letter${all.length === 1 ? '' : 's'}, numbered from 0`);
      return storeText(all[Number(index)]);
    },
    text_slice: (value, from, to) => storeText(letters(value).slice(Math.max(0, Number(from)), Math.max(0, Number(to))).join('')),
    text_upper: (value) => storeText(text(value).toUpperCase()),
    text_lower: (value) => storeText(text(value).toLowerCase()),
    text_trim: (value) => storeText(text(value).trim()),
    text_contains: (value, part) => (text(value).includes(text(part)) ? 1 : 0),
    text_starts_with: (value, part) => (text(value).startsWith(text(part)) ? 1 : 0),
    text_ends_with: (value, part) => (text(value).endsWith(text(part)) ? 1 : 0),
    text_replace: (value, from, to) => storeText(text(value).replaceAll(text(from), text(to))),
    text_split: (value, separator) => {
      const parts = text(separator) === '' ? letters(value) : text(value).split(text(separator));
      const list = newList(Math.max(parts.length, 4));
      data().setInt32(list, parts.length, true);
      parts.forEach((part, i) => {
        const packed = storeText(part);
        data().setBigInt64(listItems(list) + i * 8, packed, true);
      });
      return list;
    },
    ask: (question) => storeText(String(read(text(question)) ?? '')),
    str_to_int: (value) => {
      const clean = text(value).trim().replaceAll('_', '');
      if (!/^[-+]?[0-9]+$/.test(clean)) throw new Error(`"${text(value)}" is not a whole number`);
      const number = BigInt(clean);
      if (BigInt.asIntN(64, number) !== number) throw new Error('A number got too big for an int');
      return number;
    },
    str_to_float: (value) => {
      const clean = text(value).trim();
      const number = clean === '' ? NaN : Number(clean);
      if (Number.isNaN(number)) throw new Error(`"${text(value)}" is not a number`);
      return number;
    },
    pow_int: (base, power) => {
      if (power < 0n) throw new Error('pow() on ints needs a power of 0 or more. For a fraction, use floats: pow(2.0, -1.0)');
      const number = base ** power;
      if (BigInt.asIntN(64, number) !== number) throw new Error('A number got too big for an int');
      return number;
    },
    pow_float: (base, power) => base ** power,
    random_int: (low, high) => {
      if (low > high) throw new Error(`random() needs its smaller number first, but got ${low} and ${high}`);
      return low + BigInt(Math.floor(Math.random() * Number(high - low + 1n)));
    }
  };
  const { instance } = await WebAssembly.instantiate(bytes, { env });
  memory = instance.exports.memory;
  heap = instance.exports.heap_base.value;
  try {
    instance.exports.main();
  } catch (error) {
    throw new Error(runtimeMessage(error));
  }
}

function runtimeMessage(error) {
  const message = String(error?.message ?? error);
  if (/divide by zero/i.test(message)) return 'Your program divided an int by zero';
  if (/remainder by zero/i.test(message)) return "Your program used '%' with zero";
  if (/overflow/i.test(message)) return 'A number got too big for an int';
  if (/call stack|recursion/i.test(message)) return 'A function called itself too many times (infinite recursion?)';
  if (/grow|memory access out of bounds|out of memory/i.test(message)) return 'Your program ran out of memory';
  return message;
}

/** A mistake as `file:line:col: message`, with the line and a ^ under the spot. */
export function formatError(error, source, file = 'main.lus') {
  if (!(error instanceof KelError)) return String(error?.message ?? error);
  const line = source.replace(/\r\n?/g, '\n').split('\n')[error.line - 1] ?? '';
  return `${file}:${error.line}:${error.col}: ${error.message}\n    ${line}\n    ${' '.repeat(Math.max(0, error.col - 1))}^`;
}
