






import { writeFile } from 'node:fs/promises';
import wabtModule from 'wabt';

const range = n => Array.from({ length: n }, (_, k) => k);
const zero = '(v128.const f32x4 0 0 0 0)';

const widen = (address, prefix) => `
        (local.set $v (v128.load ${address}))
        (local.set $lo (i16x8.extend_low_i8x16_s (local.get $v)))
        (local.set $hi (i16x8.extend_high_i8x16_s (local.get $v)))
        (local.set $${prefix}0 (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo))))
        (local.set $${prefix}1 (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo))))
        (local.set $${prefix}2 (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $hi))))
        (local.set $${prefix}3 (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $hi))))`;
const locals = (names, type) => names.map(name => `(local $${name} ${type})`).join(' ');
let fused = false;

const madd = (a, b, acc, int8) => fused && int8 ? `(f32x4.relaxed_madd ${a} ${b} ${acc})` : `(f32x4.add ${acc} (f32x4.mul ${a} ${b}))`;



const matvec = int8 => `
  (func (export "matvec_${int8 ? 'i8' : 'f32'}") (param $x i32) (param $w i32) ${int8 ? '(param $scale i32)' : ''} (param $out i32) (param $rows i32) (param $cols i32)
    (local $j i32) (local $i i32) (local $p i32) (local $xp i32) (local $stride i32) (local $xi v128) (local $v v128) (local $lo v128) (local $hi v128)
    ${locals(range(8).map(k => 'a' + k), 'v128')} ${locals(range(8).map(k => 'w' + k), 'v128')}
    (local.set $stride (i32.mul (local.get $cols) (i32.const ${int8 ? 1 : 4})))
    (local.set $j (i32.const 0))
    (block $done (loop $columns
      (br_if $done (i32.ge_u (local.get $j) (local.get $cols)))
      ${range(8).map(k => `(local.set $a${k} ${zero})`).join(' ')}
      (local.set $p (i32.add (local.get $w) (i32.mul (local.get $j) (i32.const ${int8 ? 1 : 4}))))
      (local.set $xp (local.get $x))
      (local.set $i (i32.const 0))
      (block $rows_done (loop $rows
        (br_if $rows_done (i32.ge_u (local.get $i) (local.get $rows)))
        (local.set $xi (v128.load32_splat (local.get $xp)))
        ${int8 ? widen('(local.get $p)', 'w') + widen('offset=16 (local.get $p)', 'w').replace(/\$w([0-3]) /g, (_, k) => `$w${+k + 4} `)
               : range(8).map(k => `(local.set $w${k} (v128.load offset=${16 * k} (local.get $p)))`).join('\n        ')}
        ${range(8).map(k => `(local.set $a${k} ${madd('(local.get $xi)', `(local.get $w${k})`, `(local.get $a${k})`, int8)})`).join('\n        ')}
        (local.set $p (i32.add (local.get $p) (local.get $stride)))
        (local.set $xp (i32.add (local.get $xp) (i32.const 4)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $rows)))
      (local.set $p (i32.add (local.get $out) (i32.shl (local.get $j) (i32.const 2))))
      ${range(8).map(k => int8
        ? `(v128.store offset=${16 * k} (local.get $p) (f32x4.mul (local.get $a${k}) (v128.load offset=${16 * k} (i32.add (local.get $scale) (i32.shl (local.get $j) (i32.const 2))))))`
        : `(v128.store offset=${16 * k} (local.get $p) (local.get $a${k}))`).join('\n      ')}
      (local.set $j (i32.add (local.get $j) (i32.const 32)))
      (br $columns))))`;




const matmul8 = int8 => `
  (func (export "matmul8_${int8 ? 'i8' : 'f32'}") (param $x i32) (param $w i32) ${int8 ? '(param $scale i32)' : ''} (param $out i32) (param $rows i32) (param $cols i32)
    (local $j i32) (local $i i32) (local $p i32) (local $xp i32) (local $xs i32) (local $o i32) (local $stride i32) (local $xi v128) (local $lo v128) (local $w0 v128) (local $w1 v128)
    ${locals(range(16).map(k => 'a' + k), 'v128')}
    (local.set $stride (i32.mul (local.get $cols) (i32.const ${int8 ? 1 : 4})))
    (local.set $xs (i32.shl (local.get $rows) (i32.const 2)))
    (local.set $j (i32.const 0))
    (block $done (loop $columns
      (br_if $done (i32.ge_u (local.get $j) (local.get $cols)))
      ${range(16).map(k => `(local.set $a${k} ${zero})`).join(' ')}
      (local.set $p (i32.add (local.get $w) (i32.mul (local.get $j) (i32.const ${int8 ? 1 : 4}))))
      (local.set $xp (local.get $x))
      (local.set $i (i32.const 0))
      (block $rows_done (loop $rows
        (br_if $rows_done (i32.ge_u (local.get $i) (local.get $rows)))
        ${int8 ? `(local.set $lo (v128.load8x8_s (local.get $p)))
        (local.set $w0 (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo))))
        (local.set $w1 (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo))))`
               : `(local.set $w0 (v128.load (local.get $p))) (local.set $w1 (v128.load offset=16 (local.get $p)))`}
        ${range(8).map(t => `(local.set $xi (v128.load32_splat ${t ? `(i32.add (local.get $xp) (i32.mul (local.get $xs) (i32.const ${t})))` : '(local.get $xp)'}))
        ${range(2).map(k => `(local.set $a${2 * t + k} ${madd('(local.get $xi)', `(local.get $w${k})`, `(local.get $a${2 * t + k})`, int8)})`).join(' ')}`).join('\n        ')}
        (local.set $p (i32.add (local.get $p) (local.get $stride)))
        (local.set $xp (i32.add (local.get $xp) (i32.const 4)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $rows)))
      ${range(8).map(t => `(local.set $o (i32.add (local.get $out) (i32.shl (i32.add (local.get $j) (i32.mul (local.get $cols) (i32.const ${t}))) (i32.const 2))))
      ${range(2).map(k => int8
        ? `(v128.store offset=${16 * k} (local.get $o) (f32x4.mul (local.get $a${2 * t + k}) (v128.load offset=${16 * k} (i32.add (local.get $scale) (i32.shl (local.get $j) (i32.const 2))))))`
        : `(v128.store offset=${16 * k} (local.get $o) (local.get $a${2 * t + k}))`).join(' ')}`).join('\n      ')}
      (local.set $j (i32.add (local.get $j) (i32.const 8)))
      (br $columns))))`;


const rowsDot = int8 => `
  (func (export "rows_dot_${int8 ? 'i8' : 'f32'}") (param $x i32) (param $table i32) ${int8 ? '(param $scale i32)' : ''} (param $out i32) (param $rows i32) (param $cols i32)
    (local $r i32) (local $j i32) (local $p i32) (local $xp i32) (local $s v128) (local $v v128) (local $lo v128) (local $hi v128)
    ${locals(range(4).map(k => 'a' + k), 'v128')} ${locals(range(4).map(k => 'w' + k), 'v128')}
    (local.set $p (local.get $table))
    (local.set $r (i32.const 0))
    (block $done (loop $each_row
      (br_if $done (i32.ge_u (local.get $r) (local.get $rows)))
      ${range(4).map(k => `(local.set $a${k} ${zero})`).join(' ')}
      (local.set $xp (local.get $x))
      (local.set $j (i32.const 0))
      (block $row_done (loop $columns
        (br_if $row_done (i32.ge_u (local.get $j) (local.get $cols)))
        ${int8 ? widen('(local.get $p)', 'w') : range(4).map(k => `(local.set $w${k} (v128.load offset=${16 * k} (local.get $p)))`).join('\n        ')}
        ${range(4).map(k => `(local.set $a${k} ${madd(`(local.get $w${k})`, `(v128.load offset=${16 * k} (local.get $xp))`, `(local.get $a${k})`, int8)})`).join('\n        ')}
        (local.set $p (i32.add (local.get $p) (i32.const ${int8 ? 16 : 64})))
        (local.set $xp (i32.add (local.get $xp) (i32.const 64)))
        (local.set $j (i32.add (local.get $j) (i32.const 16)))
        (br $columns)))
      (local.set $s (f32x4.add (f32x4.add (local.get $a0) (local.get $a1)) (f32x4.add (local.get $a2) (local.get $a3))))
      (f32.store (i32.add (local.get $out) (i32.shl (local.get $r) (i32.const 2)))
        (f32.mul (f32.add (f32.add (f32x4.extract_lane 0 (local.get $s)) (f32x4.extract_lane 1 (local.get $s)))
                          (f32.add (f32x4.extract_lane 2 (local.get $s)) (f32x4.extract_lane 3 (local.get $s))))
                 ${int8 ? '(f32.load (i32.add (local.get $scale) (i32.shl (local.get $r) (i32.const 2))))' : '(f32.const 1)'}))
      (local.set $r (i32.add (local.get $r) (i32.const 1)))
      (br $each_row))))`;





const attention = () => `
  (func (export "head_scores") (param $q i32) (param $keys i32) (param $scores i32) (param $positions i32) (param $width i32) (param $stride i32)
    (local $p i32) (local $i i32) (local $k i32) (local $s v128) (local $rowbytes i32) (local $bytes i32)
    (local.set $rowbytes (i32.shl (local.get $stride) (i32.const 2)))
    (local.set $bytes (i32.shl (local.get $width) (i32.const 2)))
    (local.set $p (i32.const 0))
    (block $done (loop $each
      (br_if $done (i32.ge_u (local.get $p) (local.get $positions)))
      (local.set $k (i32.add (local.get $keys) (i32.mul (local.get $p) (local.get $rowbytes))))
      (local.set $s ${zero})
      (local.set $i (i32.const 0))
      (block $row_done (loop $row
        (br_if $row_done (i32.ge_u (local.get $i) (local.get $bytes)))
        (local.set $s ${madd('(v128.load (i32.add (local.get $q) (local.get $i)))', '(v128.load (i32.add (local.get $k) (local.get $i)))', '(local.get $s)')})
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $row)))
      (f32.store (i32.add (local.get $scores) (i32.shl (local.get $p) (i32.const 2)))
        (f32.add (f32.add (f32x4.extract_lane 0 (local.get $s)) (f32x4.extract_lane 1 (local.get $s)))
                 (f32.add (f32x4.extract_lane 2 (local.get $s)) (f32x4.extract_lane 3 (local.get $s)))))
      (local.set $p (i32.add (local.get $p) (i32.const 1)))
      (br $each))))
  (func (export "head_mix") (param $weights i32) (param $values i32) (param $out i32) (param $positions i32) (param $width i32) (param $stride i32)
    (local $p i32) (local $i i32) (local $v i32) (local $w v128) (local $rowbytes i32) (local $bytes i32)
    (local.set $rowbytes (i32.shl (local.get $stride) (i32.const 2)))
    (local.set $bytes (i32.shl (local.get $width) (i32.const 2)))
    (local.set $i (i32.const 0))
    (block $zeroed (loop $zero
      (br_if $zeroed (i32.ge_u (local.get $i) (local.get $bytes)))
      (v128.store (i32.add (local.get $out) (local.get $i)) ${zero})
      (local.set $i (i32.add (local.get $i) (i32.const 16)))
      (br $zero)))
    (local.set $p (i32.const 0))
    (block $done (loop $each
      (br_if $done (i32.ge_u (local.get $p) (local.get $positions)))
      (local.set $w (v128.load32_splat (i32.add (local.get $weights) (i32.shl (local.get $p) (i32.const 2)))))
      (local.set $v (i32.add (local.get $values) (i32.mul (local.get $p) (local.get $rowbytes))))
      (local.set $i (i32.const 0))
      (block $row_done (loop $row
        (br_if $row_done (i32.ge_u (local.get $i) (local.get $bytes)))
        (v128.store (i32.add (local.get $out) (local.get $i)) ${madd('(local.get $w)', '(v128.load (i32.add (local.get $v) (local.get $i)))', '(v128.load (i32.add (local.get $out) (local.get $i)))')})
        (local.set $i (i32.add (local.get $i) (i32.const 16)))
        (br $row)))
      (local.set $p (i32.add (local.get $p) (i32.const 1)))
      (br $each))))`;

const wabt = await wabtModule();

const build = fuse => {
  fused = fuse;
  return [[true, false].map(int8 => matvec(int8) + rowsDot(int8)).join('\n') + attention(), [true, false].map(matmul8).join('\n')].map(code => {
    const parsed = wabt.parseWat('kernels.wat', `(module\n  (import "env" "memory" (memory 1))\n${code}\n)`, { simd: true, relaxed_simd: fuse });
    parsed.validate();
    const bytes = new Uint8Array(parsed.toBinary({}).buffer);
    if (bytes.length > 4096) throw new Error(`a ${fuse ? 'fused' : 'plain'} kernel module is ${bytes.length} bytes, over 4 KB`);
    return bytes;
  });
};
const builds = { plain: build(false), fused: build(true) };
const list = modules => `[${modules.map(bytes => `'${Buffer.from(bytes).toString('base64')}'`).join(', ')}]`;
await writeFile(new URL('../kernels.js', import.meta.url),
  `// Generated by site/kernels/build.mjs (WebAssembly SIMD kernels for site/b1.js). Do not edit; rebuild instead.\n` +
  `export const KERNELS = ${list(builds.plain)};\n` +
  `// The same kernels with fused multiply-add (relaxed SIMD), where the browser has it.\n` +
  `export const KERNELS_FMA = ${list(builds.fused)};\n`);
for (const [name, modules] of Object.entries(builds)) console.log(`site/kernels.js: ${name} ${modules.map(bytes => bytes.length).join(' + ')} bytes of WebAssembly`);


const memory = new WebAssembly.Memory({ initial: 1024 });
const f32 = new Float32Array(memory.buffer), i8 = new Int8Array(memory.buffer);
const rows = 768, cols = 3072, n = 8;
let at = 0;
const take = (count, size) => { const address = at; at += Math.ceil(count * size / 64) * 64; return address; };
const check = (label, count, value, expected) => {
  let worst = 0;
  for (let c = 0; c < count; c++) worst = Math.max(worst, Math.abs(value(c) - expected(c)) / (Math.abs(expected(c)) + 1e-3));
  if (!(worst < 1e-3)) throw new Error(`${label}: relative error ${worst}`);
  return worst;
};
const time = (fn, repeat) => { fn(); const start = performance.now(); for (let r = 0; r < repeat; r++) fn(); return (performance.now() - start) / repeat; };
for (const [name, modules] of Object.entries(builds)) {
  let k;
  try { k = Object.assign({}, ...await Promise.all(modules.map(async bytes => (await WebAssembly.instantiate(bytes, { env: { memory } })).instance.exports))); }
  catch (error) { console.log(`${name}: not supported by this Node (${error.message})`); continue; }
  at = 0;
  const x = take(n * rows, 4), wq = take(rows * cols, 1), wf = take(rows * cols, 4), scale = take(cols, 4), out = take(n * cols, 4), single = take(n * cols, 4);
  for (let i = 0; i < n * rows; i++) f32[x / 4 + i] = Math.random() - 0.5;
  for (let i = 0; i < rows * cols; i++) { i8[wq + i] = (Math.random() * 254 - 127) | 0; f32[wf / 4 + i] = Math.random() - 0.5; }
  for (let j = 0; j < cols; j++) f32[scale / 4 + j] = 0.001 + Math.random() * 0.01;
  const reference = (t, j, int8) => { let s = 0; for (let i = 0; i < rows; i++) s += f32[x / 4 + t * rows + i] * (int8 ? i8[wq + i * cols + j] : f32[wf / 4 + i * cols + j]); return int8 ? s * f32[scale / 4 + j] : s; };
  for (const int8 of [true, false]) {
    const w = int8 ? wq : wf, type = int8 ? 'i8' : 'f32', args = int8 ? [scale] : [];
    k[`matvec_${type}`](x, w, ...args, out, rows, cols);
    check(`${name} matvec_${type}`, 64, c => f32[out / 4 + c * 41], c => reference(0, c * 41, int8));
    const one = time(() => k[`matvec_${type}`](x, w, ...args, out, rows, cols), 50);

    for (let t = 0; t < n; t++) k[`matvec_${type}`](x + t * rows * 4, w, ...args, single + t * cols * 4, rows, cols);
    k[`matmul8_${type}`](x, w, ...args, out, rows, cols);
    for (let i = 0; i < n * cols; i++) if (f32[out / 4 + i] !== f32[single / 4 + i]) throw new Error(`${name} matmul8_${type} differs from matvec at ${i}`);
    const eight = time(() => k[`matmul8_${type}`](x, w, ...args, out, rows, cols), 50) / n;

    const scores = time(() => k[`rows_dot_${type}`](x, w, ...args, out, cols, rows), 50);
    check(`${name} rows_dot_${type}`, 64, c => f32[out / 4 + c * 43], c => {
      let s = 0; for (let i = 0; i < rows; i++) s += f32[x / 4 + i] * (int8 ? i8[wq + c * 43 * rows + i] : f32[wf / 4 + c * 43 * rows + i]);
      return int8 ? s * f32[scale / 4 + c * 43] : s;
    });
    console.log(`${name.padEnd(5)} ${type.padEnd(3)}  ${rows}x${cols}: one token ${one.toFixed(3)} ms, in a prompt ${eight.toFixed(3)} ms per token; rows_dot ${scores.toFixed(3)} ms`);
  }


  const width = 64, heads = 12, stride = width * heads, positions = 800;
  at = 0;
  const q = take(stride, 4), keys = take(positions * stride, 4), values = take(positions * stride, 4), scores = take(positions, 4), mixed = take(stride, 4);
  for (let i = 0; i < stride; i++) f32[q / 4 + i] = Math.random() - 0.5;
  for (let i = 0; i < positions * stride; i++) { f32[keys / 4 + i] = Math.random() - 0.5; f32[values / 4 + i] = Math.random() - 0.5; }
  const head = 5, offset = head * width * 4;
  k.head_scores(q + offset, keys + offset, scores, positions, width, stride);
  check(`${name} head_scores`, positions, p => f32[scores / 4 + p], p => { let s = 0; for (let i = 0; i < width; i++) s += f32[q / 4 + head * width + i] * f32[keys / 4 + p * stride + head * width + i]; return s; });
  for (let p = 0; p < positions; p++) f32[scores / 4 + p] = Math.random();
  k.head_mix(scores, values + offset, mixed, positions, width, stride);
  check(`${name} head_mix`, width, i => f32[mixed / 4 + i], i => { let s = 0; for (let p = 0; p < positions; p++) s += f32[scores / 4 + p] * f32[values / 4 + p * stride + head * width + i]; return s; });
  const ms = time(() => { for (let h = 0; h < heads; h++) { k.head_scores(q + h * width * 4, keys + h * width * 4, scores, positions, width, stride); k.head_mix(scores, values + h * width * 4, mixed + h * width * 4, positions, width, stride); } }, 50);
  console.log(`${name.padEnd(5)} attention, ${heads} heads over ${positions} positions: ${ms.toFixed(3)} ms per layer`);
}
