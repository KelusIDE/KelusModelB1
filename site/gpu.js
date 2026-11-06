



import { ASSISTANT, BOS, EOS, Model, Repeats, USER, looping, pickToken } from './b1.js';

const CHUNK = 64;


const STEP = 'struct Job { tokens: u32, start: u32, chunks: u32, pad0: u32 }\n@group(0) @binding(0) var<uniform> job: Job;';
const PARAMS = names => `struct Params { ${names.map(n => `${n}: u32`).join(', ')} }\n@group(0) @binding(1) var<uniform> p: Params;`;

const UNPACK = `fn unpack(w: u32) -> vec4f {
  return vec4f(vec4i(bitcast<i32>(w << 24u), bitcast<i32>(w << 16u), bitcast<i32>(w << 8u), bitcast<i32>(w)) >> vec4u(24u));
}`;

const GELU = `fn gelu(h: vec4f) -> vec4f {
  return 0.5 * h * (1.0 + tanh(clamp(0.7978845608028654 * (h + 0.044715 * h * h * h), vec4f(-15.0), vec4f(15.0))));
}`;

const REDUCE = size => `var<workgroup> partial: array<f32, ${size}>;
fn reduce(value: f32, l: u32, largest: bool) -> f32 {
  partial[l] = value;
  workgroupBarrier();
  for (var k = ${size / 2}u; k > 0u; k >>= 1u) {
    if (l < k) { partial[l] = select(partial[l] + partial[l + k], max(partial[l], partial[l + k]), largest); }
    workgroupBarrier();
  }
  let result = partial[0];
  workgroupBarrier();
  return result;
}`;




const MATRIX = int8 => `${STEP}
${PARAMS(['rows', 'cols4', 'out4', 'cached', 'splits', 'span', 'pad0', 'pad1'])}
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<${int8 ? 'u32' : 'vec4f'}>;
@group(0) @binding(4) var<storage, read> scale: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> partial: array<vec4f>;
${UNPACK}
${GELU}
fn weight(i: u32) -> vec4f { return ${int8 ? 'unpack(W[i])' : 'W[i]'}; }
fn finish(token: u32, word: u32, sum: vec4f) {
  let total = sum${int8 ? ' * scale[word]' : ''};
  let at = p.out4 + (p.cached * job.start + token) * p.cols4 + word;
  MODE
}`;
const MODES = { store: 'out[at] = total;', add: 'out[at] = out[at] + total;', gelu: 'out[at] = gelu(total);' };



const MATVEC = `
var<workgroup> sums: array<vec4f, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let word = g.x * 16u + l.x;
  let begin = g.y * p.span;
  let end = min(p.rows, begin + p.span);
  var acc = vec4f(0.0);
  if (word < p.cols4) {
    for (var k = begin + l.y; k < end; k += 16u) { acc += x[k] * weight(k * p.cols4 + word); }
  }
  sums[l.y * 16u + l.x] = acc;
  workgroupBarrier();
  if (l.y == 0u && word < p.cols4) {
    var total = vec4f(0.0);
    for (var s = 0u; s < 16u; s++) { total += sums[s * 16u + l.x]; }
    partial[g.y * p.cols4 + word] = total;
  }
}`;
const COMBINE = `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let word = g.x;
  if (word >= p.cols4) { return; }
  var total = vec4f(0.0);
  for (var s = 0u; s < p.splits; s++) { total += partial[s * p.cols4 + word]; }
  finish(0u, word, total);
}`;


const MATMUL = `
var<workgroup> xs: array<f32, 1024>;
@compute @workgroup_size(16, 4)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let word = g.x * 16u + l.x;
  let first = g.y * 16u;
  let lid = l.y * 16u + l.x;
  let row = l.y * 256u;
  var a0 = vec4f(0.0); var a1 = vec4f(0.0); var a2 = vec4f(0.0); var a3 = vec4f(0.0);
  for (var k0 = 0u; k0 < p.rows; k0 += 64u) {
    for (var e = lid; e < 1024u; e += 64u) {
      let t = first + e / 64u;
      let k = k0 + e % 64u;
      var value = 0.0;
      if (t < job.tokens && k < p.rows) { value = x[t * p.rows + k]; }
      xs[e] = value;
    }
    workgroupBarrier();
    if (word < p.cols4) {
      let n = min(64u, p.rows - k0);
      for (var kk = 0u; kk < n; kk++) {
        let w = weight((k0 + kk) * p.cols4 + word);
        a0 += xs[row + kk] * w; a1 += xs[row + 64u + kk] * w; a2 += xs[row + 128u + kk] * w; a3 += xs[row + 192u + kk] * w;
      }
    }
    workgroupBarrier();
  }
  let t = first + l.y * 4u;
  if (word < p.cols4) {
    if (t < job.tokens) { finish(t, word, a0); }
    if (t + 1u < job.tokens) { finish(t + 1u, word, a1); }
    if (t + 2u < job.tokens) { finish(t + 2u, word, a2); }
    if (t + 3u < job.tokens) { finish(t + 3u, word, a3); }
  }
}`;

const shaders = ({ context, width: d, heads, learned, int8Table }) => {
  const hw = d / heads, half = hw / 2;
  const consts = `const D: u32 = ${d}u;\nconst D4: u32 = ${d / 4}u;\nconst HW: u32 = ${hw}u;\nconst HALF: u32 = ${half}u;\nconst HEADS: u32 = ${heads}u;`;
  return {

    embed: `${STEP}\n${consts}
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> table: array<${int8Table ? 'u32' : 'f32'}>;
@group(0) @binding(4) var<storage, read> rowScale: array<f32>;
@group(0) @binding(5) var<storage, read> positions: array<f32>;
@group(0) @binding(6) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; let t = g.y;
  if (i >= D || t >= job.tokens) { return; }
  let token = ids[t];
  var value = ${int8Table ? 'f32(bitcast<i32>(table[(token * D + i) / 4u] << (24u - 8u * (i % 4u))) >> 24u) * rowScale[token]' : 'table[token * D + i]'};
  ${learned ? 'value += positions[(job.start + t) * D + i];' : ''}
  x[t * D + i] = value;
}`,

    norm: `${STEP}\n${PARAMS(['last', 'pad0', 'pad1', 'pad2'])}\n${consts}
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> gain: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
${REDUCE(256)}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let t = select(g.x, job.tokens - 1u, p.last == 1u);
  let row = t * D;
  let destination = select(row, 0u, p.last == 1u);
  var s = 0.0;
  for (var i = l.x; i < D; i += 256u) { s += x[row + i]; }
  let mean = reduce(s, l.x, false) / f32(D);
  var v = 0.0;
  for (var i = l.x; i < D; i += 256u) { let c = x[row + i] - mean; v += c * c; }
  let inverse = inverseSqrt(reduce(v, l.x, false) / f32(D) + 1e-5);
  for (var i = l.x; i < D; i += 256u) { out[destination + i] = (x[row + i] - mean) * inverse * gain[i] + bias[i]; }
}`,

    rope: `${STEP}\n${PARAMS(['base', 'cached', 'pad0', 'pad1'])}\n${consts}
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> cosine: array<f32>;
@group(0) @binding(4) var<storage, read> sine: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let pair = g.x; let t = g.y;
  if (pair >= HEADS * HALF || t >= job.tokens) { return; }
  let head = pair / HALF; let i = pair % HALF;
  let a = p.base + (p.cached * job.start + t) * D + head * HW + i;
  let b = a + HALF;
  let c = (job.start + t) * HW + i;
  let first = v[a]; let second = v[b];
  v[a] = first * cosine[c] - second * sine[c];
  v[b] = second * cosine[c + HALF] + first * sine[c + HALF];
}`,


    attention: `${STEP}\n${PARAMS(['layer', 'pad0', 'pad1', 'pad2'])}\n${consts}
const SCALE: f32 = ${Math.fround(hw ** -0.5)};
const CHUNKS: u32 = ${Math.ceil(context / 64)}u;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> keys: array<f32>;
@group(0) @binding(4) var<storage, read> values: array<f32>;
@group(0) @binding(5) var<storage, read_write> mixed: array<f32>;
@group(0) @binding(6) var<storage, read_write> parts: array<f32>;
var<workgroup> qs: array<f32, ${hw}>;
var<workgroup> weights: array<f32, 64>;
${REDUCE(64)}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let head = g.x / job.chunks; let chunk = g.x % job.chunks; let t = g.y;
  let count = job.start + t + 1u;
  let first = chunk * 64u;
  let kv = p.layer + head * HW;
  for (var i = l.x; i < HW; i += 64u) { qs[i] = q[t * D + head * HW + i]; }
  workgroupBarrier();
  let j = first + l.x;
  var score = -3.0e38;
  if (j < count) {
    var s = 0.0;
    for (var i = 0u; i < HW; i++) { s += qs[i] * keys[kv + j * D + i]; }
    score = s * SCALE;
  }
  let top = reduce(score, l.x, true);
  var e = 0.0;
  if (j < count) { e = exp(score - top); }
  weights[l.x] = e;
  let sum = reduce(e, l.x, false);
  let n = min(64u, count - min(count, first));
  let at = ((t * HEADS + head) * CHUNKS + chunk) * (HW + 2u);
  for (var i = l.x; i < HW; i += 64u) {
    var acc = 0.0;
    for (var k = 0u; k < n; k++) { acc += weights[k] * values[kv + (first + k) * D + i]; }
    parts[at + i] = acc;
  }
  if (l.x == 0u) { parts[at + HW] = top; parts[at + HW + 1u] = sum; }
}`,
    mix: `${STEP}\n${PARAMS(['layer', 'pad0', 'pad1', 'pad2'])}\n${consts}
const CHUNKS: u32 = ${Math.ceil(context / 64)}u;
@group(0) @binding(5) var<storage, read_write> mixed: array<f32>;
@group(0) @binding(6) var<storage, read_write> parts: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let head = g.x; let t = g.y;
  let n = (job.start + t + 64u) / 64u;
  let base = (t * HEADS + head) * CHUNKS;
  var top = -3.0e38;
  for (var c = 0u; c < n; c++) { top = max(top, parts[(base + c) * (HW + 2u) + HW]); }
  var sum = 0.0;
  for (var c = 0u; c < n; c++) { let at = (base + c) * (HW + 2u); sum += parts[at + HW + 1u] * exp(parts[at + HW] - top); }
  for (var i = l.x; i < HW; i += 64u) {
    var acc = 0.0;
    for (var c = 0u; c < n; c++) { let at = (base + c) * (HW + 2u); acc += parts[at + i] * exp(parts[at + HW] - top); }
    mixed[t * D + head * HW + i] = acc / sum;
  }
}`,

    logits: `${STEP}\n${consts}
@group(0) @binding(2) var<storage, read> normal: array<vec4f>;
@group(0) @binding(3) var<storage, read> table: array<${int8Table ? 'u32' : 'vec4f'}>;
@group(0) @binding(4) var<storage, read> rowScale: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
${UNPACK}
${REDUCE(64)}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) g: vec3u) {
  let r = g.x;
  var s = 0.0;
  for (var j = l.x; j < D4; j += 64u) { s += dot(normal[j], ${int8Table ? 'unpack(table[r * D4 + j])' : 'table[r * D4 + j]'}); }
  let total = reduce(s, l.x, false);
  if (l.x == 0u) { logits[r] = total${int8Table ? ' * rowScale[r]' : ''}; }
}`,
  };
};






export async function gpuModel(cpu, { check = true } = {}) {
  const adapter = await globalThis.navigator?.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const { context, width: d, layers, vocab_size: vocab, heads } = cpu.config, cacheBytes = layers * context * d * 4;
  const limits = adapter.limits;
  if (cacheBytes > limits.maxStorageBufferBindingSize || cacheBytes > limits.maxBufferSize) throw new Error('The GPU has too little memory for this model');
  if (d % 4 || vocab % 4 || (d / heads) % 2 || vocab > limits.maxComputeWorkgroupsPerDimension || (d / heads + 128) * 4 > limits.maxComputeWorkgroupStorageSize) {
    throw new Error("This model's shape does not suit the GPU kernels");
  }
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: Math.max(cacheBytes, 134217728), maxBufferSize: Math.max(cacheBytes, 268435456) } });
  const info = adapter.info || {};
  const model = new GpuModel(cpu, device, [info.vendor, info.architecture || info.description].filter(Boolean).join(' ') || 'GPU');
  try {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    await model.build(cpu);
    if (check) await model.compare(cpu);
    const [invalid, memory] = [await device.popErrorScope(), await device.popErrorScope()];
    if (invalid || memory) throw new Error((invalid || memory).message);
    return model;
  } catch (error) {
    device.destroy();
    throw error;
  }
}

export class GpuModel {
  constructor(cpu, device, name) {
    this.manifest = cpu.manifest; this.config = cpu.config; this.tokenizer = cpu.tokenizer;
    this.device = device; this.name = name;

    this.lost = device.lost;
    this.owner = null;
  }

  get parameterCount() { return this.manifest.parameters; }
  preparePrompt(messages, reserve) { return Model.prototype.preparePrompt.call(this, messages, reserve); }

  async build(cpu) {
    const { device } = this, { context, width: d, layers, heads, vocab_size: vocab } = this.config, w = cpu.weights;
    const usage = GPUBufferUsage, learned = this.config.positions === 'learned', tied = this.config.tie_embeddings;
    const buffer = (size, flags = usage.STORAGE) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage: flags });
    const upload = (data, flags = usage.STORAGE) => {
      const target = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: flags, mappedAtCreation: true });
      new Uint8Array(target.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      target.unmap();
      return target;
    };
    const uniform = values => upload(Uint32Array.from(values), usage.UNIFORM);
    const none = buffer(16, usage.STORAGE | usage.UNIFORM);
    const matrix = tensor => tensor.int8 ? { int8: true, data: upload(tensor.int8), scale: upload(tensor.scale) } : { int8: false, data: upload(tensor), scale: none };

    const layout = kinds => device.createBindGroupLayout({ entries: kinds.map((kind, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: kind === 'u' ? 'uniform' : kind === 'r' ? 'read-only-storage' : 'storage' } })) });
    const layouts = { matrix: layout(['u', 'u', 'r', 'r', 'r', 'w', 'w']), embed: layout(['u', 'u', 'r', 'r', 'r', 'r', 'w']), norm: layout(['u', 'u', 'r', 'r', 'r', 'w']),
      rope: layout(['u', 'u', 'w', 'r', 'r']), attention: layout(['u', 'u', 'r', 'r', 'r', 'w', 'w']), logits: layout(['u', 'u', 'r', 'r', 'r', 'w']) };
    layouts.mix = layouts.attention;
    const pipeline = (code, kind) => device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layouts[kind]] }),
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    const text = shaders({ context, width: d, heads, learned, int8Table: Boolean(w.token.int8) });
    const matrices = Array.from({ length: layers }, (_, i) => ['q', 'k', 'v', 'projection', 'up', 'down'].map(n => w[`block${i}.${n}`])).flat();
    if (!tied) matrices.push(w.output);
    const kinds = [...new Set(matrices.map(t => Boolean(t.int8)))].flatMap(int8 => Object.keys(MODES).map(mode => [int8, mode]));
    const single = ['embed', 'norm', 'rope', 'attention', 'mix', 'logits'], bodies = { one: MATVEC, combine: COMBINE, many: MATMUL };
    const built = await Promise.all([
      ...single.map(kind => pipeline(text[kind], kind)),
      ...kinds.flatMap(([int8, mode]) => Object.values(bodies).map(body => pipeline(MATRIX(int8).replace('MODE', MODES[mode]) + body, 'matrix'))),
    ]);
    this.pipelines = Object.fromEntries(single.map((kind, i) => [kind, built[i]]));
    this.pipelines.matrix = {};
    kinds.forEach(([int8, mode], i) => {
      this.pipelines.matrix[`${int8}.${mode}`] = Object.fromEntries(Object.keys(bodies).map((name, b) => [name, built[single.length + 3 * i + b]]));
    });


    const flags = usage.STORAGE | usage.COPY_DST;
    this.step = buffer(16, usage.UNIFORM | usage.COPY_DST);
    this.ids = buffer(CHUNK * 4, flags);
    const x = buffer(CHUNK * d * 4), normed = buffer(CHUNK * d * 4), q = buffer(CHUNK * d * 4), mixed = buffer(CHUNK * d * 4), hidden = buffer(CHUNK * 4 * d * 4);
    const keys = buffer(layers * context * d * 4), values = buffer(layers * context * d * 4);

    const SPLITS = 8, partial = buffer(SPLITS * Math.max(4 * d, vocab) * 4), parts = buffer(CHUNK * heads * Math.ceil(context / 64) * (d / heads + 2) * 4);
    this.logits = buffer(vocab * 4, usage.STORAGE | usage.COPY_SRC);
    this.staging = buffer(vocab * 4, usage.MAP_READ | usage.COPY_DST);
    const group = (kind, buffers) => device.createBindGroup({ layout: layouts[kind], entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const rowsOf = t => t.int8 ? upload(Float32Array.from({ length: context * d }, (_, i) => t.int8[i] * t.scale[Math.floor(i / d)])) : upload(t);
    const table = w.token.int8 ? { data: upload(w.token.int8), scale: upload(w.token.scale) } : { data: upload(w.token), scale: none };
    this.embed = group('embed', [this.step, none, this.ids, table.data, table.scale, learned ? rowsOf(w.position) : none, x]);
    const op = (input, tensor, output, cols, mode, out4 = 0, cached = 0) => {
      const m = matrix(tensor), rows = input === hidden ? 4 * d : d, groups = Math.ceil(cols / 64);

      const splits = Math.max(1, Math.min(SPLITS, Math.ceil(128 / groups), Math.floor(rows / 16))), span = Math.ceil(rows / splits);
      return { pipelines: this.pipelines.matrix[`${m.int8}.${mode}`], cols4: cols / 4, splits: Math.ceil(rows / span),
        group: group('matrix', [this.step, uniform([rows, cols / 4, out4, cached, Math.ceil(rows / span), span, 0, 0]), input, m.data, m.scale, output, partial]) };
    };
    const normGroup = (name, last = 0) => group('norm', [this.step, uniform([last, 0, 0, 0]), x, upload(w[name + '.gain']), upload(w[name + '.bias']), normed]);
    const rotary = this.config.positions === 'rope';
    const cosine = rotary ? upload(Float32Array.from(cpu.cos)) : none, sine = rotary ? upload(Float32Array.from(cpu.sin)) : none;
    this.layers = Array.from({ length: layers }, (_, layer) => {
      const prefix = `block${layer}.`, base = layer * context * d;
      return {
        norm1: normGroup(prefix + 'norm1'),
        q: op(normed, w[prefix + 'q'], q, d, 'store'),
        k: op(normed, w[prefix + 'k'], keys, d, 'store', base / 4, 1),
        v: op(normed, w[prefix + 'v'], values, d, 'store', base / 4, 1),
        ropeQ: rotary && group('rope', [this.step, uniform([0, 0, 0, 0]), q, cosine, sine]),
        ropeK: rotary && group('rope', [this.step, uniform([base, 1, 0, 0]), keys, cosine, sine]),
        attention: group('attention', [this.step, uniform([base, 0, 0, 0]), q, keys, values, mixed, parts]),
        projection: op(mixed, w[prefix + 'projection'], x, d, 'add'),
        norm2: normGroup(prefix + 'norm2'),
        up: op(normed, w[prefix + 'up'], hidden, 4 * d, 'gelu'),
        down: op(hidden, w[prefix + 'down'], x, d, 'add'),
      };
    });
    this.final = normGroup('final', 1);
    this.output = tied ? group('logits', [this.step, none, normed, table.data, table.scale, this.logits]) : op(normed, w.output, this.logits, vocab, 'store');
  }


  pass(tokens, start, count, last) {
    const { device } = this, { width: d, heads, vocab_size: vocab } = this.config;
    device.queue.writeBuffer(this.step, 0, Uint32Array.of(count, start, Math.ceil((start + count) / 64), 0));
    device.queue.writeBuffer(this.ids, 0, Uint32Array.from(tokens.slice(start, start + count)));
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    const run = (pipeline, group, x, y = 1) => { pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(x, y); };
    const one = op => { run(op.pipelines.one, op.group, Math.ceil(op.cols4 / 16), op.splits); run(op.pipelines.combine, op.group, Math.ceil(op.cols4 / 64)); };
    const multiply = op => count === 1 ? one(op) : run(op.pipelines.many, op.group, Math.ceil(op.cols4 / 16), Math.ceil(count / 16));
    run(this.pipelines.embed, this.embed, Math.ceil(d / 64), count);
    for (const layer of this.layers) {
      run(this.pipelines.norm, layer.norm1, count);
      multiply(layer.q); multiply(layer.k); multiply(layer.v);
      if (layer.ropeQ) {
        run(this.pipelines.rope, layer.ropeQ, Math.ceil(d / 2 / 64), count);
        run(this.pipelines.rope, layer.ropeK, Math.ceil(d / 2 / 64), count);
      }
      run(this.pipelines.attention, layer.attention, heads * Math.ceil((start + count) / 64), count);
      run(this.pipelines.mix, layer.attention, heads, count);
      multiply(layer.projection);
      run(this.pipelines.norm, layer.norm2, count);
      multiply(layer.up); multiply(layer.down);
    }
    if (last) {
      run(this.pipelines.norm, this.final, 1);
      if (this.config.tie_embeddings) run(this.pipelines.logits, this.output, vocab);
      else one(this.output);
    }
    pass.end();
    if (last) encoder.copyBufferToBuffer(this.logits, 0, this.staging, 0, vocab * 4);
    device.queue.submit([encoder.finish()]);
  }


  async nextLogits(tokens, cache) {
    const { context, vocab_size: vocab } = this.config;

    const start = this.owner === cache ? cache.length || 0 : 0;
    if (!(tokens.length > start && tokens.length <= context)) throw new Error('Expected new tokens within the context limit');
    if (tokens.some(token => !(token >= 0 && token < vocab))) throw new Error('Invalid token ID');
    this.owner = cache;
    for (let at = start; at < tokens.length; at += CHUNK) this.pass(tokens, at, Math.min(CHUNK, tokens.length - at), at + CHUNK >= tokens.length);
    await this.staging.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    cache.length = tokens.length;
    return logits;
  }


  async *generate(messages, { maxTokens = 240, temperature = 0, topK = 30, random = Math.random, prefix = '', repeats = true } = {}) {
    if (!(maxTokens >= 1 && maxTokens <= 4096 && temperature >= 0 && temperature <= 2)) throw new Error('Invalid generation settings');
    const said = repeats ? new Repeats() : null, show = text => said ? said.push(text) : text;
    const { tokens } = this.preparePrompt(messages, maxTokens);
    const decoder = new TextDecoder('utf-8'), cache = {};
    for (const token of prefix ? this.tokenizer.encode(prefix).slice(0, -1) : []) {
      if (tokens.length >= this.config.context - 1) break;
      tokens.push(token);
      const text = show(decoder.decode(this.tokenizer.tokenBytes(token), { stream: true }));
      if (text) yield text;
    }
    const steps = Math.min(maxTokens, this.config.context - tokens.length), generated = [];
    for (let step = 0; step < steps; step++) {
      const token = pickToken(await this.nextLogits(tokens, cache), temperature, topK, random);
      if (token === EOS) break;
      tokens.push(token);
      generated.push(token);
      const text = show(decoder.decode(this.tokenizer.tokenBytes(token), { stream: true }));
      if (text) yield text;
      if (said?.stopped || looping(generated)) break;
    }
    const tail = show(decoder.decode()) + (said ? said.finish() : '');
    if (tail) yield tail;
  }





  async compare(cpu) {
    const ids = [BOS, USER, ...this.tokenizer.encode('Write a Python function that adds two numbers.'), ASSISTANT];
    const cpuCache = {}, gpuCache = {};
    for (let round = 0; round < 2; round++) {
      const expected = cpu.nextLogits(ids, cpuCache), actual = await this.nextLogits(ids, gpuCache);
      let spread = 0, worst = 0, best = 0, picked = 0;
      for (let i = 0; i < expected.length; i++) {
        spread = Math.max(spread, Math.abs(expected[i]));
        worst = Math.max(worst, Math.abs(expected[i] - actual[i]));
        if (expected[i] > expected[best]) best = i;
        if (actual[i] > actual[picked]) picked = i;
      }
      if (!(worst <= 1e-3 * Math.max(1, spread)) || best !== picked) throw new Error(`The GPU's results differ from the CPU's (by ${worst.toPrecision(3)})`);
      ids.push(best);
    }
    this.owner = null;
  }

  destroy() { this.device.destroy(); }
}
