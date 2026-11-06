
export function modelBytes(manifest) {
  return manifest.quantization === 'int8-per-channel' ? manifest.bytes : manifest.parameters * 4;
}

export function validateManifest(manifest, bytes) {
  if (manifest?.format !== 'kelus-b1-web' || manifest.version !== 1 ||
      typeof manifest.name !== 'string' || !/^[a-f0-9]{64}$/.test(manifest.weights_sha256 || '')) {
    throw new Error('Unsupported or incomplete Kani model.');
  }
  const quantized = manifest.quantization === 'int8-per-channel';
  if (manifest.quantization !== undefined && !quantized) throw new Error('Unsupported model compression.');
  if (!Number.isSafeInteger(manifest.parameters) || manifest.parameters < 1 || !Number.isSafeInteger(modelBytes(manifest)) || bytes !== modelBytes(manifest)) {
    throw new Error('The model weights are incomplete. Repackage the model.');
  }
  if (!Array.isArray(manifest.tensors) || !manifest.tensors.length) throw new Error('Missing model tensors.');
  const size = tensor => tensor.shape.reduce((a, b) => a * b, 1);
  for (const tensor of manifest.tensors) {
    if (!Array.isArray(tensor.shape) || !tensor.shape.length || !tensor.shape.every(n => Number.isSafeInteger(n) && n > 0)) throw new Error('Invalid tensor layout.');
  }

  const floats = manifest.tensors.filter(t => t.dtype === undefined), int8 = manifest.tensors.filter(t => t.dtype === 'int8');
  if (floats.length + int8.length !== manifest.tensors.length || (int8.length && !quantized)) throw new Error('Invalid tensor layout.');
  let offset = 0;
  for (const tensor of floats) {
    if (tensor.offset !== offset) throw new Error('Invalid tensor layout.');
    offset += size(tensor);
  }
  let byte = offset * 4, weights = offset;
  for (const tensor of int8) {
    const scales = tensor.shape[tensor.scale_axis === 1 ? 0 : 1];
    if (tensor.byte_offset !== byte || ![0, 1].includes(tensor.scale_axis) || !Number.isSafeInteger(tensor.scale) ||
        tensor.scale < 0 || tensor.scale + scales > offset) throw new Error('Invalid tensor layout.');
    byte += size(tensor);
    weights += size(tensor) - scales;
  }
  if (weights !== manifest.parameters || (quantized ? byte + (-byte & 3) : offset * 4) !== bytes) throw new Error('Incomplete tensor layout.');
}
