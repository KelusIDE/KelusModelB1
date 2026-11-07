import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { lstat, mkdir, mkdtemp, readFile, readlink, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { pack } from '../pack.mjs';

const shells = ['sh', 'dash', 'bash'].filter(shell => spawnSync('sh', ['-c', `command -v ${shell}`]).status === 0);
const packed = (async () => { const out = await mkdtemp(path.join(os.tmpdir(), 'kani-pack-')); return { out, ...await pack(out) }; })();
test.after(async () => rm((await packed).out, { recursive: true, force: true }));

function install(shell, script, home, url, ...options) {
  return spawnSync(shell, [script, ...options], { encoding: 'utf8', env: { ...process.env, HOME: home, SHELL: '/bin/zsh', PATH: `${path.dirname(process.execPath)}:/usr/bin:/bin`, KANI_CLI_URL: url } });
}

test('the installer is valid in every shell here', async () => {
  const { script } = await packed;
  for (const shell of shells) assert.equal(spawnSync(shell, ['-n', script]).status, 0, shell);
  assert.doesNotMatch(await readFile(script, 'utf8'), /@(VERSION|SHA256)@/);
});

test('the package runs on its own, away from the repository', async () => {
  const { script, archive, version } = await packed;
  for (const shell of shells) {
    const home = await mkdtemp(path.join(os.tmpdir(), 'kani-home-'));
    try {
      const result = install(shell, script, home, `file://${archive}`, '--no-models');
      assert.equal(result.status, 0, result.stderr);
      const kani = path.join(home, '.local/bin/kani');
      assert.ok((await lstat(kani)).isSymbolicLink());
      assert.match(await readlink(path.join(home, '.kani/current')), new RegExp(`cli-${version}`));
      assert.match(execFileSync(kani, ['--version'], { encoding: 'utf8', cwd: home }), new RegExp(version));
      assert.match(await readFile(path.join(home, '.zshrc'), 'utf8'), /\.local\/bin/);
      // Installing again must not stack up PATH lines.
      assert.equal(install(shell, script, home, `file://${archive}`, '--no-models').status, 0);
      assert.equal((await readFile(path.join(home, '.zshrc'), 'utf8')).match(/# Kani CLI/g).length, 1);
    } finally { await rm(home, { recursive: true, force: true }); }
  }
});

test('an update keeps models already downloaded', async () => {
  const { script, archive, version } = await packed;
  const home = await mkdtemp(path.join(os.tmpdir(), 'kani-home-'));
  try {
    const old = path.join(home, '.kani/cli-0.0.1/site/models/otter-4-12');
    await mkdir(old, { recursive: true });
    await writeFile(path.join(old, 'b1.bin'), 'weights');
    execFileSync('ln', ['-s', 'cli-0.0.1', path.join(home, '.kani/current')]);
    assert.equal(install('sh', script, home, `file://${archive}`, '--no-models', '--no-modify-path').status, 0);
    assert.equal(await readFile(path.join(home, `.kani/cli-${version}/site/models/otter-4-12/b1.bin`), 'utf8'), 'weights');
    await assert.rejects(lstat(path.join(home, '.kani/cli-0.0.1')));
  } finally { await rm(home, { recursive: true, force: true }); }
});

test('a changed download installs nothing', async () => {
  const { script, out } = await packed;
  const home = await mkdtemp(path.join(os.tmpdir(), 'kani-home-')), bad = path.join(out, 'bad.tar.gz');
  try {
    await writeFile(bad, 'not the release');
    const result = install('sh', script, home, `file://${bad}`, '--no-models');
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /checksum/);
    await assert.rejects(lstat(path.join(home, '.local/bin/kani')));
    await assert.rejects(lstat(path.join(home, '.kani')));
  } finally { await rm(home, { recursive: true, force: true }); }
});
