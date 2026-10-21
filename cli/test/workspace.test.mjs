import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { Workspace, extractCode, pathArgument } from '../workspace.mjs';
import { checkSyntax } from '../syntax.mjs';
import { contextPrompt, isWorkspaceRequest, workspaceReply } from '../workspace-chat.mjs';
const exec = promisify(execFile);
async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'kani-workspace-'));
  t.after(() => fs.rm(root, { recursive:true, force:true }));
  return { root, workspace: await Workspace.open(root) };
}

test('inventory respects gitignore, generated/private paths and symlinks', async t => {
  const { root, workspace } = await fixture(t);
  await exec('git', ['init','-q'], {cwd:root});
  await fs.writeFile(path.join(root,'.gitignore'), 'ignored.py\n');
  for (const file of ['main.py','ignored.py','.env.local','credentials.json','README.md']) await fs.writeFile(path.join(root,file),'visible\n');
  await fs.mkdir(path.join(root,'node_modules'));
  await fs.writeFile(path.join(root,'node_modules','dependency.js'),'visible');
  await fs.symlink(os.tmpdir(),path.join(root,'outside'));
  await fs.symlink(path.join(root,'main.py'),path.join(root,'linked.py'));
  const listing = await workspace.files();
  assert.deepEqual(listing.files,['.gitignore','README.md','main.py']);
  assert.deepEqual((await workspace.search('visible')).hits.map(h=>h.path), ['README.md','main.py']);
  await assert.rejects(workspace.read('../escape'), /inside/);
  await assert.rejects(workspace.read('.env.local'), /excluded/);
  await assert.rejects(workspace.read('linked.py'), /Symlinks/);
  await assert.rejects(workspace.stage('outside/leak.py','hello'), /Symlinks/);
});

test('proposals persist without touching files, apply once and preserve old content', async t => {
  const { root, workspace } = await fixture(t);
  await fs.writeFile(path.join(root,'a.py'),'old\n');
  const p = await workspace.stage('a.py','new\n');
  assert.equal(await fs.readFile(path.join(root,'a.py'),'utf8'),'old\n');
  const reopened = await Workspace.open(root);
  await reopened.apply(p.id);
  assert.equal(await fs.readFile(path.join(root,'a.py'),'utf8'),'new\n');
  assert.equal(await fs.readFile(path.join(root,'.kani/backups',p.id),'utf8'),'old\n');
  await assert.rejects(reopened.apply(p.id));
  const added = await workspace.stage('nested/new.py','new file\n');
  await workspace.apply(added.id);
  assert.equal(await fs.readFile(path.join(root,'nested/new.py'),'utf8'),'new file\n');
});

test('concurrent changes and a target symlink block apply without overwriting', async t => {
  const { root, workspace } = await fixture(t);
  await fs.writeFile(path.join(root,'a.py'),'before\n');
  const p = await workspace.stage('a.py','proposed\n');
  await fs.writeFile(path.join(root,'a.py'),'user edit\n');
  await assert.rejects(workspace.apply(p.id), /changed since/);
  assert.equal(await fs.readFile(path.join(root,'a.py'),'utf8'),'user edit\n');
  const q = await workspace.stage('b.py','proposed\n');
  await fs.symlink(path.join(root,'a.py'),path.join(root,'b.py'));
  await assert.rejects(workspace.apply(q.id), /Symlinks/);
  await workspace.discard(p.id);
  await assert.rejects(workspace.proposal(p.id));
});

test('malformed and escaping proposals cannot be applied', async t => {
  const { root, workspace } = await fixture(t);
  const p = await workspace.stage('a.py','valid\n');
  const store = path.join(root,'.kani/proposals',p.id+'.json');
  await fs.writeFile(store,JSON.stringify({...p,path:'../outside.py'}));
  await assert.rejects(workspace.apply(p.id),/inside/);
  assert.throws(()=>extractCode('Sorry, I cannot do that.'),/No file was staged/);
  assert.throws(()=>extractCode('```py\nprint(1)'),/No file was staged/);
  assert.equal(extractCode('```python\nprint(1)\n```'),'print(1)\n');
  assert.deepEqual(pathArgument('"folder with  spaces/a.py" fix this'),['folder with  spaces/a.py','fix this']);
});

test('source context is line-bounded, cites what fits and preserves the question', () => {
  const model={config:{context:160},tokenizer:{encode:s=>Array.from(s)}};
  const file={path:'a.py',lines:Array.from({length:25},(_,i)=>`line_${i}`)};
  const result=contextPrompt(model,'explain',file,{maxTokens:40,start:5,end:25});
  assert.match(result.prompt,/Question: explain/);
  assert.match(result.prompt,/5: line_4/);
  assert.match(result.note,/later lines omitted/);
  assert.ok(model.tokenizer.encode(result.prompt).length+3 <= 120);
  assert.throws(()=>contextPrompt(model,'x'.repeat(180),file),/too long/);
});

test('natural analyze routes to actual files; draft uses model output then requires apply',async t=>{
  const {root,workspace}=await fixture(t);
  await fs.writeFile(path.join(root,'main.py'),'def hello():\n    return 1\n');
  assert.ok(isWorkspaceRequest('analyze files'));
  const model={config:{context:1024},tokenizer:{encode:s=>Array.from(s)},*generate(){yield '```python\ndef square(x):\n    return x * x\n```';}};
  const events=[];
  for await(const event of workspaceReply(workspace,model,'analyze files',{maxTokens:128},{})) events.push(event);
  assert.match(events.map(e=>e.text||'').join(''),/main.py/);
  const draft=[];
  for await(const event of workspaceReply(workspace,model,'/draft "new  file.py" write square',{maxTokens:128},{})) draft.push(event);
  assert.match(draft.map(e=>e.text||'').join(''),/\/apply [a-f0-9-]+/);
  await assert.rejects(fs.stat(path.join(root,'new  file.py')),/ENOENT/);
  const saved=await fs.readdir(path.join(root,'.kani/proposals'));
  const p=await workspace.proposal(saved[0].replace('.json',''));
  assert.equal(p.path,'new  file.py');
  await workspace.apply(p.id);
  assert.equal(await fs.readFile(path.join(root,'new  file.py'),'utf8'),'def square(x):\n    return x * x\n');
});

test('CLI analyzes requested cwd and reads line ranges without feeding model',async t=>{
  const {root}=await fixture(t);
  await fs.writeFile(path.join(root,'hello.py'),'def answer():\n    return 42\n');
  const cli=new URL('../kani.mjs',import.meta.url).pathname;
  const {stdout}=await exec(process.execPath,[cli,'--cwd',root,'/read hello.py 2:2']);
  assert.match(stdout,/hello.py:2\s+return 42/);
  assert.doesNotMatch(stdout,/hello.py:1/);
});


test('syntax checks reject broken code and never execute generated statements',async t=>{
  const {root}=await fixture(t), marker=path.join(root,'must-not-exist');
  const source=`import fs from 'node:fs'; fs.writeFileSync(${JSON.stringify(marker)},'ran');`;
  assert.equal((await checkSyntax('a.js',source)).checked,true);
  await assert.rejects(fs.stat(marker),/ENOENT/);
  await assert.rejects(checkSyntax('a.js','function {'),/failed syntax/);
  assert.equal((await checkSyntax('a.ts','let n: number = 1')).checked,false);
});

test('one-shot tools do not load weights',async t=>{
  const {root}=await fixture(t); await fs.writeFile(path.join(root,'example.py'),'print(1)');
  const cli=new URL('../kani.mjs',import.meta.url).pathname;
  const {stdout}=await exec(process.execPath,[cli,'--cwd',root,'--model-dir',path.join(root,'missing'),'/files']);
  assert.match(stdout,/example.py/);
});


test('UTF-8 BOM files preserve byte identity for review and backup',async t=>{
  const {root,workspace}=await fixture(t);
  const source='\ufeffdef value():\n    return 1\n';
  await fs.writeFile(path.join(root,'bom.py'),source);
  const p=await workspace.stage('bom.py','def value():\n    return 2\n');
  assert.equal((await workspace.proposal(p.id)).before,source);
  await workspace.apply(p.id);
  assert.equal(await fs.readFile(path.join(root,'.kani/backups',p.id),'utf8'),source);
});

test('read refuses a zero-based range',async t=>{
  const {root,workspace}=await fixture(t);await fs.writeFile(path.join(root,'a.py'),'x=1\n');
  const consume=async()=>{for await(const event of workspaceReply(workspace,null,'/read a.py 0:1',{},{})) {}}
  await assert.rejects(consume(),/outside/);
});
