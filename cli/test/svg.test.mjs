import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { validateSvg, extractSvg, svgRequest, svgPrompt } from '../../site/svg.js';
import { Workspace } from '../workspace.mjs';
import { workspaceReply } from '../workspace-chat.mjs';

const SVG='<svg viewBox="0 0 256 256"><circle cx="128" cy="128" r="32" fill="#3b82f6"/></svg>';
test('SVG parser accepts the entire original image curriculum', async () => {
  const data=await fs.readFile(new URL('../../data/capabilities-v2.jsonl',import.meta.url),'utf8');
  for (const row of data.trim().split('\n').map(JSON.parse).filter(r=>r.family==='svg')) {
    assert.match(extractSvg(row.messages.at(-1).content), /^<svg xmlns=/);
  }
  assert.match(extractSvg('```svg\n'+SVG+'\n```'),/<circle/);
  assert.equal(svgPrompt(svgRequest('Generate an SVG image of a blue circle.')), 'Draw a medium blue circle as SVG.');
  assert.equal(svgPrompt('a big blue circle'), 'Draw a large blue circle as SVG.');
  assert.equal(svgRequest('Draw a blue circle as SVG.'),'a blue circle');
  assert.equal(svgRequest('Explain what SVG is'),null);
});

test('SVG rejects active markup, external content, broken XML and resource extremes', () => {
  const bodies=[
    '<script>alert(1)</script>','<foreignObject/>','<image href="https://example.com/x"/>',
    '<circle onload="alert(1)"/>','<circle fill="url(https://example.com/x)"/>',
    '<circle style="fill:red"/>','<circle r="1e99"/>','<path d="M0 0 L1e99 0"/>',
    '<g><circle></g>','<svg viewBox="0 0 1 1"/>','<text>&evil;</text>',
    '<circle r="1" r="2"/>','<g>'.repeat(20)+'</g>'.repeat(20),
  ];
  for(const body of bodies)assert.throws(()=>validateSvg(`<svg viewBox="0 0 1 1">${body}</svg>`),body);
  for(const value of ['<!DOCTYPE svg>'+SVG, SVG+SVG, SVG.slice(0,-1),SVG.replace('256 256','9000 256')])assert.throws(()=>validateSvg(value));
});

test('SVG command stages model output and a preview; invalid output never touches target', async t => {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'kani-svg-'));
  t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const workspace=await Workspace.open(root);
  let prompt='';
  const model={*generate(messages){prompt=messages[0].content;yield '```svg\n'+SVG+'\n```';}};
  let text='';
  for await(const event of workspaceReply(workspace,model,'/svg blue.svg a blue circle',{maxTokens:512},{}))text+=event.text||'';
  assert.equal(prompt,'Draw a medium blue circle as SVG.');
  assert.match(text,/Validated SVG preview/);
  await assert.rejects(fs.stat(path.join(root,'blue.svg')),/ENOENT/);
  const id=text.match(/\/apply ([a-f0-9-]+)/)[1];
  assert.match(await fs.readFile(path.join(root,'.kani/previews',id+'.svg'),'utf8'),/<circle/);
  await workspace.apply(id);
  assert.match(await fs.readFile(path.join(root,'blue.svg'),'utf8'),/<circle/);
  const bad={*generate(){yield '<svg><script>bad</script></svg>';}};
  await assert.rejects(async()=>{for await(const event of workspaceReply(workspace,bad,'/svg blue.svg red',{maxTokens:512},{})){}},/No file was staged/);
  assert.match(await fs.readFile(path.join(root,'blue.svg'),'utf8'),/#3b82f6/);
});
