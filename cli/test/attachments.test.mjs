import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { decodeText, validateBatch, imageHeader, prepareAttachments, readAttachment, fileName } from '../../site/attachments.js';
import { Tokenizer, conversation } from '../../site/b1.js';
const enc=new TextEncoder(), tokenizer=new Tokenizer(JSON.parse(await readFile(new URL('../../site/model/b1.json',import.meta.url))).tokenizer);
const textFile=(name,text)=>decodeText(name,enc.encode(text));

test('HTML is inspected as source and malicious markup stays literal text',async()=>{
 const source='<title>Demo</title>\n<h1>Hello</h1>\n<script>globalThis.__attachmentExecuted = true</script>\n<img src="https://invalid.example/tracker">';
 const f=await readAttachment(new File([source],'page.html',{type:'text/html'}));
 assert.match(f.report,/Title source: Demo/);assert.match(f.report,/1 headings, 0 links, 1 images, 1 script/);
 assert.equal(f.text,source);assert.equal(globalThis.__attachmentExecuted,undefined);
});
test('rejects binary and unsupported formats, size and count limits',()=>{
 assert.throws(()=>decodeText('x.txt',Uint8Array.of(0,1,2)),/binary/);
 assert.throws(()=>decodeText('x.txt',Uint8Array.of(255,254)),/UTF-8/);
 assert.throws(()=>validateBatch([{name:'report.docx',size:200}]),/unsupported/);
 validateBatch([{name:'report.pdf',size:20*1024*1024}]);
 assert.throws(()=>validateBatch([{name:'report.pdf',size:20*1024*1024+1}]),/20 MB maximum/);
 assert.throws(()=>validateBatch([{name:'big.py',size:128*1024+1}]),/128 KB/);
 assert.throws(()=>validateBatch(Array(5).fill({name:'x.txt',size:2})),/4 files/);
 assert.throws(()=>validateBatch(Array(3).fill({name:'x.png',size:8*1024*1024})),/20 MB/);
 assert.equal(fileName('../x\n.html'),'x.html');
});
test('PNG dimensions checked before decoder allocations',()=>{
 const bytes=new Uint8Array(32);bytes.set([137,80,78,71,13,10,26,10]);const view=new DataView(bytes.buffer);
 view.setUint32(16,640);view.setUint32(20,480);assert.deepEqual(imageHeader(bytes),{width:640,height:480,mime:'image/png'});
 view.setUint32(16,50000);assert.throws(()=>imageHeader(bytes),/too large/);
 assert.throws(()=>imageHeader(enc.encode('<svg/>')),/not a supported/);
});
test('generic inspection works without weights and JSON parsing errors are explicit',()=>{
 const f=textFile('settings.json','{"theme":"dark"}');
 const result=prepareAttachments('analyze this file',[f]);assert.equal(result.helper,true);assert.match(result.text,/JSON keys: theme/);
 assert.match(textFile('broken.json','{"x":').report,/invalid/);
});
test('specific file questions use actual text and exact tokenizer budget',()=>{
 const f=textFile('demo.py','def square(x):\n    return x * x');
 const result=prepareAttachments('What does square return?',[f],tokenizer,256,240);
 assert.equal(result.helper,false);assert.match(result.text,/return x \* x/);assert.match(result.receipt,/lines 1, 2 of 2/);
 assert.ok(conversation([{role:'user',content:result.text}],tokenizer).length<=256-Math.floor(256/3));
});
test('large source selects relevant whole lines and reports partial coverage',()=>{
 const lines=Array.from({length:250},(_,i)=>`# note ${i}`);lines[180]='def calculate_discount(price):';lines[181]='    return price * 0.8';
 const result=prepareAttachments('Explain calculate_discount',[textFile('large.py',lines.join('\n'))],tokenizer,256,240);
 assert.match(result.text,/181: def calculate_discount/);assert.match(result.receipt,/excerpt only/);
 assert.ok(conversation([{role:'user',content:result.text}],tokenizer).length<=171);
});
test('overlong complete source line is refused instead of silently truncated',()=>{
 assert.throws(()=>prepareAttachments('Explain',[textFile('x.js','a'.repeat(3000))],tokenizer,128,64),/no complete source line/);
});
test('attached files cannot be routed to online search',()=>{
 assert.throws(()=>prepareAttachments('/web private source',[textFile('x.txt','private')],tokenizer,256),/stay local/);
});
test('images never masquerade as visual understanding',()=>{
 const result=prepareAttachments('What is the person doing?',[{name:'pic.png',size:400,kind:'image',report:'100 × 100 pixels. Pixel inspection only; cannot recognize objects.'}]);
 assert.equal(result.helper,true);assert.match(result.text,/cannot recognize objects/);
});
test('multiple attachments share context and whitespace cannot hide oversized source',()=>{
 const many=textFile('first.py',Array.from({length:200},(_,i)=>`a${i} = ${i}`).join('\n'));
 const small=textFile('second.py','result = 42');
 const result=prepareAttachments('Explain result',[many,small],tokenizer,256,240);
 assert.match(result.text,/result = 42/);assert.match(result.receipt,/second.py: lines 1 of 1/);
 assert.throws(()=>prepareAttachments('Explain',[textFile('x.js','a'.repeat(3000)+'\n\n')],tokenizer,128,64),/no complete source line/);
});
test('decoded image pixels produce a labeled inspection and release the bitmap',async()=>{
 const bytes=new Uint8Array(32);bytes.set([137,80,78,71,13,10,26,10]);const view=new DataView(bytes.buffer);view.setUint32(16,2);view.setUint32(20,2);
 const oldBitmap=globalThis.createImageBitmap, oldDocument=globalThis.document;let closed=false;
 globalThis.createImageBitmap=async()=>({width:2,height:2,close(){closed=true;}});
 globalThis.document={createElement:()=>({getContext:()=>({drawImage(){},getImageData:()=>({data:Uint8Array.of(255,0,0,255,0,0,255,255)})})})};
 try{const f=await readAttachment(new File([bytes],'test.png'));assert.equal(f.kind,'image');assert.match(f.report,/2 × 2/);assert.match(f.report,/#800080/);assert.match(f.report,/cannot read image text/);assert.ok(closed);}
 finally{globalThis.createImageBitmap=oldBitmap;globalThis.document=oldDocument;}
});
