
const TAGS = new Set(['svg','g','rect','circle','ellipse','line','polyline','polygon','path','text','title','desc']);
const COMMON = ['fill','stroke','stroke-width','opacity','fill-opacity','stroke-opacity','stroke-linecap','stroke-linejoin','transform'];
const ATTRS = {
  svg:['xmlns','viewBox','width','height'], g:[], rect:['x','y','width','height','rx','ry'], circle:['cx','cy','r'],
  ellipse:['cx','cy','rx','ry'], line:['x1','y1','x2','y2'], polyline:['points'], polygon:['points'], path:['d','fill-rule'],
  text:['x','y','font-size','font-family','font-weight','text-anchor'], title:[], desc:[],
};
const NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;
const COLOR = /^(?:#[a-f\d]{3}(?:[a-f\d]{3})?|none|transparent|black|white|red|blue|green|yellow|orange|purple|pink|brown|gray|grey|navy|teal|cyan|magenta|gold)$/i;
const escape = value => value.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
function numeric(value) { return NUMBER.test(value) && Number.isFinite(Number(value)) && Math.abs(Number(value)) <= 1e6; }
function numbers(value) { const values=value.trim().split(/[\s,]+/); if(!values.every(numeric))throw new Error('Invalid SVG coordinates.');return values.map(Number); }
function attribute(name,value,tag) {
  if(![...COMMON,...ATTRS[tag]].includes(name))throw new Error(`SVG attribute ${name} is not allowed.`);
  if (name==='xmlns') { if(tag!=='svg'||value!=='http://www.w3.org/2000/svg')throw new Error('Invalid SVG namespace.'); }
  else if(name==='viewBox') { const n=numbers(value);if(n.length!==4||n[2]<=0||n[3]<=0||n[2]>4096||n[3]>4096)throw new Error('Invalid SVG viewBox.'); }
  else if(['fill','stroke'].includes(name)) { if(!COLOR.test(value))throw new Error('SVG paints must be plain colors, not URLs or CSS.'); }
  else if(name==='points') { const n=numbers(value);if(n.length<4||n.length%2)throw new Error('Invalid SVG point list.'); }
  else if(name==='d') {
    if(!/^[MmLlHhVvCcSsQqTtAaZz0-9eE.,+\s-]+$/.test(value)||!/^\s*[Mm]/.test(value)||value.length>12000)throw new Error('Invalid SVG path.');
    for (const n of value.match(/[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?/gi)||[]) if(!numeric(n))throw new Error('Unbounded SVG path coordinate.');
  }
  else if(name==='transform') { if(!/^(?:(?:translate|scale|rotate|matrix|skewX|skewY)\([+\d.eE,\s-]+\)\s*)+$/.test(value))throw new Error('Invalid SVG transform.'); for(const group of value.matchAll(/\(([^)]+)\)/g))numbers(group[1]); }
  else if(name==='font-family') { if(!/^(?:sans-serif|serif|monospace|Arial|Helvetica|Georgia)$/.test(value))throw new Error('Unsupported SVG font.'); }
  else if(name==='font-weight') { if(!/^(?:normal|bold|[1-9]00)$/.test(value))throw new Error('Invalid SVG font weight.'); }
  else if(name==='text-anchor') { if(!/^(?:start|middle|end)$/.test(value))throw new Error('Invalid SVG text anchor.'); }
  else if(name==='stroke-linecap') { if(!/^(?:butt|round|square)$/.test(value))throw new Error('Invalid line cap.'); }
  else if(name==='stroke-linejoin') { if(!/^(?:miter|round|bevel)$/.test(value))throw new Error('Invalid line join.'); }
  else if(name==='fill-rule') { if(!/^(?:nonzero|evenodd)$/.test(value))throw new Error('Invalid fill rule.'); }
  else {
    if(!numeric(value))throw new Error('Expected a bounded SVG number.');
    if(['width','height','r','rx','ry','stroke-width','font-size'].includes(name)&&Number(value)<0)throw new Error('Negative SVG size.');
    if(name.includes('opacity')&&(Number(value)<0||Number(value)>1))throw new Error('Invalid SVG opacity.');
    if(tag==='svg'&&['width','height'].includes(name)&&(Number(value)<=0||Number(value)>4096))throw new Error('SVG dimensions must be 1–4096.');
  }
  return `${name}="${escape(value)}"`;
}

export function extractSvg(reply) {
  const blocks=[...reply.matchAll(/```(?:svg|xml|html)?[ \t]*\r?\n([\s\S]*?)^```[ \t]*$/gm)];
  const source=blocks.length===1?blocks[0][1].trim():reply.trim();
  return validateSvg(source);
}

export function validateSvg(source) {
  if(typeof source!=='string'||source.length>50000||/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(source)||/<!|<\?/.test(source))throw new Error('SVG must be a bounded document without entities, comments or declarations.');
  const tokens=source.match(/<[^>]*>|[^<]+/g)||[], stack=[], output=[];
  if(tokens.join('')!==source)throw new Error('Incomplete SVG markup.');
  let roots=0,elements=0,viewBox=null;
  for(const token of tokens){
    if(!token.startsWith('<')){
      if(!token.trim())continue;
      if(!['text','title','desc'].includes(stack.at(-1)))throw new Error('Text is only allowed inside SVG text elements.');

      if(/&(?!amp;|lt;|gt;|quot;|apos;)/.test(token))throw new Error('Unsupported SVG entity.');
      output.push(token);continue;
    }
    const closing=token.match(/^<\/([a-z]+)\s*>$/);
    if(closing){if(stack.pop()!==closing[1])throw new Error('Mismatched SVG closing tag.');output.push(`</${closing[1]}>`);continue;}
    const open=token.match(/^<([a-z]+)([\s\S]*?)(\/?)>$/);
    if(!open||!TAGS.has(open[1]))throw new Error('Unsupported SVG element.');
    const [,tag,raw,self]=open;
    if(++elements>256)throw new Error('SVG element limit exceeded.');
    if(!stack.length){if(tag!=='svg'||++roots!==1)throw new Error('Expected one SVG root.');}
    else if(tag==='svg')throw new Error('Nested SVG roots are not supported.');
    if(stack.length>16)throw new Error('SVG nesting limit exceeded.');
    const attrs=[], seen=new Set();let rest=raw;
    while(rest.trim()){
      const match=rest.match(/^\s+([A-Za-z][A-Za-z0-9-]*)\s*=\s*(?:"([^"<>]*)"|'([^'<>]*)')/);
      if(!match)throw new Error('Malformed SVG attributes.');
      const name=match[1],value=match[2]??match[3];
      if(seen.has(name))throw new Error('Duplicate SVG attribute.');seen.add(name);
      attrs.push(attribute(name,value,tag));if(tag==='svg'&&name==='viewBox')viewBox=value;
      rest=rest.slice(match[0].length);
    }
    if(tag==='svg'&&!seen.has('xmlns'))attrs.unshift('xmlns="http://www.w3.org/2000/svg"');
    output.push(`<${tag}${attrs.length?' '+attrs.join(' '):''}${self?'/':''}>`);
    if(!self)stack.push(tag);
  }
  if(roots!==1||stack.length||!viewBox)throw new Error('SVG must have a complete root and a viewBox.');
  return output.join('');
}

export function svgRequest(text) {
  const slash=text.trim().match(/^\/svg\s+([\s\S]+)/i);
  if(slash)return slash[1];
  const natural=text.trim().match(/^(?:please\s+)?(?:draw|make|create|generate)\s+(?:an?\s+)?SVG(?:\s+(?:image|illustration))?\s+(?:of|showing|with)\s+([\s\S]+)/i);
  const suffix=text.trim().match(/^(?:please\s+)?(?:draw|make|create|generate)\s+([\s\S]+?)\s+(?:as|in)\s+SVG[.!]?$/i);
  return natural?natural[1]:suffix?suffix[1]:null;
}
export function svgPrompt(description) {
  let text=description.trim().replace(/[.!]+$/,'').replace(/^an? (?:big|huge) /i,'a large ').replace(/^an? (?:little|tiny) /i,'a small ');

  text=text.replace(/^a (red|blue|green|orange|purple|yellow|pink|black) (circle|square|triangle|heart)$/i,'a medium $1 $2');
  return `Draw ${text} as SVG.`;
}


export function attachSvgPreview(body,reply) {
  let svg;try{svg=extractSvg(reply);}catch{return null;}
  const url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));
  const figure=document.createElement('figure');figure.className='svg-preview';
  const img=document.createElement('img');img.src=url;img.alt='SVG generated by Kani';img.width=256;img.height=256;
  const caption=document.createElement('figcaption');caption.textContent='Generated SVG · review the source below';
  const download=document.createElement('a');download.href=url;download.download='kani-image.svg';download.textContent='Download SVG';
  figure.append(img,caption,download);body.before(figure);
  return ()=>URL.revokeObjectURL(url);
}
