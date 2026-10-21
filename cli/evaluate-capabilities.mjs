/** Score actual model output. Generated Python executes only in the existing OS sandbox. */
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { extractSvg } from '../site/svg.js';
import { Workspace, extractCode } from './workspace.mjs';
import { sandboxTest } from './agent.mjs';

export const CODE_TESTS = {
  lru: `from candidate import LRUCache
for capacity in (0,-1):
    try: LRUCache(capacity)
    except ValueError: pass
    else: raise AssertionError("reject nonpositive capacity")
c=LRUCache(2)
c.put("a",1);c.put("b",2)
assert c.get("a")==1
c.put("c",3)
assert c.get("b","missing")=="missing"
assert c.get("a")==1 and c.get("c")==3
c.put("a",9);c.put("d",4)
assert c.get("c") is None and c.get("a")==9
`,
  group: `from candidate import group_by
import copy
rows=[{"k":"a","v":1},{"k":"b","v":2},{"k":"a","v":3},{"v":4}]
original=copy.deepcopy(rows)
assert group_by(rows,"k")=={"a":[rows[0],rows[2]],"b":[rows[1]],None:[rows[3]]}
assert rows==original
assert group_by([],"k")=={}
`,
  flatten: `from candidate import flatten
assert flatten([1,[2,[],[3]],"hi",None])==[1,2,3,"hi",None]
assert flatten([])==[]
deep=7
for _ in range(1500): deep=[deep]
assert flatten(deep)==[7]
`,
  intervals: `from candidate import merge_ranges
import copy
source=[[8,10],[1,4],[3,6],[10,12],[14,14]]
before=copy.deepcopy(source)
assert merge_ranges(source)==[[1,6],[8,12],[14,14]]
assert source==before and merge_ranges([])==[]
assert merge_ranges([[1,9],[2,3]])==[[1,9]]
try: merge_ranges([[3,1]])
except ValueError: pass
else: raise AssertionError("reject reversed interval")
`,
  csv: `from candidate import parse_rows
assert parse_rows('name,note\\n"Ada","a,b"\\n"Lin","two\\nlines"\\n')==[{"name":"Ada","note":"a,b"},{"name":"Lin","note":"two\\nlines"}]
assert parse_rows("")==[]
assert parse_rows('a,b\\n"x""y",z\\n')==[{"a":'x"y',"b":"z"}]
`,
};

async function score(file) {
  const report=JSON.parse(await fs.readFile(file,'utf8'));
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'kani-capability-eval-'));
  const workspace=await Workspace.open(root);
  try {
    for(const row of report.cases) {
      row.passed=false;
      try {
        if(row.family.endsWith('chat')) {
          row.check='required topic terms; inspect full reply for coherence';
          row.passed=row.expected.every(word=>row.reply.toLowerCase().includes(word));
        } else if(row.family==='svg') {
          const svg=extractSvg(row.reply), {tag,attrs,contains}=row.expected;
          const tags=svg.match(new RegExp('<'+tag+'\\b[^>]*>','g'))||[];
          row.passed=tags.some(value=>Object.entries(attrs).every(([key,v])=>value.includes(`${key}="${v}"`))) && (!contains||svg.includes(contains));
          row.check='restricted SVG parse plus requested shape/color/geometry';
        } else {
          const code=extractCode(row.reply), tests=CODE_TESTS[row.expected];
          await fs.writeFile(path.join(root,'test_candidate.py'), tests);
          const result=await sandboxTest(workspace,'candidate.py',code,'python3 test_candidate.py');
          row.passed=result.ran && result.passed;
          row.check='sandboxed behavioral assertions';
          if(!row.passed) row.check_error=result.out.slice(-1600);
        }
      } catch(error) { row.check_error=error.message; }
    }
    report.scores={};
    for(const row of report.cases) {
      const value=report.scores[row.family]??={passed:0,total:0};
      value.total++;value.passed+=Number(row.passed);
    }
    await fs.writeFile(file,JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({file,scores:report.scores}));
  } finally { await fs.rm(root,{recursive:true,force:true}); }
}
if (process.argv[1] && path.resolve(process.argv[1])===new URL(import.meta.url).pathname) {
  if(!process.argv[2])throw new Error('Usage: node cli/evaluate-capabilities.mjs REPORT.json');
  await score(process.argv[2]);
}
