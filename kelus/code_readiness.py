"""Check new-task coding readiness separately from memorized snippet benchmarks.

Never executes generated code. Python uses the existing bounded expression
interpreter. Website checks inspect structure only, not browser behavior.
"""
import argparse
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from .model import Model
from .evaluate import code_result

CASES = [
    ('three_inputs', 'Write a Python function add_three(a, b, c) that adds all three arguments. Return only the function.',
     'add_three', [((1,2,3),6),((-2,4,0),2),((0,0,0),0)]),
    ('new_composition', 'Write a Python function cube(x) that returns x multiplied by itself three times. Return only the function.',
     'cube', [((3,),27),((-2,),-8),((0,),0)]),
    ('follow_specification', 'Write a Python function final_price(price, tax_rate) that returns price * (1 + tax_rate). Return only the function.',
     'final_price', [((80,0.25),100.0),((40,0.5),60.0)]),
    ('custom_name', 'Write a Python function triple_score(score) that returns score * 3. Use exactly that function name. Return only the function.',
     'triple_score', [((4,),12),((-2,),-6),((0,),0)]),
]
WEBSITE = 'Build a complete single-file HTML website with inline CSS and JavaScript. Include a heading, a visible counter starting at 0, and a button that increases the counter when clicked. Return the complete HTML document.'

class Structure(HTMLParser):
    def __init__(self):
        super().__init__(); self.tags=set(); self.script=False; self.style=False; self.scripts=[]; self.styles=[]
    def handle_starttag(self,tag,attrs):
        self.tags.add(tag)
        if tag=='script':self.script=True
        if tag=='style':self.style=True
    def handle_endtag(self,tag):
        if tag=='script':self.script=False
        if tag=='style':self.style=False
    def handle_data(self,data):
        if self.script:self.scripts.append(data)
        if self.style:self.styles.append(data)

def run(checkpoint):
    model,header,_=Model.load(checkpoint)
    results=[]
    for name,prompt,function,trials in CASES:
        reply=''.join(model.generate([{'role':'user','content':prompt}],max_tokens=240,temperature=0))
        results.append({'name':name,'prompt':prompt,'reply':reply,'passed':code_result(reply,function,trials)})
    html=''.join(model.generate([{'role':'user','content':WEBSITE}],max_tokens=240,temperature=0))
    parser=Structure();parser.feed(html)
    checks={'html_document':bool(re.search(r'<!doctype\s+html',html,re.I)) and 'html' in parser.tags,
            'heading':bool(parser.tags & {'h1','h2'}),'button':'button' in parser.tags,
            'inline_css':bool(''.join(parser.styles).strip()),'inline_javascript':bool(''.join(parser.scripts).strip())}
    return {'checkpoint':str(checkpoint),'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'parameters':model.parameter_count,'steps':header['metadata'].get('steps'),
        'scope':'Small development readiness check prompted by the requested coding/website capability. Not a general intelligence benchmark. Website checks are structural only and do not establish functionality.',
        'python_passed':sum(r['passed'] for r in results),'python_total':len(results),'python':results,
        'website':{'prompt':WEBSITE,'reply':html,'structure_checks':checks,'behavior_tested':False}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',default='checkpoints/b1-v3.best.npz');p.add_argument('--out',default='reports/code-readiness-v3.json');a=p.parse_args()
    report=run(a.checkpoint);Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'python_passed':report['python_passed'],'python_total':report['python_total'],'website_structure':report['website']['structure_checks'],'report':a.out}))
    for r in report['python']:print(r['name'],repr(r['reply']))
    print('website',repr(report['website']['reply']))
