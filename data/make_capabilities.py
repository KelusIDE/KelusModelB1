"""Original capability curriculum: conversation relevance, SVG, and longer programs.

A separate candidate dataset; does not mutate instruct-v2 or a running trainer.
Validation holds out prompt wordings/combinations, not whole task families.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent.parent
FOODS=['pancakes','waffles','pizza','noodles','soup','sushi','dumplings','tacos','pasta','sandwiches','ice cream','rice','salad','curry','cookies','toast','ramen','fruit','chocolate','bread']
HOBBIES=['Minecraft','drawing','coding','music','basketball','chess','reading','running','gardening','photography','skating','swimming','cooking','hiking']
COLORS={'red':'#ef4444','blue':'#3b82f6','green':'#22c55e','orange':'#f97316','purple':'#a855f7','yellow':'#facc15','pink':'#ec4899','black':'#111827'}


def record(family,split,prompt,answer,history=()):
    return {'family':family,'split':split,'messages':[*history,{'role':'user','content':prompt},{'role':'assistant','content':answer}]}


def chat_rows():
    rows=[]
    for topic in FOODS+HOBBIES:
        food=topic in FOODS
        reply=f'{topic.capitalize()}! What do you like most about {topic}?'
        templates=['I love {}.','I really like {}.','I enjoy {}.','{} is my favorite.','I am into {}.','Can we talk about {}?','I like {} a lot.']
        for template in templates:
            rows.append(record('conversation','train',template.format(topic),reply))
        rows.append(record('conversation','validation',f'I am a big fan of {topic}!',reply))
        for prompt in [f'I do not like {topic}.',f'I hate {topic}.']:
            rows.append(record('conversation','train',prompt,f'Fair enough. What do you prefer instead of {topic}?'))
        first=[{'role':'user','content':f'I love {topic}.'},{'role':'assistant','content':reply}]
        rows.append(record('conversation','train','What did I say I liked?',f'You said you love {topic}.',first))
        if food:
            for topping in ['strawberries','blueberries','chocolate','honey']:
                rows.append(record('conversation','train',f'I like {topic} with {topping}.',f'{topic.capitalize()} with {topping} sounds nice. Do you make it yourself?'))
                rows.append(record('conversation','train',f'With {topping}.',f'{topping.capitalize()} with your {topic} sounds nice. Do you make it yourself?',first))
            rows.append(record('conversation','validation',f'I usually have {topic} with fruit.',f'{topic.capitalize()} with fruit sounds nice. What fruit do you like?'))
        for other in (FOODS if food else HOBBIES)[:5]:
            if other==topic:continue
            rows.append(record('conversation','train',f'No, I meant {other}.',f'Got it, {other}. What do you like about {other}?',first))
        long_history=first+[{'role':'user','content':'How are you?'},{'role':'assistant','content':'Ready to help. How is your day going?'}]
        rows.append(record('conversation','train','What do I like?',f'You said you love {topic}.',long_history))
    for prompt,answer in [
        ('Think before you answer.','I can break a task into steps, check the requirements, and test a proposed solution. What would you like to work on?'),
        ('Build me an app.','What should the app do, and should it run on the web or on your computer?'),
        ('Make a website.','What is the website for? Tell me its main pages and the actions visitors should be able to take.'),
        ('Can you do complex coding?','I can try a concrete specification, then check the result with tests. Larger tasks should be split into smaller parts. I may still make mistakes.'),
        ('You answered the wrong question.','Please point out what I missed. I will focus on your request.'),
        ('I did not ask about coding.','Understood. Let us stay with what you were talking about.'),
    ]: rows.append(record('conversation','train',prompt,answer))
    return rows


def svg_document(body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">{body}</svg>'


def svg_rows():
    rows=[]
    sizes={'small':32,'medium':56,'large':88}
    for color,hexcolor in COLORS.items():
        for size,radius in sizes.items():
            for shape in ['circle','square','triangle','heart']:
                body={'circle':f'<circle cx="128" cy="128" r="{radius}" fill="{hexcolor}"/>',
                      'square':f'<rect x="{128-radius}" y="{128-radius}" width="{radius*2}" height="{radius*2}" fill="{hexcolor}"/>',
                      'triangle':f'<polygon points="128,{128-radius} {128+radius},{128+radius} {128-radius},{128+radius}" fill="{hexcolor}"/>',
                      'heart':f'<g transform="translate(128 128) scale({radius/88:.4f}) translate(-128 -128)"><path d="M128 208 C16 140 24 48 80 48 C104 48 120 64 128 80 C136 64 152 48 176 48 C232 48 240 140 128 208 Z" fill="{hexcolor}"/></g>'}[shape]
                description=f'a {size} {color} {shape}'
                answer='```svg\n'+svg_document(body)+'\n```'
                for prompt in [f'Draw {description} as SVG.',f'Make an SVG of {description}.',f'Create {description} in SVG.']:
                    rows.append(record('svg','train',prompt,answer))
                rows.append(record('svg','validation',f'Generate an SVG image showing {description}.',answer))
        for second,other in COLORS.items():
            for orientation in ['left','above']:
                x,y=(64,128) if orientation=='left' else (128,64)
                bx,by=(160,96) if orientation=='left' else (96,160)
                body=f'<circle cx="{x}" cy="{y}" r="32" fill="{hexcolor}"/><rect x="{bx}" y="{by}" width="64" height="64" fill="{other}"/>'
                answer='```svg\n'+svg_document(body)+'\n```'
                split='validation' if (list(COLORS).index(color)+list(COLORS).index(second))%7==0 else 'train'
                for prompt in [f'Draw a {color} circle {orientation} of a {second} square as SVG.',f'Make an SVG with a {color} circle {orientation} of a {second} square.']:
                    rows.append(record('svg',split,prompt,answer))
    for color,hexcolor in COLORS.items():
        scenes={
            f'a {color} mountain under a yellow sun':f'<circle cx="196" cy="52" r="28" fill="#facc15"/><polygon points="16,224 128,64 240,224" fill="{hexcolor}"/>',
            f'a {color} house with a brown roof':f'<rect x="64" y="112" width="128" height="112" fill="{hexcolor}"/><polygon points="40,112 128,32 216,112" fill="#92400e"/><rect x="112" y="160" width="32" height="64" fill="#111827"/>',
            f'a {color} flower with a green stem':f'<line x1="128" y1="128" x2="128" y2="232" stroke="#22c55e" stroke-width="12"/><circle cx="96" cy="96" r="32" fill="{hexcolor}"/><circle cx="160" cy="96" r="32" fill="{hexcolor}"/><circle cx="128" cy="144" r="32" fill="{hexcolor}"/><circle cx="128" cy="112" r="24" fill="#facc15"/>',
        }
        for description,body in scenes.items():
            rows.append(record('svg','train',f'Draw {description} as SVG.','```svg\n'+svg_document(body)+'\n```'))
            rows.append(record('svg','validation',f'Create an SVG illustration of {description}.','```svg\n'+svg_document(body)+'\n```'))
    for row in rows: ET.fromstring(row['messages'][-1]['content'].split('\n',1)[1].rsplit('\n',1)[0])
    return rows


def coding_rows():
    rows=[]
    for suffix in ['','_items','_values','_records','_data','_entries']:
        implementations=[
            (f'group{suffix or "_by"}', 'group a list of dictionaries by a requested key without modifying the input',
             '''def NAME(records, key):
    groups = {}
    for record in records:
        value = record.get(key)
        groups.setdefault(value, []).append(record)
    return groups
'''),
            (f'flatten{suffix}', 'flatten nested lists of any depth while preserving item order',
             '''def NAME(items):
    result = []
    stack = [iter(items)]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, list):
            stack.append(iter(item))
        else:
            result.append(item)
    return result
'''),
            (f'merge{suffix or "_ranges"}', 'merge overlapping intervals, sort the result, and preserve the original input',
             '''def NAME(intervals):
    result = []
    for start, end in sorted(intervals):
        if start > end:
            raise ValueError("interval start exceeds end")
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''),
            (f'parse{suffix or "_rows"}', 'parse CSV text with headers, quoted commas, and quoted newlines using the standard library',
             '''import csv
import io

def NAME(text):
    return list(csv.DictReader(io.StringIO(text)))
'''),
        ]
        for name,spec,source in implementations:
            code=source.replace('NAME',name);ast.parse(code)
            for wording in ['Write a Python function named NAME to SPEC.','Implement NAME in Python: SPEC. Include edge cases.','I need NAME in Python. It should SPEC.']:
                prompt=wording.replace('NAME',name).replace('SPEC',spec)
                rows.append(record('deeper_code','train',prompt,'```python\n'+code+'```'))
            rows.append(record('deeper_code','validation',f'Please implement {name} to {spec}. Use Python.','```python\n'+code+'```'))
            old=f'def {name}(items):\n    return items\n'
            rows.append(record('source_edit','train',f'Update this Python code to {spec}. Keep the name {name}.\n```python\n{old}```','```python\n'+code+'```'))
    for class_name in ['LRUCache','RecentCache','BoundedCache','AccessCache']:
        code=f'''from collections import OrderedDict

class {class_name}:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, key, default=None):
        if key not in self.items:
            return default
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
'''
        for wording in [f'Implement a Python {class_name} with a positive capacity, get and put. Reading updates recency. Evict the least recently used item.',f'Write a Python class called {class_name} for an LRU cache. Support get(key, default=None) and put(key, value), rejecting nonpositive capacities.']:
            rows.append(record('deeper_code','train',wording,'```python\n'+code+'```'))
    js='''function debounce(fn, delay) {
  let timer;
  function wrapped(...args) {
    clearTimeout(timer);
    const context = this;
    timer = setTimeout(() => fn.apply(context, args), delay);
  }
  wrapped.cancel = () => clearTimeout(timer);
  return wrapped;
}
'''
    rows.append(record('deeper_code','train','Write JavaScript debounce(fn, delay), preserving this and arguments and supporting cancel().','```javascript\n'+js+'```'))
    return rows


def build(seed=2030, replay=1200):
    rows=chat_rows()+svg_rows()+coding_rows()
    source=ROOT/'data/instruct-v2.jsonl'
    previous=[json.loads(line) for line in source.read_text().splitlines()]
    rng=random.Random(seed)
    for split,count in [('train',replay),('validation',80)]:
        candidates=[row for row in previous if row['split']==split]
        rows.extend(dict(row,family='replay') for row in rng.sample(candidates,min(count,len(candidates))))
    unique={json.dumps(row['messages'],sort_keys=True):row for row in rows}
    result=list(unique.values());rng.shuffle(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'data/capabilities-v2.jsonl');args=parser.parse_args()
    rows=build();data=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    if args.out.exists() and args.out.read_text()!=data:parser.error('Output exists with different content; choose a new dataset path.')
    args.out.write_text(data)
    print(json.dumps({'path':str(args.out),'rows':len(rows),'families':dict(Counter(row['family'] for row in rows)),'sha256':hashlib.sha256(data.encode()).hexdigest()},indent=2))

if __name__=='__main__':main()
