"""Small handwritten smoke checks for actual Cide code and learned delegation.

These are narrow arithmetic/repair checks, not a general coding benchmark.
No model-generated Python is executed unless its AST contains only one pure
arithmetic function. Results preserve every prompt and response for inspection.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
from .model import Model

ROUTES = [
    ('Write a Python function add(a, b) that returns a + b.', 'cide'),
    ('Write a JavaScript function multiply(a, b) that returns a * b.', 'cide'),
    ('Can you make a function in Python that adds two numbers?', 'cide'),
    ('Create a simple web page with a button and a heading.', 'cide'),
    ('Fix this Python function. It should add both arguments. Return only the function.\ndef sum_two(a, b):\n    return a - b', 'cide'),
    ('hi', 'chat'),
    ('I feel sad today', 'chat'),
    ('Explain recursion', 'chat'),
    ('who is the president of France', 'search'),
    ('latest news about NASA', 'search'),
]
CASES = [
    ('Write a Python function add(a, b) that returns a + b.', 'add', [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
    ('Write a Python function multiply(a, b) that returns a * b.', 'multiply', [(2, 3, 6), (-4, 9, -36), (0, 8, 0)]),
    ('Write a Python function subtract(a, b) that returns a - b.', 'subtract', [(2, 3, -1), (-4, 9, -13), (0, 0, 0)]),
    ('Write a Python function cide_probe_sum(left, right) that returns left + right.', 'cide_probe_sum', [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
    ('Fix this Python function. It should add both arguments. Return only the function.\ndef sum_two(a, b):\n    return a - b', 'sum_two', [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
    ('This code subtracts instead of adding. Fix it:\n```python\ndef sum_pair(a, b):\n    return a - b\n```', 'sum_pair', [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
]
JS_CASES = [
    ('Write a JavaScript function add(a, b) that returns a + b.', 'add', [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
    ('Write a JavaScript function multiply(a, b) that returns a * b.', 'multiply', [(2, 3, 6), (-4, 9, -36), (0, 8, 0)]),
    ('Fix this JavaScript function. It should add both arguments. Return only the function.\nfunction sumPair(a, b) {\n  return a - b;\n}', 'sumPair', [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
]


def check_javascript(response, name, cases):
    block = re.search(r'```(?:javascript|js)?\n([\s\S]*?)```', response)
    code = (block[1] if block else response).strip()

    match = re.fullmatch(r'function\s+(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{\s*return\s+([\w\s()+*/.-]+);?\s*\}', code)
    if not match or match[1] != name or set(re.findall(r'[A-Za-z_]\w*', match[4])) - {match[2], match[3]}:
        return {'passed': False, 'error': 'Expected one plain arithmetic JavaScript function with the requested name'}
    if '.' in match[4] or re.search(r'\w\s*\(', match[4]):
        return {'passed': False, 'error': 'Calls and property access are outside this arithmetic smoke check'}
    script = """const fs = require('node:fs'), vm = require('node:vm');
const {code, name, cases} = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(([a,b,want]) => vm.runInNewContext(code+'; '+name+'('+a+','+b+')',
  Object.create(null), {timeout:100,contextCodeGeneration:{strings:false,wasm:false}}) === want)));"""
    try:
        result = subprocess.run(['node', '-e', script], input=json.dumps({'code': code, 'name': name, 'cases': cases}),
                                capture_output=True, text=True, timeout=5, check=True)
        checks = json.loads(result.stdout)
        return {'passed': all(checks), 'checks': checks}
    except Exception as error:
        return {'passed': False, 'error': str(error)}


def check_python(response, name, cases):
    block = re.search(r'```(?:python|py)?\n([\s\S]*?)```', response)
    code = block[1] if block else response
    try:
        tree = ast.parse(code)
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.BinOp,
                   ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
            raise ValueError('Expected the requested function name')
        function = tree.body[0]
        if function.decorator_list or function.args.defaults or function.args.kw_defaults or function.returns:
            raise ValueError('Only a plain arithmetic function is allowed in this smoke check')
        if any(not isinstance(node, allowed) or isinstance(node, ast.Constant) and not isinstance(node.value, (int, float))
               for node in ast.walk(tree)):
            raise ValueError('Generated code is outside the arithmetic smoke-check subset')
        namespace = {'__builtins__': {}}
        exec(compile(tree, '<cide-smoke>', 'exec'), namespace)
        results = [namespace[name](a, b) == expected for a, b, expected in cases]
        return {'passed': all(results), 'checks': results}
    except Exception as error:
        return {'passed': False, 'error': str(error)}


def evaluate(checkpoint, role):
    model, header, _ = Model.load(checkpoint)
    rows = []
    if role == 'router':
        for prompt, expected in ROUTES:
            text = ''.join(model.generate([{'role': 'user', 'content': prompt}], max_tokens=160, temperature=0))
            actual = 'cide' if text.strip() == 'CIDE' else 'search' if re.fullmatch(r'SEARCH: .{1,300}', text.strip()) else 'chat'
            rows.append({'prompt': prompt, 'expected': expected, 'actual': actual, 'response': text, 'passed': actual == expected})
    else:
        for suite, check in [(CASES, check_python), (JS_CASES, check_javascript)]:
            for prompt, name, cases in suite:
                text = ''.join(model.generate([{'role': 'user', 'content': prompt}], max_tokens=160, temperature=0))
                rows.append({'prompt': prompt, 'response': text, **check(text, name, cases)})
    return {'checkpoint': str(checkpoint), 'steps': header.get('metadata', {}).get('steps'), 'role': role,
            'passed': sum(r['passed'] for r in rows), 'total': len(rows), 'results': rows,
            'scope': 'Handwritten smoke checks, not a general coding benchmark; parent training may contain related examples.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--role', choices=['router', 'coding'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.role)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}))


if __name__ == '__main__':
    main()
