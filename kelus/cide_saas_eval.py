"""Behavior checks for generated SaaS data-layer functions, separate from loss.

The prompts are held out, but concepts overlap training. Passing is not proof
of a complete working SaaS app. Generated code is restricted to straight-line
Python and an authorized in-memory SQLite database (no filesystem SQL).
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sqlite3

from .corpus import digest
from .model import Model

CASES = [
    ('list_entries', 'Implement Python list_entries(db, actor). Return (id, title) rows from entries owned by actor, ordered by descending id. Use bound SQLite parameters.'),
    ('create_entry', 'Implement Python create_entry(db, actor, title). Trim and validate title: string, 1 to 120 characters; otherwise raise ValueError. Insert into entries(owner_id,title) with bound parameters, commit, return new id.'),
    ('edit_entry', 'Implement Python edit_entry(db, entry_id, actor, title). Validate a trimmed string title of length 1..120 or raise ValueError. Update entries only where id and owner_id both match. Bind values, commit, return a boolean indicating one changed row.'),
    ('remove_entry', 'Implement Python remove_entry(db, entry_id, actor). Delete from entries only where id and owner_id match. Use bound parameters, commit, return a boolean indicating one deleted row.'),
    ('entry_count', 'Implement Python entry_count(db, actor), returning the integer count of entries owned by actor, using a parameterized SQLite query.'),
]
ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
           ast.Name, ast.Load, ast.Store, ast.Constant, ast.Expr, ast.Call, ast.Attribute,
           ast.Tuple, ast.List, ast.Dict, ast.Subscript, ast.If, ast.IfExp, ast.Compare,
           ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
           ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Raise)
BUILTINS = {'len':len, 'isinstance':isinstance, 'str':str, 'ValueError':ValueError}
METHODS = {'execute', 'commit', 'fetchall', 'fetchone', 'strip'}


def functions(response, required):
    match = re.search(r'```(?:python|py)?\s*\n(.*?)```', response, re.S)
    code = match[1] if match else response
    tree = ast.parse(code)
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError('Expected only function definitions')
    names = [n.name for n in tree.body]
    if set(names) != set(required) or len(names) != len(required): raise ValueError('Missing or unexpected functions')
    for fn in tree.body:
        if fn.decorator_list or fn.returns or fn.args.defaults or fn.args.kw_defaults or any(arg.annotation for arg in fn.args.args):
            raise ValueError('Decorators, defaults and annotations are outside this check')
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED): raise ValueError('Unsupported operation: '+type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith('__'): raise ValueError('Reserved identifier')
        if isinstance(node, ast.Attribute) and node.attr not in METHODS | {'rowcount','lastrowid'}:
            raise ValueError('Unsupported attribute')
        if isinstance(node, ast.Call):
            valid = isinstance(node.func, ast.Name) and node.func.id in BUILTINS
            valid |= isinstance(node.func, ast.Attribute) and node.func.attr in METHODS
            if not valid: raise ValueError('Unsupported call')
        if isinstance(node, ast.Constant) and (isinstance(node.value,str) and len(node.value)>1024 or
                                              type(node.value) in (int,float) and abs(node.value)>10000):
            raise ValueError('Oversized literal')
    scope = {'__builtins__':BUILTINS}
    exec(compile(tree, '<cide-saas-eval>', 'exec'), scope)
    return scope


def database():
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE entries(id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL, title TEXT NOT NULL)')
    db.executemany('INSERT INTO entries VALUES(?,?,?)', [(1,11,'First'),(2,22,'Private'),(3,11,'Last')]); db.commit()
    def authorize(action, first, second, database, trigger):
        if action in (sqlite3.SQLITE_READ, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            return sqlite3.SQLITE_OK if first == 'entries' else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if second.lower() in ('count', 'coalesce') else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_TRANSACTION) else sqlite3.SQLITE_DENY
    db.set_authorizer(authorize)

    db.set_progress_handler(lambda:1, 10000)
    return db


def verify(scope, name):
    db = database(); f = scope[name]
    try:
        if name == 'list_entries':
            assert f(db,11) == [(3,'Last'),(1,'First')]
            assert f(db,22) == [(2,'Private')]
            assert f(db,'11 OR 1=1') == []
        elif name == 'create_entry':
            injected = "x'); DROP TABLE entries; --"
            new_id = f(db,11,'  '+injected+'  ')
            assert db.execute('SELECT owner_id,title FROM entries WHERE id=?',(new_id,)).fetchone() == (11,injected)
            for bad in (None,123,'','   ','x'*121):
                try: f(db,11,bad)
                except ValueError: pass
                else: raise AssertionError('Invalid title accepted')
            assert db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 4
        elif name == 'edit_entry':
            assert f(db,2,11,'attack') is False
            assert f(db,1,11,'  Updated  ') is True
            assert f(db,999,11,'Missing') is False
            for bad in (None,'',' '*5,'x'*121):
                try: f(db,1,11,bad)
                except ValueError: pass
                else: raise AssertionError('Invalid title accepted')
            assert db.execute('SELECT title FROM entries WHERE id=2').fetchone()[0] == 'Private'
            assert db.execute('SELECT title FROM entries WHERE id=1').fetchone()[0] == 'Updated'
        elif name == 'remove_entry':
            assert f(db,2,11) is False
            assert f(db,'1 OR 1=1',11) is False
            assert f(db,1,11) is True
            assert f(db,1,11) is False
            assert db.execute('SELECT id FROM entries ORDER BY id').fetchall() == [(2,),(3,)]
        elif name == 'entry_count':
            assert f(db,11) == 2 and f(db,22) == 1 and f(db,999) == 0
            assert f(db,'11 OR 1=1') == 0
        return True
    finally: db.close()


def evaluate(checkpoint):
    model, header, _ = Model.load(checkpoint); results = []
    for name, prompt in CASES:
        response = ''.join(model.generate([dict(role='user',content=prompt+' Return only code.')], max_tokens=480, temperature=0))
        try: passed = verify(functions(response,[name]),name); error = None
        except Exception as exc: passed = False; error = str(exc) or type(exc).__name__
        results.append(dict(name=name, prompt=prompt, response=response, passed=passed, error=error))
    return dict(checkpoint=str(checkpoint), sha256=digest(checkpoint), steps=header.get('metadata',{}).get('steps'),
                passed=sum(r['passed'] for r in results), total=len(results), results=results,
                scope='Held-out prompts for SQLite data-layer components with ownership, validation and injection tests. Related patterns occur in training. Does not test a complete website, login flow, deployment, or app security.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); report = evaluate(a.checkpoint)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__ == '__main__': main()
