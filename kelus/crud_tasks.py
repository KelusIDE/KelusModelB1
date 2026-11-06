"""Small varied SQLite contracts and behavior checks for Cide's next milestone.

No framework or external model. The checker executes only the restricted
Python subset already used by cide_saas_eval, against authorized in-memory SQL.
"""
import json
import re
import sqlite3

from .cide_saas_eval import functions

OPERATIONS = ('list', 'get', 'count', 'create', 'update', 'delete')


def signature(s):
    args = [s['db']]
    if s['operation'] in ('get','update','delete'): args.append(s['item'])
    args.append(s['user'])
    if s['operation'] in ('create','update'): args.append(s['value'])
    return f"{s['name']}({', '.join(args)})"


def prompt(s, style=0):
    op=s['operation']; d,u,i,v=s['db'],s['user'],s['item'],s['value']
    t,o,c=s['table'],s['owner'],s['column']; direction=s['direction']
    descriptions = {
        'list': f'Return all (id, {c}) tuples owned by {u}, ordered by id {direction}.',
        'get': f'Return the (id, {c}) tuple whose id equals {i} and {o} equals {u}, or None if no such row exists.',
        'count': f'Return the integer number of rows owned by {u}.',
        'create': f'Insert a row with {o}={u} and {c}=the trimmed {v}. Commit and return its integer lastrowid.',
        'update': f'Change {c} to the trimmed {v} only where id={i} AND {o}={u}. Commit and return True if exactly one row changed, else False.',
        'delete': f'Delete only where id={i} AND {o}={u}. Commit and return True if exactly one row was deleted, else False.',
    }
    validation = (f' {v} must be a string whose stripped length is 1..{s["limit"]}; otherwise raise ValueError.'
                  if op in ('create','update') else '')
    contract = f'SQLite table {t}(id INTEGER PRIMARY KEY, {o} INTEGER, {c} TEXT). {d} is its connection. {u} is the authenticated owner. '
    contract += descriptions[op]+validation+' Bind user values with ? parameters. Return only Python code.'
    if style==0: return f'Write Python function {signature(s)}. '+contract
    if style==1: return f'Implement {signature(s)} in Python for this database task. '+contract
    if style==2: return f'Add a data-access function named {signature(s)} to my app. '+contract
    return f'My app needs {signature(s)}. Follow this contract exactly:\n'+contract


def reference(s):
    op=s['operation']; d,u,i,v=s['db'],s['user'],s['item'],s['value']
    t,o,c=s['table'],s['owner'],s['column']
    lines=[f'def {signature(s)}:']
    if op in ('create','update'):
        lines += [f'    if not isinstance({v}, str) or not 1 <= len({v}.strip()) <= {s["limit"]}:',
                  '        raise ValueError("Invalid text")']
    if op=='list': lines += [f'    return {d}.execute("SELECT id, {c} FROM {t} WHERE {o} = ? ORDER BY id {s["direction"]}", ({u},)).fetchall()']
    elif op=='get': lines += [f'    return {d}.execute("SELECT id, {c} FROM {t} WHERE id = ? AND {o} = ?", ({i}, {u})).fetchone()']
    elif op=='count': lines += [f'    return {d}.execute("SELECT COUNT(*) FROM {t} WHERE {o} = ?", ({u},)).fetchone()[0]']
    else:
        if op=='create': query=f'INSERT INTO {t} ({o}, {c}) VALUES (?, ?)'; params=f'{u}, {v}.strip()'
        elif op=='update': query=f'UPDATE {t} SET {c} = ? WHERE id = ? AND {o} = ?'; params=f'{v}.strip(), {i}, {u}'
        else: query=f'DELETE FROM {t} WHERE id = ? AND {o} = ?'; params=f'{i}, {u}'
        lines += [f'    cursor = {d}.execute("{query}", ({params}))', f'    {d}.commit()',
                  '    return cursor.lastrowid' if op=='create' else '    return cursor.rowcount == 1']
    return '\n'.join(lines)+'\n'


def database(s):
    for key in ('table','owner','column'):
        if not re.fullmatch('[a-z][a-z0-9_]*',s[key]): raise ValueError('Invalid fixture identifier')
    t,o,c=s['table'],s['owner'],s['column']; db=sqlite3.connect(':memory:')
    db.execute(f'CREATE TABLE {t}(id INTEGER PRIMARY KEY, {o} INTEGER NOT NULL, {c} TEXT NOT NULL)')
    db.executemany(f'INSERT INTO {t} VALUES(?,?,?)',[(1,11,'First'),(2,22,'Private'),(3,11,'Last')]);db.commit()
    def authorize(action,first,second,database,trigger):
        if action in (sqlite3.SQLITE_READ,sqlite3.SQLITE_INSERT,sqlite3.SQLITE_UPDATE,sqlite3.SQLITE_DELETE):
            return sqlite3.SQLITE_OK if first==t else sqlite3.SQLITE_DENY
        if action==sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if second and second.lower()=='count' else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_TRANSACTION) else sqlite3.SQLITE_DENY
    db.set_authorizer(authorize);db.set_progress_handler(lambda:1,10000)
    return db


def verify(response,s):
    scope=functions(response,[s['name']]);f=scope[s['name']];db=database(s)
    t,o,c=s['table'],s['owner'],s['column'];op=s['operation']
    try:
        if op=='list':
            expected=[(1,'First'),(3,'Last')]
            if s['direction']=='DESC': expected.reverse()
            assert f(db,11)==expected and f(db,22)==[(2,'Private')]
            assert f(db,'11 OR 1=1')==[] and f(db,999)==[]
        elif op=='get':
            assert f(db,1,11)==(1,'First') and f(db,2,11) is None
            assert f(db,999,11) is None and f(db,'1 OR 1=1',11) is None
        elif op=='count':
            assert f(db,11)==2 and f(db,22)==1 and f(db,999)==0 and f(db,'11 OR 1=1')==0
        elif op=='delete':
            assert f(db,2,11) is False and f(db,'1 OR 1=1',11) is False
            assert f(db,1,11) is True and f(db,1,11) is False
            assert db.execute(f'SELECT id FROM {t} ORDER BY id').fetchall()==[(2,),(3,)]
        else:
            args=[db,11] if op=='create' else [db,1,11]
            for bad in (None,123,'','  ','x'*(s['limit']+1)):
                try: f(*args,bad)
                except ValueError: pass
                else: raise AssertionError('Invalid input accepted')
            text="x'); --"
            result=f(*args,'  '+text+'  ')
            if op=='create':
                assert type(result)==int
                assert db.execute(f'SELECT {o},{c} FROM {t} WHERE id=?',(result,)).fetchone()==(11,text)
            else:
                assert result is True and f(db,2,11,'attack') is False
                assert f(db,999,11,'missing') is False
                assert f(db,'1 OR 1=1',11,'injection') is False
                assert db.execute(f'SELECT {c} FROM {t} WHERE id=1').fetchone()==(text,)
            assert db.execute(f'SELECT {c} FROM {t} WHERE id=2').fetchone()==('Private',)
        if op in ('create','update','delete'): assert not db.in_transaction, 'Missing commit'
        return True
    finally: db.close()


def evaluate(checkpoint,cases_path):
    from .model import Model
    from .corpus import digest
    model,header,_=Model.load(checkpoint); results=[]
    for row in json.loads(cases_path.read_text()):
        s=row['spec'];response=''.join(model.generate([dict(role='user',content=row['prompt'])],max_tokens=400,temperature=0))
        try: passed=verify(response,s);error=None
        except Exception as exc: passed=False;error=str(exc) or type(exc).__name__
        results.append(dict(operation=s['operation'],prompt=row['prompt'],response=response,passed=passed,error=error))
    return dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),cases_sha256=digest(cases_path),
                steps=header['metadata'].get('steps'),passed=sum(r['passed'] for r in results),total=len(results),results=results,
                scope='Held-out schema/identifier combinations from the same six generated CRUD contract families. Executed behavior checks, not a full SaaS or independent broad coding benchmark.')
