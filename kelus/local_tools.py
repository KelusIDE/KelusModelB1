"""Explicit local helpers, separate from B1's learned text generation.

Only anchored clock questions and bounded numeric expressions are accepted.
No eval/exec, commands, file access, network access, or model-produced tool calls.
"""
import ast
from datetime import datetime
import math
import operator
import re

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def calculate(expression):
    if len(expression) > 100 or not re.fullmatch(r'[0-9.\s+*/%()\-]+', expression):
        raise ValueError('Use numbers and +, -, *, /, //, %, or parentheses.')
    tree = ast.parse(expression.strip(), mode='eval')
    if len(list(ast.walk(tree))) > 40:
        raise ValueError('That calculation is too large.')
    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = (-1 if isinstance(node.op, ast.USub) else 1) * number(node.operand)
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            value = _OPERATORS[type(node.op)](number(node.left), number(node.right))
        else:
            raise ValueError('That operator is not supported by the local calculator.')
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError('Keep numbers and results within one trillion.')
        return value
    value = number(tree.body)
    return str(int(value)) if value == int(value) else format(value, '.12g')


def local_answer(prompt, now=None):
    """Return (text, source) or None. Source must be shown in the UI."""
    query = ' '.join(prompt.casefold().replace('’', "'").split()).rstrip('?! .')
    query = re.sub(r'^please\s+|\s+please$', '', query)
    date_queries = {"what date is today", "what is the date today", "what's the date today",
        "what is today's date", "what's today's date", "today's date", "todays date",
        "what is the date", "what's the date", "current date", "today date", "date today",
        "what day is it", "what day is today", "tell me today's date", "tell me the date",
        "what is the current date"}
    time_queries = {"what time is it", "what is the time", "what's the time", "current time",
        "tell me the time", "what is the current time"}
    if query in date_queries | time_queries:
        current = now if now is not None else datetime.now().astimezone()
        zone = current.strftime('%Z') or 'local time'
        if query in date_queries:
            return (f"Today is {current.strftime('%A, %B')} {current.day}, {current.year} ({zone}, this computer's clock).", 'local_clock')
        return (f"It is {current.strftime('%H:%M:%S')} ({zone}, this computer's clock).", 'local_clock')
    expression = re.sub(r'^(?:what is|what\'s|calculate|compute|solve|evaluate)\s+', '', query)
    expression = expression.replace('×', '*').replace('÷', '/').replace('−', '-')
    if expression.endswith('='):
        expression = expression[:-1].rstrip()
    # Require an operator; bare numbers and prose remain model input.
    if not re.fullmatch(r'[0-9.\s+*/%()\-]+', expression) or not re.search(r'[+*/%\-]', expression):
        return None
    try:
        return (f'{expression} = {calculate(expression)}', 'local_calculator')
    except ZeroDivisionError:
        return ('Division by zero is undefined.', 'local_calculator')
    except (ValueError, SyntaxError, RecursionError, OverflowError) as error:
        detail = str(error) if isinstance(error, ValueError) else 'Check the numbers and parentheses.'
        return (f'I could not calculate that. {detail}', 'local_calculator')
