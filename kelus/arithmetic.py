"""Anchored calculator phrasing, separate from learned model answers.

Numeric expressions retain the existing calculator grammar. Word expressions
support a single binary operation and English integer operands up to 999.
"""
import re

SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
TENS['fourty'] = 40


def english_number(text):
    text = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', text.strip())
    if re.fullmatch(r'-?\d+(?:\.\d+)?', text):
        return text
    words = text.split()
    sign = -1 if words and words[0] == 'negative' else 1
    if sign < 0:
        words = words[1:]
    hundreds = 0
    if len(words) >= 2 and words[1] == 'hundred' and 1 <= SMALL.get(words[0], -1) <= 9:
        hundreds = SMALL[words[0]] * 100
        words = words[2:]
        if not words:
            return str(sign * hundreds)
        if words[0] == 'and':
            words = words[1:]
    if len(words) == 1 and words[0] in SMALL:
        rest = SMALL[words[0]]
    elif len(words) == 1 and words[0] in TENS:
        rest = TENS[words[0]]
    elif len(words) == 2 and words[0] in TENS and 1 <= SMALL.get(words[1], -1) <= 9:
        rest = TENS[words[0]] + SMALL[words[1]]
    else:
        return None
    return str(sign * (hundreds + rest))


def arithmetic_expression(query):
    if len(query) > 300:
        return None
    text = re.sub(r'^(?:what is|what\'s|whats|how much is|(?:can|could) you (?:please )?calculate|calculate|compute|solve|evaluate)\s+', '', query)
    text = re.sub(r'(?:[,!? .]+|\s)+(?:please|heh|haha|lol)$', '', text).rstrip('?! .')
    text = text.replace('×', '*').replace('÷', '/').replace('−', '-')
    text = re.sub(r'\s*=\s*$', '', text)
    if re.fullmatch(r'[0-9.\s+*/%()\-]+', text) and re.search(r'[+*/%\-]', text):
        return text
    match = re.fullmatch(r'(.+?)\s+(plus|minus|times|multiplied by|divided by|over)\s+(.+)', text)
    if not match:
        return None
    a, operator, b = match.groups()
    a, b = english_number(a), english_number(b)
    if a is None or b is None:
        return None
    operation = {'plus': '+', 'minus': '-', 'times': '*', 'multiplied by': '*', 'divided by': '/', 'over': '/'}[operator]
    return f'{a} {operation} {b}'
