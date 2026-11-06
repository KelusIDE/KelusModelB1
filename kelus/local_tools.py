"""Explicit local helpers, separate from B1's learned text generation.

Only anchored clock questions and bounded numeric expressions are accepted, plus a
fixed crisis-support reply for suicide and self-harm wording.
No eval/exec, commands, file access, network access, or model-produced tool calls.
"""
import ast
from datetime import datetime
import math
import operator
import re
from .arithmetic import arithmetic_expression

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}



# Fixed replies for people who may be in danger. site/support.js holds the same patterns and replies; tests/test_web.py
# checks that the two agree. These never go to the model: Kani is too small to be trusted here.
CRISIS_REPLY = "I'm really sorry you're in this much pain. You deserve support right now, and I'm only a small AI, so please reach out to a person. In the US, call or text 988 for the Suicide and Crisis Lifeline. Elsewhere, call your local emergency number or a crisis line, or tell someone you trust right away."
DANGER_REPLY = "Please get help right now. Call your local emergency number (911 in the US), or ask someone near you to call. If you or someone else took something, US Poison Help is 1-800-222-1222, day and night. Please don't wait to see how you feel, and don't stay alone. I'm only a small AI, so a person needs to help with this."
OTHER_PERSON_REPLY = "That sounds really scary, and it's good that you're taking it seriously. Stay with them if you can, and don't keep it a secret, even if they ask you to. In the US, you can call or text 988 together, or on your own to ask how to help. If they are in immediate danger, call your local emergency number now, and tell an adult or someone you trust."
LOSS_REPLY = "I'm so sorry for your loss. Losing someone this way is incredibly painful, and whatever you're feeling is okay. Please don't carry it alone, and talk to someone you trust. In the US, 988 is there day and night for anyone who is struggling, including after losing someone to suicide."
CHECK_IN_REPLY = "That sounds really heavy, and I'm glad you told me. You don't have to carry it alone. If you're thinking about ending your life or hurting yourself, please reach out to a person now: in the US, call or text 988. Elsewhere, call your local emergency number or a crisis line. If it's something else, what's been going on?"
STILL_HERE_REPLY = "I'm still here, and I'm glad you're talking. I'm only a small AI and can't keep you safe the way a person can, so please call or text 988 in the US, or your local crisis line or emergency number elsewhere, or tell someone near you what you told me. Are you safe right now?"
HARMLESS = re.compile('\\bsuicide (?:squad|doors?|burn|king|forest)\\b|\\bkill(?:ing)? my ?self laughing\\b|\\b(?:hurt(?:s|ing)?|cut(?:s|ting)?|harm(?:s|ing)?) (?:my ?self|himself|herself|themsel(?:f|ves))(?: \\w+)? (?:while|when|playing|shaving|cooking|by accident|accidentally|at (?:the gym|work|practice)|on (?:a|an|the|my))\\b')
DANGER = re.compile("\\b(?:took|taken|take|swallowed|ate)\\b(?: \\w+)? (?:too many|too much|a bunch of|a lot of|a whole bottle of|the whole bottle of|handfuls? of|all (?:of )?(?:my|the))\\b.{0,30}\\b(?:pills?|meds|medication|medicine|tablets?|painkillers|tylenol)\\b|\\boverdos(?:ed|ing)\\b|^(?:no,? )?i'?m not safe\\b")
SELF = re.compile("\\b(?:kill(?:ing)?|hurt(?:ing)?|harm(?:ing)?|cut(?:ting)?|unalive|hang(?:ing)?|shoot(?:ing)?|drown(?:ing)?|poison(?:ing)?|stab(?:bing)?) ?my ?self\\b|\\bslit(?:ting)? my wrists?\\b|\\b(?:want|wanna|wanting|ready)(?: to)? (?:just )?die\\b|\\bwish i (?:could|would) (?:just )?die\\b|\\bend(?:ing)? (?:my (?:own )?life|it all|my ?self)\\b|\\b(?:want to|wanna|going to|gonna) end it(?: tonight| today| now)?$|\\btake my (?:own )?life\\b|\\b(?:don'?t|do not) want to (?:live|be alive|exist|be here|wake up)\\b|\\bstop existing\\b|\\bgo on living\\b|(?:\\bnot|n'?t) worth living\\b|\\bwish i (?:was|were) dead\\b|\\bwish i (?:wasn'?t|weren'?t) alive\\b|\\bwish i(?:'d| had| was| were)? never (?:been )?born\\b|\\bbetter off (?:dead|without me)\\b|\\b(?:wish|want|hope)\\b.{0,40}\\b(?:never|not|don'?t|wouldn'?t) wake up(?: again| tomorrow| anymore)?$|\\b(?:no|see a|see any|find a|find any) (?:reason|point) (?:to|in) (?:live|living|keep going|go on|going on)\\b|\\bnothing to live for\\b|\\b(?:tired|sick) of (?:living|being alive)\\b|\\bwould care if i died\\b|\\b(?:nobody|no one) would (?:care|notice|miss me)\\b.{0,12}\\bif i (?:died|was gone|were gone|disappeared)\\b|\\bdisappear forever\\b|(?:^|\\b(?:i|i'?m|gonna|wanna|to|will|should|just|might)\\s)kms\\b|\\b(?:rather|better) be dead\\b|\\b(?:had|have) enough of (?:living|life|being alive)\\b|\\bcan'?t (?:keep|go on|continue) living\\b|\\bthink(?:ing)? (?:about|of) (?:not being (?:alive|here)|not existing|being dead|dying)\\b|\\b(?:easier|better)\\b.{0,30}\\bif i (?:wasn'?t|weren'?t|was not|were not) (?:here|around|alive)\\b|\\bif i (?:was|were) (?:gone|dead)\\b.{0,30}\\b(?:easier|better)\\b|\\b(?:pills?|rope|gun)\\b.{0,30}\\bend (?:it|things|my)\\b|\\b(?:(?:want|wanna|going|gonna|might|will)(?: to)?) jump off (?:a|the|this) (?:bridge|building|roof|cliff|balcony)\\b|\\b(?:i'?m|i am|feel(?:ing)?|been|have|having|getting)\\b.{0,24}\\bsuicid")
LOSS = re.compile('\\bmy\\b.{0,40}\\b(?:killed (?:him|her|them)sel(?:f|ves)|committed suicide|died by suicide|took (?:his|her|their) (?:own )?life)\\b|\\blost\\b.{0,30}\\bto suicide\\b')
OTHER_WHO = re.compile('\\b(?:friend|bestie|bff|sister|brother|sibling|mom|mum|mother|dad|father|parent|son|daughter|kid|child|boyfriend|bf|girlfriend|gf|wife|husband|partner|cousin|aunt|uncle|classmate|roommate|coworker|colleague|student|neighbor|teammate|grandma|grandpa|someone|somebody|he|she)\\b')
OTHER_ACT = re.compile('\\b(?:kill(?:s|ing)?|hurt(?:s|ing)?|harm(?:s|ing)?|cut(?:s|ting)?|unalive(?:s)?) (?:him|her|them)sel(?:f|ves)\\b|\\bend(?:s|ing)? (?:his|her|their) (?:own )?life\\b|\\bself[- ]?harm|\\bsuicid|\\boverdos|\\bwants? to die\\b|\\bwanna die\\b')
GENERAL = re.compile('\\bsuicid|\\bself[- ]?harm|\\boverdos')
CHECK_IN = re.compile("\\b(?:what'?s|what is) the point(?: of (?:living|life|anything|it all|going on|being alive|trying))?(?: any ?more)?$|\\b(?:don'?t|do not|can'?t) see the point(?: of (?:anything|living|life|going on|trying))?(?: any ?more)?$|\\bi can'?t (?:do this|do it|take (?:it|this)|go on|keep going|cope|handle (?:it|this|life))(?: any ?more| any longer)$|\\bi can'?t go on$|\\bi hate (?:my life|myself|being alive)$|\\bmy life is (?:over|pointless|worthless|meaningless|ruined)$|\\blife is (?:pointless|meaningless)$|\\bi'?m (?:just )?(?:a burden|worthless)(?: to (?:everyone|everybody|my family|my friends|anyone))?$|\\bi (?:should|want to|wanna|wish i could) (?:just )?disappear$|\\bwhy (?:should|would|do) i (?:even )?(?:keep going|go on|bother living|stay alive|live)\\b")
NEW_SUBJECT = re.compile('^(?:(?:hi|hey|ok|okay|thanks|thank you|anyway)[,. ]+)?(?:what|who|when|where|which|how|why|can you|could you|write|make|build|create|draw|tell me|explain|define|translate|search|find|show|give|calculate|solve|code)\\b')


def normalized(text):
    return ' '.join(text.casefold().replace('\u2019', "'").split()).rstrip('?! .')


def support_reply(query):
    """The fixed reply for a message (already lower case, as `normalized` gives), or None."""
    text = ' '.join(HARMLESS.sub(' ', query).split())
    if DANGER.search(text):
        return DANGER_REPLY
    if SELF.search(text):
        return CRISIS_REPLY
    if LOSS.search(text):
        return LOSS_REPLY
    if OTHER_WHO.search(text) and OTHER_ACT.search(text):
        return OTHER_PERSON_REPLY
    if GENERAL.search(text):
        return CRISIS_REPLY
    return CHECK_IN_REPLY if CHECK_IN.search(text) else None


def support_follow_up(messages):
    """The next message after a fixed reply to someone in danger stays with a person's help, unless it plainly moves on.
    Only one turn: after that the model answers, or "yes I'm safe" would get the same text forever."""
    if len(messages) < 2 or messages[-2]['role'] != 'assistant' or messages[-2]['content'] not in (CRISIS_REPLY, DANGER_REPLY):
        return None
    query = normalized(messages[-1]['content'])
    moved_on = NEW_SUBJECT.search(query) and len(query.split()) >= 3 and not re.search(r"\b(?:i|i'?m|me|my|myself)\b", query)
    return None if moved_on else STILL_HERE_REPLY


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



def word_math(query):
    """A bounded grammar for percentage word problems, with explicit arithmetic."""
    number = r'(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?'
    text = re.sub(r"^(?:what is|what's|calculate|compute|how much is)\s+", '', query)
    text = re.sub(r'^(?:a|an)\s+', '', text)
    text = re.sub(r'\s+is$', '', text)
    match = re.fullmatch(rf'({number})\s*(?:%|percent)\s*(?:(tip|tax|discount)\s+)?(?:of|on|off)\s+(?:a\s+)?([$€£₩]?)({number})(?:\s+bill)?', text)
    if not match:
        return None
    rate, kind, currency, amount = match.groups()
    if ' off ' in text: kind = 'discount'
    rate, amount = rate.replace(',', ''), amount.replace(',', '')
    try:
        if kind == 'discount' and float(rate) > 100:
            return 'A discount cannot exceed 100% of the original price.'
        portion = calculate(f'{amount} * {rate} / 100')
        equation = f'{amount} × {rate} / 100 = {portion}'
        if kind in ('tip', 'tax', 'discount'):
            total = calculate(f'{amount} {"-" if kind == "discount" else "+"} {portion}')
            return f'{kind.capitalize()}: {currency}{portion}. Total: {currency}{total}.\n{equation}'
        return f'{rate}% of {currency}{amount} is {currency}{portion}.\n{equation}'
    except (ValueError, OverflowError):
        return 'I could not calculate that. Keep numbers and results within one trillion.'


def local_answer(prompt, now=None):
    """Return (text, source) or None. Source must be shown in the UI."""
    query = ' '.join(prompt.casefold().replace('’', "'").split()).rstrip('?! .')
    query = re.sub(r'^please\s+|\s+please$', '', query)
    support = support_reply(query)
    if support:
        return support, 'crisis_support'
    word_answer = word_math(query)
    if word_answer is not None:
        return word_answer, 'local_calculator'
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
    expression = arithmetic_expression(query)
    if expression is None:
        return None
    try:
        return (f'{expression} = {calculate(expression)}', 'local_calculator')
    except ZeroDivisionError:
        return ('Division by zero is undefined.', 'local_calculator')
    except (ValueError, SyntaxError, RecursionError, OverflowError) as error:
        detail = str(error) if isinstance(error, ValueError) else 'Check the numbers and parentheses.'
        return (f'I could not calculate that. {detail}', 'local_calculator')
