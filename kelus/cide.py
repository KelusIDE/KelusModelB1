"""A single learned Kani -> Cide handoff. No keyword routing or code execution."""
from pathlib import Path
import json

CALL = 'CIDE'
ROOT = Path(__file__).resolve().parents[1]
_build = next(b for b in json.loads((ROOT / 'kani-builds.json').read_text()) if b['id'] == 'cide')
RELEASE = ROOT / 'checkpoints' / (_build['checkpoint'] + '.npz')
_model = None


def load_cide():
    global _model
    if _model is None:
        from .model import Model
        if not RELEASE.exists():
            raise FileNotFoundError('Cide is not trained/exported yet. See CIDE.md for the training command.')
        model, header, _ = Model.load(RELEASE)
        if header.get('metadata', {}).get('specialist') != 'coding':
            raise ValueError('The installed checkpoint is not a Cide coding release.')
        _model = model
    return _model


def context(model, messages, max_tokens):
    reserve = min(max_tokens, max(16, model.config.context // 3))
    latest = messages[-1]
    if latest['role'] != 'user':
        raise ValueError('Cide needs a user request.')
    if len(model.tokenizer.encode(latest['content'])) + 3 > model.config.context - reserve:
        raise ValueError('This request is too long for Cide. Send a smaller function or source excerpt.')
    retained = messages[-15:]
    while len(retained) > 1 and sum(len(model.tokenizer.encode(m['content'])) + 2 for m in retained) + 1 > model.config.context - reserve:
        retained = retained[2:]
    return retained


def answer(messages, max_tokens=4096, temperature=0, load=load_cide, cancelled=lambda: False):
    if cancelled():
        return
    expert = load()
    if cancelled():
        return
    yield {'type': 'start', 'source': 'cide', 'omitted': 0}
    for text in expert.generate(context(expert, messages, max_tokens), max_tokens=max_tokens, temperature=temperature):
        if cancelled():
            return
        yield {'type': 'token', 'text': text}


def handoff(output, messages, max_tokens, temperature, load=load_cide, cancelled=lambda: False, coding_max_tokens=4096):
    """Hold only a possible CIDE control sequence; stream ordinary text unchanged."""
    held, text = True, ''
    for piece in output:
        if cancelled():
            return
        if not held:
            yield {'type': 'token', 'text': piece}
            continue
        text += piece
        if not (CALL.startswith(text) or text.rstrip() == CALL):
            held = False
            yield {'type': 'token', 'text': text}
    if held and not cancelled():
        if text.strip() == CALL:
            yield from answer(messages, coding_max_tokens, temperature, load, cancelled)
        elif text:
            yield {'type': 'token', 'text': text}
