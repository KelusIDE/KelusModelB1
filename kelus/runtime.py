"""Application reply policy; raw Model.generate remains available for research.

Small checkpoints retain one exchange; larger contexts retain recent complete
turns that fit. Tool replies are not fed back as learned model prose.
"""
from .tokenizer import conversation
from .local_tools import local_answer, support_follow_up
import re


def prepare_reply(model, messages, max_tokens=180):
    conversation(messages, generation=True, tokenizer=model.tokenizer)
    follow_up = support_follow_up(messages)
    tool = local_answer(messages[-1]['content']) or ((follow_up, 'crisis_support') if follow_up else None)
    recent = messages[-(15 if model.config.context >= 512 else 3):]
    retained = []
    for index in range(0, len(recent) - 1, 2):
        online = re.match(r'^(?:/web\s+|(?:(?:please|(?:can|could|would|will) you(?: please)?)\s+)?(?:search(?:\s+(?:the\s+)?(?:web|internet|online|google|duckduckgo))?(?:\s+(?:for|up))?|look\s+up|google)\s+)', recent[index]['content'].strip(), re.I)
        if local_answer(recent[index]['content']) is None and not online:
            retained.extend(recent[index:index + 2])
    retained.append(recent[-1])
    omitted = len(messages) - len(retained)
    if not tool:
        _, extra = model.prepare_prompt(retained, reserve=max_tokens)
        omitted += extra
        retained = retained[extra:]
    return retained, omitted, tool
