"""Application reply policy; raw Model.generate remains available for research.

Small checkpoints retain one exchange; larger contexts retain recent complete
turns that fit. Tool replies are not fed back as learned model prose.
"""
from .tokenizer import conversation
from .local_tools import local_answer
import re


def prepare_reply(model, messages, max_tokens=180):
    conversation(messages, generation=True, tokenizer=model.tokenizer)  # validate full role sequence
    tool = local_answer(messages[-1]['content'])
    recent = messages[-(15 if model.config.context >= 512 else 3):]
    retained = []
    for index in range(0, len(recent) - 1, 2):
        online = re.match(r'^(?:/web\s+|(?:please\s+)?(?:search (?:the )?web for|look up)\s+)', recent[index]['content'].strip(), re.I)
        if local_answer(recent[index]['content']) is None and not online:
            retained.extend(recent[index:index + 2])
    retained.append(recent[-1])
    omitted = len(messages) - len(retained)
    if not tool:
        _, extra = model.prepare_prompt(retained, reserve=max_tokens)
        omitted += extra
        retained = retained[extra:]
    return retained, omitted, tool
