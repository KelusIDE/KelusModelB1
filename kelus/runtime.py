"""Application reply policy; raw Model.generate remains available for research.

B1's small context handles one preceding exchange more reliably than a long
chat. Tool replies are not fed back as if they were learned model prose.
"""
from .tokenizer import conversation
from .local_tools import local_answer


def prepare_reply(model, messages, max_tokens=180):
    conversation(messages, generation=True, tokenizer=model.tokenizer)  # validate full role sequence
    tool = local_answer(messages[-1]['content'])
    retained = messages[-3:]
    if len(retained) == 3 and local_answer(retained[0]['content']) is not None:
        retained = retained[-1:]
    omitted = len(messages) - len(retained)
    if not tool:
        _, extra = model.prepare_prompt(retained, reserve=max_tokens)
        omitted += extra
        retained = retained[extra:]
    return retained, omitted, tool
