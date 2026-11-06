"""Grounded answers: Kani reads search results and answers from them with a citation.

The same prompt layout is used for training (data/make_reading.py) and at run time, when
a /web search's passages are handed to the model. `select` keeps the sentences of the
results that share the most words with the question, so the passages fit the context.
"""
import math
import re

NO_ANSWER = "The search results don't say. Try a more specific search."
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(])')
WORD = re.compile(r"[a-z0-9]+")
COMMON = set('a an the of in on at to for and or is are was were be by with as from that this it its what who when where which how why '
             'do does did tell me about you know explain please kani hey'.split())


def prompt(results, question):
    """results: [(title, passage)] in citation order."""
    lines = ['Search results:'] + [f'[{i}] {title}: {passage}' for i, (title, passage) in enumerate(results, 1)]
    return '\n'.join(lines) + f'\n\nQuestion: {question}'


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in COMMON]


def select(results, question, budget=90):
    """Trim each (title, text) result to its sentences most related to the question, at most
    `budget` words, kept in their original order. The first sentence (usually a definition) stays."""
    wanted = set(words(question))
    trimmed = []
    for title, text in results:

        sentences = [' '.join(s.split()[:budget]) for s in SENTENCE.split(' '.join(text.split())) if s.strip()]
        if not sentences:
            continue
        score = lambda i: (i == 0) * 0.5 + sum(1 for w in set(words(sentences[i])) if w in wanted) / math.sqrt(1 + len(sentences[i].split()) / 20)
        keep, used = set(), 0
        for i in sorted(range(len(sentences)), key=lambda i: -score(i)):
            size = len(sentences[i].split())
            if used + size > budget and keep:
                continue
            keep.add(i)
            used += size
        trimmed.append((title, ' '.join(sentences[i] for i in sorted(keep))))
    return trimmed
