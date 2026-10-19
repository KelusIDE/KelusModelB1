# Kelus Model B1 v3

**Status:** Experimental narrow model. Reliable on its practiced chat replies, coding concepts, and short code snippets, including new wording for them. It is not a general assistant: unfamiliar slang, open-ended chat, and anything outside the practice topics still fail.

| Property | Selected checkpoint |
|---|---|
| File | `checkpoints/b1-v3.best.npz` (browser copy: `site/model/`, https://kelus-b1.web.app) |
| Weights | Random initialization, seed 42; trained locally |
| Pretrained weights / external model API | None |
| Parameters | 431,424 |
| Architecture | 3 pre-LN causal transformer blocks; width 96; 4 heads; GELU MLP; rotary positions; token embeddings tied to the output layer; dropout 0.1 during training |
| Tokenizer | Own byte-pair encoding v2, 1,024 IDs including 5 special IDs; a code fence and its language tag form one piece; lossless UTF-8 byte fallback |
| Context | 256 BPE tokens; the app keeps at most one earlier exchange |
| Corpus | Original synthetic curriculum v3 (`data/make_v3.py`): 37,075 train / 1,160 validation conversations, 1,890,217 BPE tokens |
| Selected step | 10,000 of 12,000 (lowest full-validation loss, 0.00495 nats per token) |
| Optimization | Our AdamW; batch 16; LR 0.0015 with warmup/cosine; weight decay 0.01; gradient clipping 1 |
| Compute | About 24 minutes on an Apple M4 CPU, NumPy only |

## What it can do

- Chat replies for 20 intents: greetings, identity, thanks, goodbye, clarification, corrections, help, abilities, training, limits, debugging, learning, a joke, Minecraft, and out-of-scope requests.
- Explanations of 20 programming concepts, and a code example as a follow-up ("show me an example").
- 24 Python snippets, including fibonacci, factorial, primes, min/max, average, palindromes, temperature conversion, and fizzbuzz, plus 4 JavaScript snippets.
- An honest decline for specific programs it has not learned ("make me a snake game"), instead of asking again what to build.

Arithmetic and date/time questions are answered by labeled local helpers, not by the model.

## Results

All checks use temperature 0. Reports with every raw reply are in [reports/benchmark/](reports/benchmark/).

| Check | b1-conversation (original, 554K) | b1-current (repair, 554K) | **B1 v3 (431K)** |
|---|---:|---:|---:|
| Held-out probes, all | 72/199 (36%) | 81/199 (41%) | **158/199 (79%)** |
| · chat wording | 58/115 | 62/115 | 75/115 |
| · concept wording | 8/43 | 12/43 | 42/43 |
| · code requests | 4/31 | 4/31 | 31/31 |
| · two-turn | 2/10 | 3/10 | 10/10 |
| Context regression (not held out) | 146/184 | 164/184 | **179/184** |
| Original suite: bug regressions | 5/5 | 5/5 | 5/5 |
| Original suite: reported screenshot sequence | 5/5 | 5/5 | 5/5 |
| Original suite: new wordings | 9/15 | 13/15 | 12/15 |
| Repair suite: app sequence / probes | 5/8, 3/6 | 8/8, 5/6 | 8/8, 5/6 |

**How to read this.** The 199 probes in `kelus/probes.py` are hand-written new phrasings of *practiced topics*. Chat probes must return the exact canonical reply, and Python function probes are checked by a bounded AST interpreter. Training data is decontaminated against every probe and the older suites: exact matches, prompts containing a probe, and prompts asking for the same content words are removed (1,288 prompts). The probes and curriculum v3 were written by the same author in the same session, so the v3 probe score is optimistic. The original and repair suites were written earlier by someone else, and on those v3 is about level with b1-current. It gains code wording and loses some greeting wordings. The context check reuses trained messages (`hi!`, `thanks`, `bye`…) after 23 kinds of earlier exchange; it measures whether history hijacks the answer, which caused the originally reported bug.

## New-program readiness

The user requested new programs and websites, beyond practiced snippets. `python3 -m kelus.code_readiness` tests four small new specifications (three inputs, cubing, tax calculation, and a requested function name) with the existing bounded AST interpreter. v3 passes **0/4**. For a single-file counter website it produces a JavaScript square function instead; all five HTML/CSS/JavaScript structure checks fail. Browser behavior is not tested. Raw results: [reports/code-readiness-v3.json](reports/code-readiness-v3.json). These are disclosed development probes, not a standardized benchmark.

The 37,075 training conversations contain 53,561 supervised assistant turns but only **84 distinct supervised answer texts**. The number of prompt variations and training steps therefore overstates the diversity of tasks learned. More steps on this unchanged curriculum do not constitute broad code pretraining.

## Known failures

At greedy decoding B1 v3 still:

- Misreads slang and wording outside its vocabulary: "howdy", "cya", "wut", "peace out", "cheers!", and "i'm confused" get unrelated canned replies.
- Answers some questions about itself with the wrong canned reply ("who am i talking to?" → goodbye; "are you connected to openai?" → square function).
- Treats some debugging reports as requests for help building ("i got an error").
- Occasionally lets a previous "I generate text only…" reply capture the next greeting.
- Gives a canned reply, never an open-ended answer: it cannot write programs outside its snippet list, explain topics outside its 20 concepts, or reason about your code.

It never runs code, browses, edits files, or remembers chats.

## Changes from the previous model

- **Tokenizer fix.** BPE v1 split ```` ``` ```` into three single-backtick tokens, so an opening fence looked like a closing one, and unfamiliar code requests often ended right after ```` ``` ````. v2 keeps ```` ```python ```` as one piece.
- **Curriculum v3.** More phrasings per intent, plural and alias concept forms, compositional code requests, typo/casing noise, one-earlier-exchange histories, corrections after wrong replies (context only), new snippets, and declines for unknown programs.
- **Architecture.** Rotary positions, tied embeddings, and dropout. A comparison on the same data favored this over the original layout (learned positions, separate output matrix) on held-out probes and validation loss, with 22% fewer parameters. A wider 920K-parameter variant trained more slowly and scored worse on the context check after the same wall-clock time.
- **Inference.** Key/value cache: about 2,400 tokens/s versus 60 before, identical output.
- **Evaluation.** The AST checker now accepts plain f-strings, `/`, `max`, and `min`. Before, it marked the correct `greet` function as wrong.

## Earlier checkpoints

- `b1-demo.npz`: 90,048-parameter byte model on 94 examples; memorized and produced gibberish on variations.
- `b1-conversation.best.npz`: 554,304 parameters, curriculum v1 (5,875 conversations), step 4,000. It was the default before v3.
- `b1-refined.*`: final-reply-only refinement; did not improve validation.
- `b1-repair*.npz`, `b1-current.npz`: continued training on repair curricula (`data/make_repair*.py`) for corrections and short chat; `b1-current` is the best of these.

Earlier reports (`reports/b1-before.json`, `reports/b1-after.json`, `reports/b1-*-app.json`) are kept unchanged.

## Intended use

Study and extend a complete local pipeline: tokenizer → transformer → loss → custom gradients → optimizer → checkpoint → chat, in Python and in the browser. Use the practiced topics and inspect the reported failures. Generated code is shown as text and never executed; review and test it before use.
