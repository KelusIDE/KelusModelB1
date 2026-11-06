# Goal: Kani reaches GPT-3

Set 2026-10-31. Kani climbs the GPT-3 family one size at a time, measured on the same six zero-shot tests GPT-3 was
measured on. A rung counts as reached when Kani's average over the six tests is at least that GPT-3 size's average.

## How it is measured

```
.venv/bin/python -m kelus.standard_eval --checkpoint <checkpoint.npz> --data <benchmark folder> --out training-runs/standard-eval/<name>.json
```

It runs HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande and LAMBADA with EleutherAI's lm-evaluation-harness prompts
and scoring, then prints the rung reached and the points to the next one (`kelus/standard_eval.py`, `GPT3`, `ladder`).
The harness was checked on GPT-2 small, which it scores as published: ARC-Easy 43.8, PIQA 62.9, Winogrande 51.6,
LAMBADA 32.6. GPT-3's own numbers came from OpenAI's prompts, so a rung is accurate to a point or two.

## The ladder

GPT-3 zero-shot scores, from Brown et al. 2020, Table H.1. Every size was trained on 300B tokens.

| Rung | HellaSwag | LAMBADA | PIQA | Winogrande | ARC-E | ARC-C | Average |
|---|---|---|---|---|---|---|---|
| GPT-2 small, 124M (reference, measured here) | 31.2 | 32.6 | 62.9 | 51.6 | 43.8 | 22.4 | 40.8 |
| **1. GPT-3 Small, 125M** | 33.7 | 42.7 | 64.6 | 52.0 | 43.6 | 26.6 | 43.9 |
| 2. GPT-3 Medium, 350M | 43.6 | 54.3 | 70.2 | 52.1 | 46.5 | 29.5 | 49.4 |
| 3. GPT-3 Large, 760M | 51.0 | 60.4 | 72.9 | 57.4 | 53.0 | 31.8 | 54.4 |
| 4. GPT-3 XL, 1.3B | 54.7 | 63.6 | 75.1 | 58.7 | 53.8 | 35.5 | 56.9 |
| 5. GPT-3 2.7B | 62.8 | 67.1 | 75.6 | 62.3 | 58.2 | 38.0 | 60.7 |
| 6. GPT-3 6.7B | 67.4 | 70.3 | 78.0 | 64.5 | 60.2 | 41.4 | 63.6 |
| 7. GPT-3 13B | 70.9 | 72.5 | 78.5 | 67.9 | 63.8 | 43.7 | 66.2 |
| 8. GPT-3 175B | 78.9 | 76.2 | 81.0 | 70.2 | 68.8 | 51.4 | 71.1 |

## Where Kani is

- Kani base (Big Kani v2, 205M parameters, 2.8B tokens: FineWeb-Edu, Cosmopedia and 25% code): about GPT-2 small on the
  first 200 questions of each test. The full run was stopped on 2026-10-31 because it heated the Mac; it is the first
  thing to finish (about 25 minutes per model, best run while the Mac is not in use).
- Kani Otter 4.9 (the chat model): not measured yet.

## The plan

Costs use the measured speed of the rented RTX A6000 for the 205M model (about 35,000 tokens a second, so about 8 hours
and $4.70 per billion tokens at $0.59 an hour). The Mac's GPU trains the 205M model at very roughly 1,000 tokens a
second: about 12 days per billion tokens, so pretraining belongs on rented or sponsored GPUs and chat tuning on the Mac.

| Rung | Model | Tokens | Rented A6000 (estimate) |
|---|---|---|---|
| 1. GPT-3 Small | Kani 205M as it is, trained on | ~20B in total (17B more), English-heavy: code down to ~10% | ~135 h, ~$80 |
| 2. GPT-3 Medium | a new ~400M Kani | ~60B | ~930 h, ~$550 |
| 3–4. GPT-3 Large / XL | a new ~1B Kani | 150–200B | ~8,000 h, ~$5,000 |
| 8. GPT-3 175B | — | 300B tokens on 175B parameters | about 3×10²³ operations: over $1M, and it could never run on a phone or in a browser |

Better data closes much of the size gap: SmolLM-360M, trained on 600B carefully filtered tokens, scores 51.8 on
HellaSwag, level with GPT-3 Large. The realistic end goal is GPT-3 XL quality from a Kani of 1B parameters or fewer,
with the best open data. A GPT-3-quality base model then still needs chat tuning to talk like ChatGPT.

## Rules

- Kani's weights always start from random; no pretrained weights (as since 2026-10-22).
- The six test sets are never trained on. Before each run, training shards are checked against them for 13-word
  overlaps (GPT-3's own contamination check), and overlapping documents are dropped.
- Nothing is rented or paid for without the user's go-ahead. Checkpoints leave a cloud machine before it stops.
- One GPU job at a time on the Mac.

## Next steps

1. Finish the baseline: Kani base and Otter 4.9 on all six tests (about an hour, Mac idle).
2. Rung 1, started 2026-10-31 on Kaggle: `kaggle/kani-gpt3-step1.ipynb` continues Kani Big v2 from 2.77B tokens
   toward 10B (65% FineWeb-Edu, 25% Cosmopedia v2, 10% code, all files new to it, test sets filtered out). A Kaggle
   session (T4 x2, 10 h) trains about 0.58B tokens, so 10B is about 12 sessions: about 4 weeks of the free quota,
   faster with Modal's free credit or rented/sponsored GPUs. Each session's `kani-advanced.npz` is scored on the Mac.
   If 10B falls short of GPT-3 Small, the run goes on toward ~20B. (The first attempt, 2026-10-31, stopped at the
   start: the dataset with the checkpoint was not attached.)
   **TPU:** `kaggle/kani-gpt3-tpu.ipynb` runs the same run on Kaggle's TPU v5e-8 (a separate quota of about 20 hours a
   week) with `kaggle/kani_tpu.py`: JAX, exactly the same model and AdamW (checked against PyTorch in float64), the
   same `checkpoint.pt`, and a feeder that tokenizes while it trains. Estimated 150–400K tokens a second (10–25x the
   T4s), so a session might train 5–10B tokens: the first session measures it. Sessions can switch between the GPU and
   TPU notebooks (progress `next_file` keeps them on new files), but never run at the same time.
3. Chat-tune the result as the next Otter and ship it if it beats Otter 4.9 on Kani's own chat tests.
