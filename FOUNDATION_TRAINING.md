# B1 foundation training project

The aim is to learn language and code from varied documents before teaching B1 to follow coding requests. This is a new, randomly initialized model using our existing NumPy transformer, automatic differentiation, tokenizer, and AdamW. It does not load a pretrained model or use an AI framework.

This is an experiment, not a claim that a small Mac-trained model will become a general coding assistant. The 10,000-step v3 checkpoint, browser model, and IDE bundle remain the released chat model until a candidate demonstrates better coding behavior.

## Current experiment

| Property | Value |
|---|---|
| Parameters | 8,691,840 (v3: 431,424) |
| Architecture | Width 320, 6 transformer blocks, 8 heads, rotary positions, tied embeddings |
| Context | 1,024 tokens (v3: 256) |
| Vocabulary | 4,096 local byte-pair tokens, learned only from training documents |
| Objective | Predict every next token in source code and explanatory documents |
| Training documents / tokens | 13,402 / 26,890,165 |
| Validation documents / tokens | 404 / 885,273 |
| Batch | 1 sequence per microbatch, 4 microbatches accumulated per optimizer step |
| First phase | 4,000 total optimizer steps, 16,384,000 token presentations |
| Checkpoints | `training-runs/b1-foundation/latest.npz` and `best.npz` |
| Checkpoint interval | Every 50 steps, and on a clean stop |
| Validation | Fixed sample of 8,192 held-out tokens, seed 2401; sampled loss, not a coding score |

The first phase samples about 0.61 times the number of training tokens, with replacement: it does **not** guarantee visiting every document. More pretraining and instruction training will be needed. That decision should use loss curves and new-task evaluations, not a reassuring step count.

The pilot completed 11 steps from random weights, with sampled validation loss decreasing from 8.3626 to 7.9310. Including startup, evaluation, and checkpoint writes, it took about 105 seconds. The initial full-phase estimate is approximately 9–11 hours on this M4, subject to other activity, throttling, sleep, and power state. This early loss drop only proves learning is happening; it does not prove useful code generation.

## Controls

- **Foundation Progress.command**: view progress. Closing this viewer does not stop training.
- **Stop Foundation.command**: request a stop after a complete optimizer step; the trainer saves before exiting.
- **Train Foundation.command**: start or resume this foundation run. A lock prevents two trainers from writing the same run.

The old **Train B1.command** still trains v3. It is deliberately unchanged.

Equivalent terminal commands, from this directory:

```sh
python3 -m kelus.foundation_status --watch
python3 -m kelus.foundation_status --stop
nice -n 10 python3 -m kelus.pretrain --resume
```

**Train Foundation.command** now prevents idle system sleep for the lifetime of its training process; the display can still sleep. The already-running job also has temporary idle-sleep prevention attached to its PID. This does not prevent manual sleep, shutdown or a lid close. Resume after a restart with **Train Foundation.command**. A crash/power loss can lose steps since the last checkpoint; atomic writes retain the previous complete checkpoint. Keep at least several hundred MB free for checkpoint replacement.

`--steps` is the **total** step budget in this new pretraining command, unlike `kelus.train` where it means additional steps. Resume requires the original recipe and corpus fingerprint. The resume path restores Adam moments, RNG state, and the absolute-step learning-rate schedule. Architecture is loaded from the checkpoint. For a fresh experiment, choose another `--run` directory.

## Data and provenance

Sources were downloaded over verified TLS from GitHub archives, pinned to these revisions:

- [CPython](https://github.com/python/cpython/tree/036d309cf542986a39eeba5ca805b2ec7619731d): selected `Lib/*.py` and `Doc/*.rst` files, under the upstream Python license stack and retained file notices.
- [MDN content](https://github.com/mdn/content/tree/a1aa79c09717761fc72c20bac94bb9c430ed487a): English web and learning pages. Prose is CC-BY-SA-2.5; code terms are described in the retained upstream license.

Every document records its source URL, path, content identity, and license reference. See `data/foundation/sources.json`, `documents.jsonl`, and `licenses/`. The corpus is for this local experiment; the new dataset and model have not been published. Preserve attribution and review the applicable license obligations before distributing corpus extracts or artifacts.

The preparation step normalizes line endings, deduplicates whitespace-normalized whole-document identities, then assigns whole documents to a fixed hash-based split. It does not deduplicate every snippet or near-identical page across documents. Related material can occur on both sides; therefore token validation loss is optimistic as a measure of new-task skill.

The tokenizer uses a deterministic source-balanced 500,000-character sample **only from training documents**. Token files are memory mapped and packed with document start/end markers. Windows may cross document boundaries within a split. All next-token targets are supervised; this is not assistant-reply-only training.

Reproduce preparation in a **new** directory:

```sh
python3 -m kelus.corpus --download --root data/foundation-copy
```

Downloads are capped at 256 MiB per archive and checked against recorded SHA256 hashes. Accepted text is capped at 200 MB. Raw archives, licenses, documents, tokenizer, and binary tokens occupy approximately 249 MB; the two model/optimizer checkpoints currently add approximately 184 MB.

## Next stages and release gate

1. Complete this initial pretraining phase; inspect sampled held-out loss, stability, throughput, and repetition in raw continuations. Do not serve a base checkpoint as a chat assistant.
2. Expand or rebalance the corpus based on observed weaknesses. The current corpus teaches Python and web-related prose/code; it is not broad world knowledge or a full pretraining dataset for a general LLM.
3. Prepare a separately tracked instruction dataset containing varied specifications, implementations, explanations, corrections, and tests. Split by task families/source documents before generating variants. Preserve provenance; do not recycle the same 84 replies as the whole instruction dataset.
4. Continue from a selected foundation checkpoint using the existing `kelus.train --init-from ... --data ... --out ...` path. This stage has **not** been started: a useful pretrained checkpoint and the new instruction corpus are prerequisites.
5. Run specification-following function tests and separate website checks on requests absent from training. `python3 -m kelus.code_readiness --checkpoint ...` is the current development baseline. Website structure checks alone are insufficient: inspect the page and test the requested interactions before promotion. Re-run the existing chat/code regressions and report any regressions.
6. Export to the browser and bundle into Kelus IDE only after the candidate shows a concrete improvement on user requests. Training completion alone never replaces the released checkpoint.

## Verification

```sh
python3 -m unittest discover -s tests -v
```

Focused pretraining tests verify next-token alignment, corpus tamper detection, whole-document split identity, exact uninterrupted-versus-resumed weights and Adam state, safe signal stopping, output protection, and recipe mismatch rejection. Existing model gradient, cached decoder, browser parity, and local-tool checks remain in the suite.

## Automatic website publication

A guarded local release watcher is configured. See [AUTO_DEPLOY.md](AUTO_DEPLOY.md) for its prerequisites, status and stop commands. It waits for completed instruction training and coding/browser checks; completing pretraining alone does not publish the model.

## Expanded language target

The requested scope now covers 23 coding languages and seven chat/script targets. See [LANGUAGE_TRAINING.md](LANGUAGE_TRAINING.md). The active corpus/checkpoint/tokenizer are unchanged; a separately prepared, reviewed multilingual stage is still needed. The deployment gate now requires per-language verification in addition to the earlier Python/web tests.
