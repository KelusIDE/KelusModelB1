# B1 language expansion

These are **requested training targets**, not capabilities of the released v3 model or the running foundation checkpoint. The exact target list and aliases live in `config/languages.json`.

| Type | Targets |
|---|---|
| Coding (23) | Python, JavaScript, TypeScript, Java, C, C++, C#, PHP, Go, Rust, Swift, Kotlin, Ruby, SQL, R, Scala, Perl, Dart, Lua, COBOL, PowerShell, Zsh, Bash |
| Chat (7 language/script targets) | English (`en`), Korean (`ko`), French (`fr`), Spanish (`es`), German (`de`), Simplified Chinese (`zh-Hans`), Traditional Chinese (`zh-Hant`) |

`ps1` maps to PowerShell. C, C++ and C# stay distinct, as do Bash/Zsh and the two Chinese writing systems. The ambiguous labels `shell` and `Chinese` must be made specific. HTML/CSS remain part of the existing website requirement; tag a mixed web project with its primary scripting language and retain its HTML/CSS in the actual example.

## What is ready and what remains

Ready: target registry, attributed instruction-data preparation, family-based splits, coverage/answer-diversity reporting, optional balanced training batches, checkpoint/export metadata, and per-language release verification requirements.

Pending: assembling and reviewing a varied corpus and instruction dataset for the new targets, training on them, and measuring the resulting model. **No multilingual training run has been started.** The current English/Python/web foundation run continues unchanged. These changes do not make the current model understand the listed languages. A small 8.69M model should be evaluated incrementally; no general coding/fluent-chat quality is promised across this scope.

The existing byte/BPE tokenizer can represent all requested writing systems. Round-trip tests prove text preservation only. Its vocabulary was learned from English-heavy material; before a multilingual stage, measure token lengths and truncation on real Korean/Chinese records. Preserve token IDs when continuing existing weights. Changing vocabulary requires a separate, explicit migration or training run, not replacement of the active tokenizer file.

## Prepare reviewed instruction data

Each JSONL record contains a complete user/assistant conversation, language labels, a task family shared by related implementations/translations, and source provenance. Example schema (one example does not constitute a training corpus):

```json
{"family":"string-prefix-removal","code_language":"python","reply_language":"en","reviewed":true,"source":{"id":"kelus-original-example","license":"Original project-authored text/code","reference":"LANGUAGE_TRAINING.md example"},"messages":[{"role":"user","content":"In Python, remove the leading demo_ from a string only if it is present."},{"role":"assistant","content":"Use text.removeprefix(\"demo_\"). Strings without that prefix are unchanged."}]}
```

Use `code_language: null` for plain conversation. `reviewed: true` is an author/reviewer assertion that the answer and provenance were checked; the builder cannot verify its truth. Imported records must identify their actual source, license and revision/reference. Do not relabel third-party material as original.

A family represents a semantic task, not an individual phrasing: all translations, template variants, and language ports of that task share it. Default splitting hashes the family into train (80%), validation (10%), or held-out test (10%). Explicit splits are accepted but conflicts are rejected. Exact duplicated conversations with conflicting labels are rejected. This cannot automatically find semantic duplicates assigned misleading family names; review family assignment.

```sh
python3 -m kelus.curriculum --input data/reviewed-instructions.jsonl \
  --out data/instructions-multilingual-v1 --require-all
```

The output directory must be new. It contains:

- `instruction.jsonl`: train and validation examples, readable by the existing trainer.
- `heldout.jsonl`: test examples, excluded from training.
- `coverage.json`: source/output hashes; examples, distinct final answers, task-family counts, missing languages and code-language/reply-language cell counts per split.

`--require-all` requires every target coding language and chat/script target in each split. It is a presence check, not a quality threshold or coverage of all 161 code/chat combinations. Inspect the cell counts and varied task families as well. Without it, partial datasets can be prepared for inspection, with gaps explicitly reported. Nothing is downloaded or synthesized automatically.

## Train after curation and foundation evaluation

The existing trainer accepts prepared records and supports `--balanced-languages`. It samples each **populated** code-language/reply-language cell equally, then chooses an example inside that cell. This prevents a large English/Python cell from dominating every batch. It can repeat small cells more often; collect more varied examples rather than treating oversampling as new knowledge.

A future instruction-stage invocation, after the input data exists and has been reviewed:

```sh
python3 -m kelus.train --init-from training-runs/b1-foundation/best.npz \
  --data data/instructions-multilingual-v1/instruction.jsonl \
  --out checkpoints/b1-multilingual-v1.npz --balanced-languages \
  --batch 1 --steps 1000 --lr 0.0001
```

Those step/LR values are a starting experiment, not an estimated route to full support. Measure held-out performance before selecting the next budget. The example is **not running**. It continues our own randomly initialized model; no pretrained weights, remote model or AI framework is introduced. Use temporary sleep prevention for a real long run.

Balanced sampling is saved in checkpoint metadata and restored on resume; switching the sampling policy during resume is rejected. `language_training` metadata in exported browser manifests describes data exposure, not verified fluency. Legacy untagged datasets retain their original sampling behavior. Tagged held-out test records are rejected by the training loader.

## Release evidence

Automatic deployment now additionally requires `language-verification.json`, bound to the candidate checkpoint and the target-list SHA256. Preparation writes an **unverified** `language-verification.template.json`; every check starts false. The normal coding, website-interaction and regression gates still apply.

For each coding language, verify generated outputs on previously unseen requests for:

1. Implementing a concrete specification.
2. Repairing a supplied bug.
3. Handling edge cases with correct results.

Record prompts, generated output/artifact paths, compiler/interpreter/database versions or dialects, expected/actual results, and reviewer identity. Syntax highlighting, a code fence, or compilation alone is insufficient. Shell/PowerShell examples must be tested in an appropriate isolated environment; the release gate never executes generated commands on this Mac.

For each chat/script target, review instruction following, a code explanation and a clarification exchange for meaning and correct language/script. Include coding prompts in the target human languages so code-explanation tests cover cross-language requests. A Unicode round trip or the presence of Chinese/Korean characters is insufficient. Keep the actual prompt/output evidence available; a passing local JSON record is a review receipt, not an automated competence test.

Missing languages, failed checks, changed target lists, or a checkpoint mismatch block publication. The current v3 site is not relabeled with these targets. Passing this small release checklist still does not establish general expertise in every language.
