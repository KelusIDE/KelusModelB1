# Kani topic-following experiment

The screenshots reveal failures in intent following, arithmetic paraphrases,
reference resolution and grounded answers. The deployed Otter 4.3 weights directly
reproduce the casual `1+1` failure with helpers disabled; see
`training-runs/topic-failure-reproduction.json`. This is a model failure as well
as a limitation of the exact-expression calculator route. The website's source
reader and tool routing are not evidence that the underlying model understands.

## Local experiment

`training-runs/kani-topic-v1/` continues the separate 205.6M-parameter
`checkpoints/kani-big-v2-chat.best.npz` candidate. A byte-identical parent is now
preserved at `training-runs/kani-topic-v1/parent-preserved.npz`, because the Big
Kani chat run was resumed and can update its original best-checkpoint path.
The run records parent/data/probe
hashes before starting and preserves all original checkpoints.

- 17,508 training and 898 validation conversations.
- Diverse, deduplicated Dolly/OASST2 instruction replay, with no legacy generic
  decline replies, plus 1,885 checked procedural examples across arithmetic
  wording, reference contrasts and corrections in conversation.
- Procedural validation uses different first operands and object nouns. The
  exact 16 development probes are excluded from both prepared splits.
- Latest-reply-only supervision: earlier turns supply context; only the final
  assistant reply is trained. Earlier replies in the correction exercises are
  explicitly marked as context only.
- 1,500 updates on the Mac, batch 2, initial learning rate 0.00005, 9 GiB GPU
  memory cap. Rust handles tokenization; MLX/Metal handles neural updates.
- Atomic checkpoints every 250 updates with fast lossless compression. The
  latest retains optimizer state and the best retains weights only.

The run first records direct-model baseline outputs. After training it records
outputs from the best and final checkpoints using exactly the same probes.
Automatic checks cover short answers; writing, emotional relevance and factual
grounding remain manual review tasks. Removing canned replies does not by itself
establish correctness. The reported screenshots are known development failures,
not an untouched benchmark. Procedural exercises cover narrow skills rather
than general reasoning; broad conversation improvement remains to be measured.

## Monitor

```sh
.venv/bin/python -m kelus.progress training-runs/kani-topic-v1/instruction.log --full
```

Before training begins, read `status.json` and `baseline.log` in the run directory.
The stages are baseline → training → evaluating → complete. Keep the Mac awake
with its lid open; the background launch uses `caffeinate -i`. This run creates
no cloud charges and does not publish a new website model.

To resume an interrupted run once the old process has exited:

```sh
caffeinate -i .venv/bin/python -u -m kelus.local_chat_train --resume \
  --run training-runs/kani-topic-v1 \
  --parent training-runs/kani-topic-v1/parent-preserved.npz \
  --data training-runs/kani-topic-v1/data/instructions.jsonl \
  --steps 1500 --cases evaluation/topic_v1.json --last-reply-only
```

Prepared data manifests retain source hashes; source licensing/attribution for
the human instruction replay is in
`training-runs/kani-advanced-instruction-pilot/data/ATTRIBUTION.md`.
