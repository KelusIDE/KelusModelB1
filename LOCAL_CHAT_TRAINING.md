# Local paragraph and conversation training

Started October 27, 2026. The run lives in `training-runs/kani-paragraph-mac-v1/`.
It continues our own **205,588,480-parameter** base at
`training-runs/runpod-big-v2/kani-advanced.npz`, a separate candidate from the
Otter 4.2 model currently selected on the website. Original weights are preserved.

## What runs

- **Rust BPE** prepares the instruction examples using `kaggle/kani_tokenize`.
  A local benchmark encoded 511,405 tokens in 0.0335 seconds versus Python's
  0.2168 seconds (**6.47×**). This measures tokenization, not neural-network
  training speed. Exact token/mask/sample parity and Unicode parity tests passed.
- **MLX/Metal** runs the existing transformer and AdamW updates on the Mac GPU.
  The numerical loss, gradients and update checks passed against the NumPy model.
  This is not a Rust rewrite of transformer training.
- **8,000 updates**, batch 2, initial learning rate 0.00005, a 9 GiB MLX memory
  cap, checkpoints every 250 updates, progress reports every 25 updates.
  The previous larger local run exited 137; the cause was not conclusively
  established, but the smaller batch and cap reduce memory pressure.
- **15,735 training / 786 validation conversations** from the prepared Dolly and
  OASST2 subset: writing, explanations, summarization, extraction and dialogue.
  Repeated exact prompts/replies and the old stock-response templates are excluded.
  The source hashes, identifiers, license attribution and filtering counts are in
  `training-runs/kani-advanced-instruction-pilot/data/`.
- Validation during training uses a fixed **64-example sample**. Lowest sample
  loss selects the best weights; it does not establish better user-facing behavior.

The runner records direct-model baseline replies before training, then evaluates
best/final weights on the same fixed reading, correction, writing, rewriting and
cats-conversation probes. These probes are absent verbatim from the prepared
instruction data. Their responses require review; novel prose can still be wrong.
They are development checks, not a broad intelligence or age-equivalence score.

## Monitor or resume

```sh
.venv/bin/python -m kelus.progress training-runs/kani-paragraph-mac-v1/instruction.log --full
```

`status.json` shows the stage: baseline, training, evaluating, complete, stopped,
or failed. `runner.log` contains supervisor failures. `instruction.log` contains
training progress. Closing the monitor does not stop the background trainer.
`caffeinate -i` prevents idle sleep; keep the Mac's lid open and plugged in.

After an interruption, only when the old process has stopped:

```sh
caffeinate -i .venv/bin/python -u -m kelus.local_chat_train --resume
```

The latest `instruction.npz` retains optimizer state. `instruction.best.npz`
contains weights only to reduce disk use; resume from the latest file. Training
saves atomically and never overwrites its parent. Several GiB of free disk are
required for optimizer checkpoints and their temporary replacements.

An efficiency update resumed the run from the complete step-200 checkpoint:
checkpoint spacing increased from 100 to 250 updates and ZIP compression changed
from level 6 to level 1. This is lossless and preserves all weights, optimizer
arrays, metadata and tokenizer contents. It reduces save frequency by 60%; the
end-to-end speedup must be measured rather than inferred from tokenization speed.

No cloud job or publication is launched by this runner. Review the actual
before/after replies before choosing a release. Prepared data has human-authored
content but was not exhaustively fact-checked; exact deduplication does not remove
all semantic overlap.
