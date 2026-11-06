# Big Kani v2 paragraph pilot

This local experiment continues the completed step-9000 Big Kani v2 chat
checkpoint for **400 additional updates**. It uses the existing MLX/Metal
trainer with a 9 GiB memory cap and Rust tokenization. No cloud spending.

The mixture contains 1,725 distinct paragraph answers and 431 short-answer
replay examples from the existing attributed Dolly/OASST2 corpus. There are
2,052 training and 104 validation records. Original source IDs and splits are
preserved; exact duplicate answers are excluded. Format filtering does not
establish factual correctness. The parent may already have seen related data.
Training loss checks use a fixed sample of 64 of the 104 validation records.

Artifacts live in `training-runs/kani-big-v2-paragraph-pilot/`. `data/manifest.json`
records hashes and selection counts; `data/ATTRIBUTION.md` records source licenses.

Six writing tasks and two reading/correction regression checks are recorded
before training and repeated on best/final checkpoints. Writing needs manual
review for relevance, clarity, requested details, repetition and paragraph
format. Length alone is not a quality score. A lower validation loss alone
does not justify a website release. This runner never deploys automatically.

Monitor:

```sh
.venv/bin/python -m kelus.progress training-runs/kani-big-v2-paragraph-pilot/instruction.log --full
```

Resume only after the old trainer has stopped:

```sh
caffeinate -i .venv/bin/python -u -m kelus.local_chat_train \
  --run training-runs/kani-big-v2-paragraph-pilot \
  --parent checkpoints/kani-big-v2-chat.best.npz \
  --data training-runs/kani-big-v2-paragraph-pilot/data/instructions.jsonl \
  --steps 400 --cases evaluation/paragraph_v1.json \
  --last-reply-only --probe-max-tokens 256 --resume
```

The runner requires 6.5 GiB of free disk space before training starts. Latest
checkpoints include optimizer state; best checkpoints contain weights only.
Closing the progress monitor does not stop training. Keep the Mac awake with
its lid open and connected to power while the experiment runs.
