# Kani training archive

Old Kani experiments moved off the training Mac on 2026-10-29 to free disk space. None of these models is live.
Weights are Git LFS files (`*.npz`, kelus checkpoint format, load with `kelus.model.Model.load`). Paths match the
main branch layout, so a folder can be copied back into `training-runs/` or `checkpoints/` as it is.

Only weights and results are kept. Optimizer states (over GitHub's 2 GB LFS file limit), in-between snapshots,
an old cloud upload bundle and a Python environment were deleted, not archived.

| Path | What it was |
|---|---|
| `training-runs/kani-topic-v1/` | Topic-following experiment on Big Kani v2 (best checkpoint, data, evaluations) |
| `training-runs/overnight-20261028/` | Overnight topic run: 9,000 updates, probes 1/9 before and after |
| `training-runs/kani-big-v2-paragraph-pilot/` | 400-update paragraph-writing pilot; replies still repeated |
| `training-runs/kani-paragraph-mac-v1/` | Earlier paragraph-writing experiment |
| `training-runs/kani-chat-v2-eval/` | Chat v2 final snapshot (s9000) and its evaluations; replaced by Otter 4.4 |
| `training-runs/kani-big-v1/` | Big Kani v1 base model; replaced by Big Kani v2 |
| `training-runs/kani-big-v2-seed/` | Starting weights of Big Kani v2 |
| `checkpoints/kani-big-v2-chat.best.npz` | Big Kani v2 chat run, best checkpoint |
| `checkpoints/kani-otter-4.npz`, `kani-otter-4-1.npz` | Released Kani Otter 4 and 4.1 |
