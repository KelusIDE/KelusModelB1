# Overnight Kani run — October 28, 2026

Supervisor: `training-runs/overnight-20261028/`.

The `450/450 failed` display came from an intentionally interrupted topic run.
The progress monitor now uses the recorded 1,500-step target and distinguishes
KeyboardInterrupt from an actual crash. Restart an already-open monitor to load
that display fix. The stopped topic run currently shows 450/1,500 (30%).

The overnight supervisor waits for the existing Big Kani v2 chat trainer to finish
its 9,000-step target. It then freezes that run's best checkpoint and starts a
separate 1,500-update topic-following curriculum using latest-reply supervision.
All GPU stages run sequentially. The new topic stage records baseline, best and
final responses on the same 16 probes. It does not automatically publish weights.

The supervisor has a 12-hour ceiling, checks space before starting another stage,
and permits one recovery attempt per stage. Big Kani recovery uses a smaller
batch/memory cap if needed. Failed recovery stops with a saved status and logs.
The latest checkpoints contain optimizer state; atomic replacements preserve
the prior complete checkpoint if a write is interrupted.

Keep the Mac plugged in with its lid open. `caffeinate -i` prevents idle sleep;
it cannot keep training through shutdown or a closed laptop lid.

## Status

```sh
cat training-runs/overnight-20261028/status.json
.venv/bin/python -m kelus.progress training-runs/kani-big-v2-chat.log --full
```

Later, the focused run logs appear under
`training-runs/overnight-20261028/topic/`. A completed overnight run writes
`RESULTS.md` and `results.json` in the supervisor directory. These distinguish
the nine narrow automatic checks from the seven manual-review cases. A lower
loss or a higher short-answer score does not establish better general dialogue.

To stop the entire overnight workflow, including its active training stage:

```sh
touch training-runs/overnight-20261028/STOP
```

Inspect `supervisor.log` and the current stage's training log if status says
failed. The workflow does not delete training data, spend cloud credit, or
replace the website/CLI model automatically.
