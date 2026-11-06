# Cide: Kani's coding specialist

The current `/cide` expert is **Cide 2**, selected by `kani-builds.json` in both
the Node CLI and Python handoff. Its exported coding weights match the website.
The older expert remains selectable with `kani --build cide-v1`.

Cide 2 scored 8/9 on basic coding checks and 23/24 on reserved CRUD checks,
but 0/5 on broader development checks. A backend probe generated invalid
JavaScript. It is experimental and cannot reliably build complete SaaS apps.
The v1 training and evaluation details below are historical.

## Broken-code repair pass (2026-10-28)

Two separate problems were reproduced in Cide 2. Its generated website was cut
off at both the Node CLI's 512-token limit and the site's 768-token limit; the
same weights finished the HTML document in 1,277 tokens when given more room.
Completing that document did **not** fix a missing requested feature.

Coding responses now have a separate 2,048-token output ceiling, still bounded
by the model's total context. The website no longer inherits the router's short
chat budget. Node `--max-tokens` remains an explicit override; Python `ask` has
`--cide-max-tokens`, and the editor uses `kani.cideMaxTokens`. This is a local
runtime fix, not new weights or evidence of improved reasoning.

The current release passed **0/12** newly authored natural-wording SQLite
behavior checks (`training-runs/cide-2-1/baseline-wording.json`). The candidate
curriculum contains 5,760 training and 84 validation examples: 2,880 new wording
and repair examples plus existing code/website replay. All 384 new reference
contracts passed execution checks. Test styles and function names are excluded
from both prepared splits; existing replay retains its original split.

`python -m kelus.cide_candidate` waits for active GPU training, then runs at most
1,500 new updates from the preserved Cide 2 release (batch 2, learning rate
0.00002, 8 GiB GPU cap). It requires 6 GiB free disk space and refuses duplicate
runners. `--resume` restores a periodic checkpoint and uses only the remaining
step budget. It does not automatically retry failed jobs or publish weights.

Both final and best-validation weights receive new wording, basic code,
development, reserved CRUD, website structure, and language checks. Review
requires at least 10/12 new wording and 3/5 development passes, with no regression
below the recorded Cide 2 baselines in the other suites. The runner writes
`status.json` and `review.json` under `training-runs/cide-2-1/`; even a passing
candidate needs review. These are narrow, procedural component tests, not proof
that Cide can build an arbitrary working application.

Cide is a separate, experimental coding checkpoint descended from our own
Kani v7 weights. Kani's companion routing checkpoint learns to emit `CIDE`
for code requests, `SEARCH: query` for web lookups, or an ordinary chat reply.
The application executes that decision. It does not choose Cide by matching
coding words in the user's message.

The original request and recent conversation are passed to Cide. A visible
**Cide · coding specialist** label identifies its response. Specialist output
is never interpreted as another tool call, preventing recursive handoffs.

## Use

Once training and local export finish:

```sh
kani --build kani-cide
```

Ask normally: `Write a Python function add(a, b) that returns a + b.`
Kani generates the handoff itself. `/model kani-cide` selects the same build
inside an existing terminal conversation.

Use `/cide TASK` to ask the specialist directly, or `/model cide` to stay on
Cide. The VS Code extension offers **Kani + Cide · experimental handoff** and
**Cide · coding specialist** in its build picker. Reload the extension after
installing version 0.7.0.

In the CLI's `kani-cide` and `cide` builds, `/ask`, `/draft`, and `/do` use
Cide's weights with the existing workspace tools:

```text
/do utils.py add a function sum_pair(a, b) that returns a + b --test "python3 -m unittest -q"
/diff PROPOSAL_ID
/apply PROPOSAL_ID
```

`/do` reads the current function, generates candidates, checks syntax, and
runs the supplied test command in the existing macOS sandbox. Successful
edits become proposals. `/apply` writes the reviewed proposal. Natural chat
returns code; it does not independently run shell commands or edit files.
The extension retains its Copy/Insert controls.

The default `latest` build remains Kani v7. This preview has a separate
checkpoint selected explicitly. Website handoff support is in the source;
the live site is not changed by this training run. Hosting it requires the
router at `site/model/` and Cide at `site/cide/`.

## Training

```sh
python3 data/make_cide.py
python3 -m kelus.cide_train
```

The runner allows up to 1,500 expert steps and 1,000 router steps. This first
expert run stopped after step 1,000 when validation loss rose; its best
checkpoint at step 750 was selected. The router stopped early after its
250-step checkpoint passed the routing checks. It creates separate descendants of
`checkpoints/kani-b2-2f-v7-s10000.npz`:

| Release | Training | Runtime role |
|---|---|---|
| `cide-v1-release.npz` | 750 selected steps on code replay and simple repairs | Write code |
| `kani-cide-v1-release.npz` | 250 selected steps on handoffs plus chat/search/reading replay | Choose Cide, search, or chat |

Both use the project's NumPy transformer, autograd and AdamW for this run.
No external pretrained checkpoint is downloaded. This is specialization of
our existing Kani model, **not a new random initialization**. Each model has
8,691,840 parameters and a 1,024-token context.

The builder preserves existing data splits. New repair validation uses
different function names. Replay validation was already available to parent
v7 training, so it is not an unseen benchmark. Releases record their parent
and dataset hashes. JavaScript exports live in `training-runs/build-exports/`;
training checkpoints retain optimizer state and release checkpoints omit it.

### Longer local continuation (2026-10-24)

`training-runs/cide-continuation-v4/` starts from Cide CRUD v3's final 6,000-step
checkpoint and targets **30,000 additional updates** on this Mac's CPU. It uses
the existing NumPy/Kelus trainer with batch 2 and a 0.00005 initial learning rate,
leaving the active Big Kani GPU trainer alone. The parent checkpoint is preserved.

The expanded dataset has **16,251 training examples and 143 validation examples**:
varied SQLite schemas, identifiers and limits, plus existing SaaS/web-code and
basic-code replay. All **9,696 newly generated reference contracts** were executed
before training. Reserved CRUD test schemas are excluded and training/validation
records do not overlap. These are procedural variations and replay, not thousands
of independent complete applications.

Checkpoints save every 250 updates. Both final and best-validation checkpoints
receive behavior checks when training finishes; neither is automatically deployed.
The previous run's weak generalization remains the baseline, so a lower training
loss alone will not be described as improved SaaS capability.

```sh
python3 -m kelus.progress training-runs/cide-continuation-v4/instruction.log --full
```

Closing the progress monitor does not stop training. `caffeinate` prevents idle
sleep during the local job; closing the lid or shutting down can interrupt it.
After an interruption, resume the saved checkpoint with:

```sh
.venv/bin/python -m kelus.cide_continue --steps 30000 --resume
```

The existing CPU trainer restores optimizer/RNG state, with a new learning-rate
warmup/decay over the remaining updates. Dataset provenance and the fixed run
recipe are saved in `data/cide-continuation-v4/manifest.json` and the run directory.

## Evaluation and limits

### Local contract continuation v5 (2026-10-26)

The v4 run finished 30,000 updates on October 25. Both its final and best
checkpoints passed 9/9 basic checks, 0/5 development checks and 1/24 reserved
CRUD checks. Several generated functions changed the exact requested function
name, even when their SQL was close to the contract.

`training-runs/cide-contracts-v5/` continues v4's final checkpoint for **10,000
new updates on this Mac**, using NumPy/Kelus autograd, batch 2 and initial
learning rate 0.00005. This is the existing 8.69M-parameter Cide lineage,
not the separate 205.6M-parameter Kani Advanced model from Modal. No cloud
service is launched and no pretrained external weights are used.

The new curriculum contains **13,200 training and 191 validation examples**:
independently varied function/table names, exact-signature exercises, and
existing SaaS/basic-code replay. All **7,344 newly generated reference functions**
were executed before launch. Reserved CRUD schemas remain excluded. These are
procedural component exercises, not complete applications; the old reserved
checks are reused evaluations rather than a new untouched benchmark.

Checkpoints save every 250 updates. The runner evaluates final and best weights
on the existing basic, development and reserved CRUD checks, without deploying.

```sh
# Live monitor; Ctrl+C closes only the monitor.
.venv/bin/python -m kelus.progress training-runs/cide-contracts-v5/instruction.log --full

# Resume this exact run after an interruption, only when it is not running.
.venv/bin/python -m kelus.cide_continue --steps 10000 --resume \
  --run training-runs/cide-contracts-v5 \
  --data data/cide-contracts-v5/instructions.jsonl \
  --init-from training-runs/cide-continuation-v4/instruction.npz
```

`process.json`, `recipe.json`, `status.json` and `instruction.log` in the run
directory record the process, provenance, state and progress. The background
launch uses `caffeinate -i`; keep the Mac awake with its lid open. Old checkpoints
and prepared datasets are retained. `data/make_cide_contracts.py` reproduces the
dataset in a fresh output location and refuses to overwrite existing data.

### Longer SaaS curriculum: Cide v2

The separate `training-runs/cide-saas-v2/` run continues our own Cide v1
checkpoint for **50,000 additional optimizer steps** using NumPy and Kelus
autograd. It does not download pretrained weights or replace a selected build.

1. **30,000 code steps:** 1,024 tokens per update, learning rate 0.00015,
   checkpoints every 100 steps, fixed sampled validation. The prepared training
   stream contains 10,208,918 tokens from 3,649 distinct source documents.
   Next.js and FastAPI app files are repeated eight times in this stream;
   that weighting does not represent additional unique data.
2. **20,000 instruction steps:** batch 2, learning rate 0.00008, no early stop,
   checkpoints every 250 steps. There are 16,732 training examples and 104
   validation examples: web-code continuations, eight small SaaS exercise
   families with identifier variations, and Cide v1 replay.

Sources are pinned revisions of [Next.js SaaS Starter](https://github.com/nextjs/saas-starter),
[FastAPI's full-stack template](https://github.com/fastapi/full-stack-fastapi-template),
and the already-audited [MDN content](https://github.com/mdn/content) corpus.
Authentication, database access, ownership checks, dashboards, forms and CRUD
appear in these sources. Source URLs, hashes, splits and retained licenses
are in `data/cide-saas-v2/`. Downloaded code is treated as training data and
is not executed. Validation loss does not establish working-app capability.

Watch the whole run, including the automatic transition between phases:

```sh
python3 -m kelus.cide_saas_train --watch
```

Use `--status` for one snapshot. Ctrl+C closes the monitor without stopping
training. The launched background job uses `caffeinate` to prevent idle sleep
while it runs; closing the laptop or shutting it down can still interrupt it.

To reproduce the dataset/run in a fresh output directory, use
`python3 -m data.make_cide_saas` and `python3 -m kelus.cide_saas_train`.
Existing output is protected against accidental replacement. To continue
after stopping the pipeline, use `python3 -m kelus.cide_saas_train --resume`.
Raw-code resume restores its exact optimizer/RNG schedule. The existing
instruction trainer restores optimizer/RNG but starts a fresh learning-rate
warmup/decay for its remaining steps.

After both budgets finish, the runner writes
`training-runs/cide-saas-v2/cide-saas-v2-candidate.npz` and runs arithmetic
and SQLite behavior checks. The prior Cide v1 scored **0/5** on the new
SQLite checks; its responses are recorded in `baseline-saas.json`.
These checks cover user ownership, input validation, injection resistance,
and CRUD behavior in memory. They do **not** certify a complete SaaS site,
login flow or deployment. The candidate is not automatically released.

### First-preview checks

The v2 run completed all 30,000 code updates and 20,000 instruction updates.
Its selected candidate passed 9/9 basic arithmetic/repair checks but **0/5**
SQLite development checks; the final-step checkpoint also passed 0/5.
The candidate reproduced all five original CRUD training prompts exactly.
Those observations show failure to generalize the learned CRUD patterns to
changed requests; they do not establish working SaaS capability.

A separate `cide-crud-v3` repair run completed all 6,000 instruction updates
from the v2 candidate. Its 9,398 training examples emphasize six CRUD operations
with varied function names, schemas, argument names, validation limits, ordering
and prompt wording. All 2,952 reference contracts were executed before training.
These are procedural variations, not independent apps. A separate set of 24
schema/identifier cases starts at 0/24; those cases are absent from the new
training/validation schemas. The old five checks are now development checks.

```sh
python3 -m kelus.progress training-runs/cide-crud-v3/instruction.log
```

The repair run saves separate checkpoints, then automatically runs basic-code,
development and held-out behavior checks. It does not replace the active CLI,
extension or website model. This milestone is reliable CRUD components; complete
authentication, browser behavior and multi-file SaaS apps still require validation.

The completed run was checked at both the automatically selected checkpoint
(step 250, lowest validation loss) and the final checkpoint (step 6000):

| Checkpoint | Basic code | SQLite development | Unseen CRUD contracts |
| --- | --- | --- | --- |
| Selected candidate, step 250 | 9/9 | 0/5 | 0/24 |
| Final weights, step 6000 | 9/9 | 0/5 | 1/24 |

These results do not establish reliable CRUD or full SaaS capability. The final
checkpoint is retained as `training-runs/cide-crud-v3/instruction.npz`;
`final-basic-code.json`, `final-development.json`, and `final-heldout.json` in the
same directory preserve its evaluated outputs. Training is complete; neither
checkpoint was promoted to the CLI, extension, or website by this run.

```sh
python3 -m kelus.cide_eval --checkpoint checkpoints/cide-v1-release.npz --role coding --out training-runs/cide-v1/code.json
python3 -m kelus.cide_eval --checkpoint checkpoints/kani-cide-v1-release.npz --role router --out training-runs/cide-v1/router.json
```

These are small, inspectable smoke checks for arithmetic, addition repairs,
and routing. They preserve every prompt and answer. Passing them does not
establish broad programming competence. Cide has not reached Codex's
coding, planning, debugging, context capacity, or tool-use ability.
Open-ended tasks and even conversational identity questions remain unreliable;
the named Cide label comes from its checkpoint metadata. Keep ordinary chat
with Kani and verify Cide's code against the actual requested behavior.

First-preview results: **9/9** Python/JavaScript behavior checks and **10/10**
route-category checks. A four-turn conversation also switched chat → Cide →
chat → search. A real CLI repair passed sandboxed tests, staged a proposal,
and passed again after applying it in a temporary project. The extension's
real Python bridge streamed a learned handoff and repaired code. Full results
and weight hashes are in `training-runs/cide-v1/release.json`.

Search-query completeness is not guaranteed: in a multi-turn probe, both v7
and this router omitted "of France" from a search. The routing score measures
the choice of tool, not the correctness of every query or final answer.

Large source excerpts are rejected rather than silently cut. Older complete
chat exchanges may be omitted. Syntax checks do not establish correctness;
use behavioral tests and review proposals. Missing weights produce an error
instead of labeling a Kani answer as Cide.
