# Kelus Model B1

A small local chat/code model trained from random weights. **No AI framework, pretrained weights, remote model API, or tokenizer library.** Our transformer, gradients, optimizer, tokenizer, training loop, and inference are implemented here. NumPy supplies array arithmetic.

The current model, **B1 v3**, has **431,424 parameters**: three pre-LN transformer blocks, width 96, four heads, rotary positions, token embeddings shared with the output layer, a locally learned 1,024-token BPE vocabulary, and a 256-token context. B1 is a version name, not a billion-parameter claim.

B1 is narrow. It knows a closed set of chat replies, 20 coding concepts, and 28 short Python/JavaScript snippets, and it now declines specific programs it has not learned. Slang it never saw, open-ended chat, and anything outside its practice topics still fail. See [MODEL_CARD.md](MODEL_CARD.md) for measured results and failures.

## Run

Requires Python 3.10+ and NumPy:

```sh
python3 -m pip install -r requirements.txt
python3 -m kelus.server
```

Open **http://127.0.0.1:8765**, or double-click **Start Kelus.command**. Terminal chat:

```sh
python3 -m kelus.chat
```

Both load `checkpoints/b1-v3.best.npz`; pass `--checkpoint` to use another file. Older checkpoints (`b1-conversation.best.npz`, `b1-current.npz`, `b1-demo.npz`) still load. Replies stream from the neural model, except labeled calculator and clock answers from `kelus/local_tools.py`. Chats stay in memory and never become training data. Generation uses a key/value cache, about 40× faster than the original full-recompute loop with identical output.

## Browser version

`site/` is a static page that runs B1 entirely in the visitor's browser: `site/b1.js` ports the tokenizer, cached transformer decoder, reply rules, and local helpers to JavaScript. The deployed copy is **https://kelus-b1.web.app** (Firebase Hosting site `kelus-b1` in project `kelus-ide`). A banner offers a reload when newer model weights are deployed.

**Accounts.** The site uses the Kelus IDE's Firebase Authentication (same project, same accounts; email/password, Google, GitHub). Signed out, chats exist only on the page. Signed in, each chat is saved after every reply to `users/{uid}/b1Chats/{chatId}` (title, up to 200 messages, server timestamps) and the saved-chat list updates live across tabs and devices. Inference never leaves the browser; only signed-in transcripts are stored. Access rules live in the Kelus repository's `firestore.rules` (owner only, validated shape) and deploy from there with `firebase deploy --only firestore:rules`. `site/account.js` holds the account code; `site/vendor/firebase.js` is the Firebase Web SDK bundled from `site-src/firebase.js` (build command in that file). Google and GitHub sign-in need `kelus-b1.web.app` in Authentication → Settings → Authorized domains.

```sh
python3 -m kelus.export_web --checkpoint checkpoints/b1-v3.best.npz --out site/model
python3 -m http.server 8790 --directory site          # local preview
firebase deploy --only hosting:kelus-b1 --project kelus-ide
```

`tests/test_web.py` checks that the JavaScript engine matches Python token for token (both position types, Unicode, code fences, calculator edge cases); it needs Node.js. `tests/run_site_e2e.sh` tests accounts in two headless Chrome tabs against the Firebase emulators: sign-up, saving, reopening, live sync, deletion, and sign-out. It needs Chrome, Java, and the Firebase CLI, and never touches real accounts.

## From-scratch foundation project

A separate **8.69M-parameter, 1,024-token-context** model is now being pretrained from random weights on Python source/documentation and MDN web-development documents. See [FOUNDATION_TRAINING.md](FOUNDATION_TRAINING.md) for the measured pilot, source attribution, progress/stop/resume controls, and release criteria. This research run does not replace the current v3 chat checkpoint.

## Requested coding and chat languages

The next training target is 23 coding languages and seven chat language/script targets, including Korean and both Chinese writing systems. These are not current model capabilities. [LANGUAGE_TRAINING.md](LANGUAGE_TRAINING.md) documents the exact list, reviewed dataset format, balanced sampling and release checks. Inspect the registry with `python3 -m kelus.curriculum --targets`.

## Train

Curriculum v3 is generated deterministically:

```sh
python3 data/make_v3.py        # writes data/conversations-v3.jsonl
python3 -m kelus.train --out checkpoints/my-b1.npz
python3 -m kelus.progress      # live progress bars, or double-click "Training Progress.command"
```

Defaults reproduce B1 v3: `data/conversations-v3.jsonl`, 12,000 steps, batch 16, width 96, 3 layers, 4 heads, context 256, BPE 1,024, rotary positions, tied embeddings, dropout 0.1, AdamW LR 0.0015 with weight decay 0.01, validation every 250 steps, early stopping after 10 checks without improvement. The included run took about 24 minutes on an Apple M4 CPU; its best validation loss came at step 10,000.

Each run writes `my-b1.npz` (latest completed interval; resume from this) and `my-b1.best.npz` (lowest full-validation loss; chat with this). Existing outputs are protected; choose a new path or pass `--overwrite`.

```sh
python3 -m kelus.train --resume checkpoints/b1-v3.npz --steps 1000
```

Double-clicking **Train B1.command** runs that resume command. Resume checks dataset identity and restores Adam moments, seed, and sampling state. Warmup/cosine scheduling restarts for each invocation. Ctrl+C keeps the latest completed checkpoint.

Architecture flags for new runs: `--positions learned|rope`, `--tie-embeddings/--no-tie-embeddings`, `--dropout`, `--width`, `--layers`, `--heads`. `--positions learned --no-tie-embeddings --dropout 0 --batch 8 --steps 6000 --patience 8 --data data/conversations.jsonl` recreates the original B1 recipe (that tokenizer was BPE v1). `--bpe-vocab 0` selects the byte tokenizer. `--init-from` starts a new curriculum from existing weights with fresh optimizer state.

## Evaluate and verify

For the requested ability to invent code and build websites, run `python3 -m kelus.code_readiness`. The current v3 model passes **0/4** small specification-following Python tasks and produces none of the requested HTML/CSS/JavaScript structure for the website task. This is a separate development check, not the practiced-snippet benchmark. Adding a reasoning label or more training steps on the same 84 answer texts does not supply the missing coding capability.

```sh
python3 -m unittest discover -s tests -v
python3 -m kelus.benchmark --show-failures              # held-out probes + context regression
python3 -m kelus.evaluate --out reports/b1-eval.json    # original regression suite
python3 -m kelus.evaluate_repair --checkpoint checkpoints/b1-v3.best.npz --out reports/b1-repair.json
```

`kelus.benchmark` scores 199 hand-written prompts (`kelus/probes.py`) on practiced topics with new wording: exact canonical replies for chat, an AST-checked interpreter for Python functions. It refuses to run if any probe appears in a training split. A separate context check sends trained messages such as `hi!` after 23 kinds of earlier exchange. Generated code is never passed to `eval` or `exec`. Reports for the three main checkpoints are in [reports/benchmark/](reports/benchmark/).

## Data format

One conversation per JSONL line, alternating user/assistant and ending with a reply:

```json
{"split":"train","messages":[{"role":"user","content":"hello!"},{"role":"assistant","content":"Hi! What would you like to build?"}]}
```

An assistant message with `"supervise": false` is context only and is never a training target, for example a deliberately wrong reply that the user then corrects. Use explicit `train`/`validation` splits for every record, or omit `split` everywhere for a seeded 85/15 split. At least four distinct conversations are needed. The BPE vocabulary is learned only on the training split.

## Implementation

| File | Responsibility |
|---|---|
| `kelus/tokenizer.py` | Local BPE (v2 keeps a code fence and its language tag in one piece), byte fallback, roles, loss masks |
| `kelus/tensor.py` | Reverse-mode automatic differentiation |
| `kelus/model.py` | Causal pre-LN transformer, learned or rotary positions, optional tied embeddings and dropout, cached generation, checkpoints |
| `kelus/optim.py` | AdamW and gradient clipping |
| `kelus/data.py`, `kelus/train.py` | Splits, batches, full validation, best selection, resume |
| `kelus/runtime.py`, `kelus/local_tools.py` | One-prior-exchange reply policy; labeled calculator and clock |
| `kelus/benchmark.py`, `kelus/probes.py` | Held-out probes and context regression |
| `kelus/evaluate.py`, `kelus/evaluate_repair.py` | Original regression suites |
| `kelus/export_web.py`, `site/` | Browser export and static in-browser chat; `site/account.js` accounts and saved chats |
| `kelus/progress.py` | Live training progress bars |
| `kelus/server.py`, `kelus/chat.py`, `web/` | Local HTTP and terminal chat |
| `data/make_v3.py` | Curriculum v3, decontaminated against every probe |
| `data/make_conversations.py`, `data/make_repair*.py` | Earlier curricula, kept for comparison |

The original 94-example corpus remains in `data/starter.jsonl`; its checkpoint mainly memorized examples.
