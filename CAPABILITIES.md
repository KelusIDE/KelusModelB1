# Kani capability experiments

The shipping 431,424-parameter model and the experimental 8,691,840-parameter
checkpoints are different models. Tool improvements do not change the released
weights. All model training here still uses our transformer, gradients and BPE;
NumPy performs array arithmetic. No pretrained model or external model API is used.

## Conversation, code and vector images

`data/make_capabilities.py` creates an original curriculum with topical conversation,
corrections, short history, SVG, several longer Python programs, JavaScript debounce,
source-edit examples and replay from prior instruction data. `capabilities-v2.jsonl`
has 2,851 records. A separate candidate continued from our frozen instruction
checkpoint for 1,400 steps; the existing instruction run was left intact.

The diagnostic results in `reports/capabilities-v2-final.json` are deliberately
narrow. This is mostly held-out wording within practiced task families:

| Check | Before this curriculum | After 1,400 steps |
| --- | ---: | ---: |
| Topic terms and memory/correction | 0/6 | 4/6 |
| Entirely new conversation topics | 0/2 | 0/2 |
| Valid SVG plus requested geometry/colors | 0/5 | 4/5 |
| Python behavioral tests | 0/5 | 2/5 |

All replies are retained for inspection. Topic-term matches alone do not prove
coherence. Python grouping and CSV parsing passed edge-case assertions in the
macOS sandbox; LRU cache, flattening and interval probes still failed. These
results **do not establish general reasoning or readiness for large projects**.
The candidate answers `I love pancakes.` on topic, but still fails unfamiliar
paraphrases. Training loss by itself hid this weakness.

Try the candidate explicitly, without changing the shipping model:

```sh
python3 -m kelus.export_web --checkpoint checkpoints/kani-capabilities-v2.best.npz --out training-runs/capabilities-v2-web/model
kani --model-dir training-runs/capabilities-v2-web/model "I love pancakes."
kani --model-dir training-runs/capabilities-v2-web/model "/svg icon.svg a small blue circle"
```

SVG previews/downloads use the actual model output. The validator accepts a
limited shape/path/text profile and rejects active or linked content. Failed
generation produces an error, not a hidden template image. `/do` now supplies
the existing function and failed checks to retries; it still edits one function
at a time and requires a reviewed proposal before applying.

## Math and online reading

`30% tip of 100 is?` now produces `Tip: 30. Total: 130.` and the arithmetic,
labeled LOCAL CALCULATOR in Python, Node and browser clients. Percentage, tip,
tax and discount patterns are supported. Arbitrary prose problems are not solved.

`/web QUERY` searches English Wikipedia; `/web read TITLE` reads article text.
These are labeled retrieved excerpts with source URLs, not generated claims.
Only that lookup goes online. Article text cannot issue commands or modify files.
HTTP 429 responses honor Retry-After; there is no automatic retry loop.

The integration follows the official [search API](https://www.mediawiki.org/wiki/API:Search)
and [plaintext extraction API](https://www.mediawiki.org/wiki/API:Get_the_contents_of_a_page).

## Article training

The original foundation corpus already contains Python and MDN documentation.
`python3 -m kelus.articles` separately collects 34 general Wikipedia articles:
28 training articles and six whole validation articles. It also replays 80
training and eight validation documents from the original corpus. The resulting
dataset has 593,771 training tokens and 94,417 validation tokens, with the existing
4,096-token vocabulary unchanged. No article is converted into a canned chat reply.

Source URLs, revision IDs, retrieval times, Wikipedia contributor attribution and
the upstream CC BY-SA 4.0 license are retained in `data/articles-v1/documents.jsonl`.
Raw API receipts are cached under `raw/`. These third-party texts retain their own
licenses; the repository's Apache license does not replace them. Related facts
can still overlap between articles; validation loss is not a comprehension score.

The separate article run starts from our own foundation checkpoint with a fresh
optimizer. It does not replace the chat weights or alter the completed foundation run:

```sh
python3 -m kelus.pretrain --data data/articles-v1 --run training-runs/kani-articles-v1 --init-from training-runs/b1-foundation/best.npz --steps 200 --batch 1 --accumulate 1 --lr 0.00005 --checkpoint-every 50 --validation-batches 4
cat training-runs/kani-articles-v1/status.json
```

This is a small continued-pretraining experiment. Raw article prediction must be
followed by instruction training and independent comprehension/coding evaluation
before these weights can replace the chat model. Reading an article at runtime
does not automatically train or permanently teach Kani.

The first run completed all 200 steps. On the same fixed sample of 4,096 held-out
tokens, loss decreased from 4.0931 to 3.4122. See `reports/article-training.json`
for checkpoint/corpus identities and the measurement scope. This measures text
prediction only; these article weights have not passed chat or coding release tests.

## Kani Seal 3: attachments

Public name: **Kani Seal 3**; development codename: **Kani B2:10F**. These labels
do not specify parameter count. Checkpoint filenames, tokenizer IDs and training
runs are preserved. Older weights may still answer identity questions using
names learned in their training data.

The browser, Python local chat and VS Code panel accept up to four attachments
through a picker, drag/drop or image paste. UTF-8 source/text, HTML, JSON, CSV and
SVG are inspected as text (128 KB each). PNG, JPEG and WebP receive previews,
dimensions and sampled average color (8 MB each, 24 MP max). PDF/Office/archive
formats are unsupported; export to text first.

An empty prompt or “analyze these files” produces **LOCAL FILE INSPECTION**, a
deterministic report. Specific text questions use actual model inference on
complete, relevance-selected source lines within the loaded tokenizer's context
budget. Answers report the exact included lines. Source lines that cannot fit
are refused. The model can still give incorrect explanations.

Image pixels are not passed to the language model. This is not OCR, object
recognition or screenshot understanding. The separate scratch UI-region
prototype remains accessible through `/vision` in the CLI.

File bytes and attached-source prompts are processed on the device. On the
hosted site, the whole conversation becomes local-only after an attachment is
sent, including subsequent replies; starting a new chat restores normal account
saving. Files remain selected until removed or a new chat starts. HTML/scripts
are never executed; attached sources never trigger tools. No attached file is
automatically added to training.
