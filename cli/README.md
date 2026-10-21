# Kani Seal 3 terminal workspace tools

Public name: **Kani Seal 3**. Development codename: **Kani B2:10F**. Checkpoint filenames and parameter counts are unchanged.

Run `kani` inside the project you want to work on, or `kani --cwd /path/to/project`.
Restart an already-running CLI to pick up these changes. No external model API is used.
Text chat uses the same scratch model and JavaScript engine as the website.

| Command | Result |
| --- | --- |
| `analyze files` or `/analyze` | File types, project files, entry points and available package scripts |
| `/files [name]` | List eligible project files, optionally filter by name |
| `/read "src/my file.py" 20:60` | Read a bounded line range with source references |
| `/search next_logits` | Literal, case-insensitive source search |
| `/analyze src/main.py` | List declarations/imports found by a text scan |
| `/ask src/main.py:20-40 Explain this function` | Send a source excerpt and question to the model; show exactly which lines fit |
| `/draft square.py Write a Python square function.` | Generate one complete file, syntax-check Python/JS, and save a proposed diff |
| `/do utils.py add square(x) that returns x*x --test "python3 -m unittest -q"` | Generate function candidates, parse and test them in a temporary macOS sandbox, then stage a proposal |
| `/svg icon.svg a small blue circle` | Generate SVG from the loaded model, validate it, and save a preview plus a proposed file |
| `/diff ID` | View a saved proposal |
| `/apply ID` | Write that proposal after review; keep a backup of an existing file |
| `/discard ID` | Delete a proposal without changing the target file |
| `/vision screenshot.png` | Predict a coarse UI region grid with the separate experimental vision model |
| `/web percentages` | Search Wikipedia online and show article excerpts with sources |
| `/web read Percentage` | Read a Wikipedia article by title or its English Wikipedia URL |

The commands also work once and exit:

```sh
kani --cwd ./my-project "analyze files"
kani --cwd ./my-project "/draft square.py Write a Python square function."
kani --cwd ./my-project "/diff ID_FROM_DRAFT"
kani --cwd ./my-project "/apply ID_FROM_DRAFT"
```

File commands are deterministic local helpers, clearly labeled. They work without
loading model weights. `/ask`, `/draft`, `/do` and `/svg` use the actual loaded neural network.
The released 431K model has only 256 tokens of context and a narrow practice set;
these tools do not make it a general coding agent. Larger instruction-trained
weights can be loaded with `--model-dir`. The CLI never silently swaps weights.

`/ask` keeps complete source lines within the model's token budget and reports
omissions. `/draft` requires the entire existing file and request to fit: it refuses
to make a complete replacement from a truncated source. Incomplete/multiple code
blocks and Python/JavaScript syntax errors are rejected. Other file types are
explicitly marked as unchecked. Syntax correctness does not prove behavior.

`/do` sends the complete existing function, reference lines that fit, and the
previous candidate with its syntax/test failure on retries. Required source and
failure context are never silently truncated; an oversized request stops before
staging. This loop supports one Python or JavaScript function, not arbitrary
multi-file projects. A larger model can retain several recent chat exchanges
within its context. The 256-token release still retains only one.

`/svg` needs weights trained for SVG. The old website model usually cannot do it.
Open the printed `.kani/previews/ID.svg` path to view the generated image, then
review and `/apply ID` to save the target. Browser chat also renders supported
SVG answers and offers a download: try `/svg a small blue circle`.
Only bounded SVG shapes, groups, paths and text are supported. Scripts, embedded
HTML, CSS, linked assets and external fonts are rejected. There is no hidden
image-template fallback when generation fails. These are vector illustrations,
not photographs. This feature is independent of the screenshot vision prototype.

Proposals persist in `.kani/proposals/`; originals go in `.kani/backups/` on apply.
Add `.kani/` to other projects' `.gitignore` because proposals contain source code.
Apply refuses if the original changed. File tools and `/draft` do not execute generated
source. `/do --test "COMMAND"` explicitly runs your test command against a temporary
copy on macOS, with network access blocked and writes limited to that copy. It
requires the requested tests to run and pass before staging. No tests are run when
`--test` is omitted. The sandbox does not restrict all filesystem reads; do not
interpret it as isolation from every local resource. Workspace tools stay inside the
selected root, reject symlinks, skip common credential paths and generated folders,
and cap file size, search output and total scanning. Git inventories honor ignore
rules; the fallback directory scan uses standard exclusions and says so.

Text chat and file tools need only Node.js 18+. Python syntax checks use an available
`python3` (override with `--python`); if unavailable, the draft is marked unchecked.
`/vision` additionally needs Python, NumPy, Pillow and a vision checkpoint. See
[the vision research notes](../VISION.md). It predicts regions only; it cannot read
screenshot text, explain code in an image, or operate the computer.

Benchmark the loaded JavaScript engine without running generated code:

```sh
node cli/benchmark.mjs
node cli/benchmark.mjs /absolute/path/to/export
node --test cli/test/*.test.mjs
```

The benchmark measures a fixed prompt prefill plus 63 cached decode steps. It is
not a reasoning or coding-quality score. Terminal phase labels describe the work
being performed; no hidden chain of thought is fabricated or displayed.

Percentage word problems use the labeled local calculator: `30% tip of 100`,
`20% off $1,000`, and `8.25% tax on $80` include the arithmetic and resulting total.
This is a bounded English grammar, not a general word-problem solver.

`/web` (or `search the web for QUERY`) is explicitly online. Only the query or
article title is sent to Wikipedia; workspace files and chat history are not sent.
Results are attributed excerpts, not a model-written summary. The first provider
is English Wikipedia, not a general search engine. Rate limits and network
failures are surfaced; Kani never substitutes an invented article. Retrieved
pages are not automatically added to training. See [CAPABILITIES.md](../CAPABILITIES.md)
for the separate, attributed article-training experiment and measured limits.
