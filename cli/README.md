# Kani terminal workspace tools

**Cide preview:** `kani --build kani-cide` lets the trained Kani router hand
coding requests to a separate coding checkpoint. `/cide TASK` asks it
directly; `/model cide` stays on Cide. `/do`, `/draft`, and `/ask` use Cide
when either preview build is selected. See [CIDE.md](../CIDE.md) for training,
evaluation, and limits. The `cide` build is Cide 4.5 (206M parameters, coding replies up to
4,096 tokens; better Python than Cide 4.3, and it writes Kel (`.lus`), but Kel programs that compile are
often logically wrong, so check them; it follows the chat, so a new ask after a page gets a new page and simple edits
return the changed page, answers plain asks like "login html" with a full page and writes a
numbered plan before big pages); `--build cide-4-3` keeps Cide 4.3, `--build cide-4-2` Cide 4.2, `--build cide-4-1` Cide 4.1,
`--build cide-3` Cide 3 and `--build cide-2` the smaller Cide 2.

Default model: **Kani Otter 4.12** (206M parameters, the same int8 weights as the website's default), the `latest` build in `kani-builds.json`. `/model` lists every build; `/model otter-4-9` goes back to Otter 4.9, `/model otter-4-8` to Otter 4.8, `/model otter-4-2` to the earlier 88M Otter 4.2, and `/model seal-3` or `--build seal-3` switches to the smaller, quicker Kani Seal 3 that the website also offers.

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
| `draw the ocean at sunset with two sailboats` or `/draw ...` | Hand the request to Kani Painter, a separate drawing model; save its Kani Paint program and the rendered PNG in `.kani/drawings/` |
| `/svg icon.svg a small blue circle` | Generate SVG from the loaded model, validate it, and save a preview plus a proposed file |
| `/diff ID` | View a saved proposal |
| `/apply ID` | Write that proposal after review; keep a backup of an existing file |
| `/discard ID` | Delete a proposal without changing the target file |
| `/vision screenshot.png` | Predict a coarse UI region grid with the separate experimental vision model |
| any fact question, e.g. `when was Albert Einstein born?` | With a model trained to decide (Kani v6+): Kani searches by itself, reads the top results and answers with numbered sources |
| `/web percentages` or `search the web for ...` | Search online through the LangSearch Worker, with DuckDuckGo/Wikipedia fallbacks; shows sources |
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
The default chat build in `kani-builds.json` is Kani Otter 4.12: 206M parameters
and 1024 tokens of context, the website's default too. These tools
do not make it a general coding agent. Larger instruction-trained
weights can be loaded with `--model-dir`.

`/cide` and learned Cide handoffs now load **Cide 2**, matching the website's
coding weights. `kani --build cide-v1` still selects the legacy expert.
Cide 2 passed 8/9 basic coding checks and 23/24 reserved CRUD checks, but
0/5 broader development checks. A backend-generation probe produced invalid
JavaScript; complete backends and SaaS apps remain unreliable.

`/ask` keeps complete source lines within the model's token budget and reports
omissions. `/draft` requires the entire existing file and request to fit: it refuses
to make a complete replacement from a truncated source. Incomplete/multiple code
blocks and Python/JavaScript syntax errors are rejected. Other file types are
explicitly marked as unchecked. Syntax correctness does not prove behavior.

`/do` sends the complete existing function, reference lines that fit, and the
previous candidate with its syntax/test failure on retries. Required source and
failure context are never silently truncated; an oversized request stops before
staging. This loop supports one Python or JavaScript function, not arbitrary
multi-file projects. The 1024-token model retains several recent chat exchanges.

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

Kani Painter (`training-runs/kani-painter-web`, override with `--painter-dir`) writes a
program in Kani Paint, a small drawing subset of Python. `python3 -m kelus.paint` renders
it with a checked interpreter: the program is parsed and never run as Python. It knows 19
subjects (people, cats, dogs, cars, flowers, birds, a castle, houses, trees, sailboats and
more) in nine kinds of scene. A request naming none of them gets that list instead of a
guess (the list is `kelus/paint_subjects.json`). Wording far from its practice can still
miss parts of the request. Rendering needs Pillow.

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
Casual arithmetic such as `whats 1+1 heh` and `what is fourty five plus five`
also uses this explicitly labeled helper. Single operations on English integer
operands up to 999 are supported. This parser change does not improve the
neural model's arithmetic or understanding.

`/web` (or `search the web for QUERY`) is explicitly online. Only the query or
article title is sent to Wikipedia; workspace files and chat history are not sent.
Results are attributed excerpts, not a model-written summary. The first provider
is English Wikipedia, not a general search engine. Rate limits and network
failures are surfaced; Kani never substitutes an invented article. Retrieved
pages are not automatically added to training. See [CAPABILITIES.md](../CAPABILITIES.md)
for the separate, attributed article-training experiment and measured limits.

## Switch Kani builds

Test the completed 205.6M-parameter Big Kani v2 chat checkpoint locally:

```sh
kani --build big-v2
kani --build big-v2 --max-tokens 192 "Write a paragraph about why cats like playing."
```

This separate experimental build uses the step-9000 best checkpoint, exported
with int8 matrix quantization for the CLI. It does not change the default chat
model or website release. Its validation loss is not a general capability score.

`kani` uses the latest Kani build, which draws with Kani Painter. `kani --build pixels`
adds Kani Pixels v2 for subjects Kani Painter does not know: an experimental 64×64
diffusion model for 174 photo subjects. It needs the repository's `.venv` (MLX,
NumPy, Pillow) and `training-runs/kani-pixels-v2`; anything else goes to Kani Sketch.

In terminal chat, `/model` lists builds and `/model kani-v3` switches to an older
build with a fresh conversation. `/model latest` returns to latest Kani + Pixels.
For a single invocation: `kani --build b1-v3 "hello"`.

Available builds are declared in `kani-builds.json`. Older builds use their own
chat weights and the original drawing-code Painter. Their browser-format chat
weights are exported from local checkpoints on first selection. No pretrained
external weights are downloaded. A missing checkpoint produces an error.

Natural requests such as `can you search up percents` also trigger online search.
The CLI uses the configured LangSearch Worker without a local API key. An optional
`LANGSEARCH_API_KEY` uses LangSearch directly instead. Restart an open CLI after updates.

## Credits

Kani Otter 4.12 was trained in part on SODA by Hyunwoo Kim et al. (2023), https://huggingface.co/datasets/allenai/soda, licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The dialogues were filtered and reformatted for training.

Cide 4.5 was trained in part on the StarCoder2 Self-Alignment Dataset by BigCode (https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) and OpenCoder's algorithmic corpus (https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus), both under the Open Data Commons Attribution License (https://opendatacommons.org/licenses/by/1-0/), and on Magicoder OSS-Instruct (https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K, MIT) and Magicoder Evol-Instruct (https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K, Apache-2.0). The rows were filtered and reformatted for training.
