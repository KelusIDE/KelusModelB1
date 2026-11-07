# Kani for VS Code

Version 0.17.0 makes **Cide 4.5** (206M parameters) the coding specialist: better Python than Cide 4.3, and it writes Kel
(`.lus`), but Kel programs that compile are often logically wrong, so check them. It does one development task and one
website worse than Cide 4.3 in our tests. Cide 4.3 stays in the build picker.

Version 0.15.0 makes **Cide 4.3** (206M parameters) the coding specialist: it follows the chat, so a new request after
a page gets a new page and simple edits ("make the text bigger", "add a footer") return the changed page. Color and
title edits are still hit and miss. Cide 4.2 stays in the build picker.

Version 0.14.0 makes **Cide 4.2** (206M parameters) the coding specialist: plain asks such as "Code a basic login
html" or "hello world html" now get a complete working page. Cide 4.1 stays in the build picker.

Version 0.13.0 makes **Cide 4.1** (206M parameters) the coding specialist: it writes a numbered plan before big
pages and builds them part by part. Cide 3 stays in the build picker.

Version 0.16.0 makes **Kani Otter 4.12** (206M parameters, the website's default) the `latest` build.
Otter 4.9 stays in the build picker.

Version 0.11.0 makes **Cide 3** (206M parameters, replies up to 4,096 tokens) the coding specialist; Cide 2 stays in the
build picker.

Version 0.10.0 makes **Kani Otter 4.9** (206M parameters, the website's default) the `latest` build.
Otter 4.8, Otter 4.2 and Kani Seal 3 stay in the build picker.

Version 0.7.0 adds **Kani + Cide · experimental handoff** and **Cide · coding
specialist** to the build picker when their trained checkpoints are installed.
Kani learns to delegate code requests; the extension labels Cide's response.
`/cide TASK` asks Cide directly. See [CIDE.md](../CIDE.md) for training and limits.

Chat with Kani Otter 4.12, the language model from the
[KelusModelB1](https://github.com/KelusIDE/KelusModelB1) project. It was trained
from random weights and runs on your computer through the repository's own
Python engine, with no API key or network connection.

- Click the crab in the activity bar, or run **Kani: Open in New Tab**.
- Code blocks are syntax-colored and have **Copy** and **Insert**. Insert replaces
  the selection in the last editor you used, as one undo step. Inserted code is
  never saved or run.
- Calculator and clock answers come from labeled local helpers, not the model.
- Ask it to draw ("draw the ocean at sunset with two sailboats" or `/draw ...`) and the
  request goes to **Kani Painter**, a separate drawing model. The picture appears in the
  chat with its program below it. The program is rendered by a checked interpreter and
  never run as Python. Drawing needs Pillow for the configured Python. It knows 19 subjects
  (people, cats, dogs, cars, flowers, birds, a castle, houses, trees, sailboats and more);
  for anything else it says so and lists them instead of guessing.
- `/web QUERY` (or "search the web for ...", "look up ...") searches online through the LangSearch Worker, with DuckDuckGo/Wikipedia fallbacks. Only the query is
  sent; results are labeled as retrieved, not generated. `/web read TITLE` reads a Wikipedia article.
- A model trained to decide when to search (Kani v6 and later) replies `SEARCH: ...` to fact
  questions. The panel never shows that: it runs the search, gives Kani the top results, and shows
  Kani's answer with numbered sources.
- Messages about suicide or self-harm always get a fixed crisis-line reply.

## Requirements

- A checkout of KelusModelB1 with the chat checkpoint `checkpoints/kani-otter-4-12-release.npz`
  (the model live on kani.kelus.org). Drawing uses `checkpoints/kani-painter-v4b.npz` when present.
- Python 3.10+ with NumPy.

Open the repository folder in VS Code, or set **Kani: Repository Path**.
Set **Kani: Python Path** if `python3` is not the interpreter with NumPy.

The model is small (206M parameters, 1024 tokens of context). It practiced everyday chat,
support on hard days, deciding when to search the web, coding concepts, and short Python
and JavaScript snippets. It looks facts up rather than answering them from memory. Review generated code before using it.

## Attachments (0.2.0)

Use **＋ Attach**, drop files onto the composer, or paste an image. Up to four
files: UTF-8 text/code/HTML/JSON/CSV/SVG (128 KB each) and PNG/JPEG/WebP
(8 MB each, at most 24 megapixels). PDF/Office/archive files must be exported
as text first. Attached HTML is source text; it is never run or loaded as a page.

Send an empty prompt or “analyze these files” for labeled local inspection. Ask
a specific question to send relevant complete source lines to the loaded text
model. The reply lists exactly which lines were included; a too-large question
or source line is refused, not silently truncated. Attachments remain selected
for follow-up questions until removed or a new chat starts.

Image inspection reports dimensions and a sampled average color with a preview.
It does not read screenshot text or recognize objects. The separate experimental
region classifier remains available through the CLI's `/vision` command.

The panel reads shared attachment/tokenizer modules from the configured Kani
repository; keep that repository and this extension updated together.

## Kani build picker

Use the **Kani build** dropdown beside the composer to switch between latest
Kani Otter 4.12, Otter 4.11, Otter 4.9, Otter 4.8, Otter 4.2, Kani Seal 3, Kani B2 v3, and original B1 v3. Switching starts a fresh conversation
and changes the actual chat checkpoint. Selection persists in `kani.build`.

Every build draws with Kani Painter. The `pixels` build adds Kani Pixels v2
(an experimental 64×64 diffusion model for 174 photo subjects; it runs with the
repository's `.venv`, which has MLX) for subjects Kani Painter does not know, before Kani Sketch.
There is no separate image-model selector. Local checkpoints and the repository's
`kani-builds.json` are required; only installed builds appear in the dropdown.

Search also recognizes `can you search up percents`. No local LangSearch key is
required when using the configured Worker. Reload VS Code after updating shared search code.

## Credits

Kani Otter 4.12 was trained in part on SODA by Hyunwoo Kim et al. (2023), https://huggingface.co/datasets/allenai/soda, licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. The dialogues were filtered and reformatted for training.

Cide 4.5 was trained in part on the StarCoder2 Self-Alignment Dataset by BigCode (https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) and OpenCoder's algorithmic corpus (https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus), both under the Open Data Commons Attribution License (https://opendatacommons.org/licenses/by/1-0/), and on Magicoder OSS-Instruct (https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K, MIT) and Magicoder Evol-Instruct (https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K, Apache-2.0). The rows were filtered and reformatted for training.
