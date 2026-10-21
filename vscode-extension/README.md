# Kani Seal 3 for VS Code

Chat with Kani Seal 3, the small language model from the
[Kani Seal 3](https://github.com/KelusIDE/KelusModelB1) project. It was trained
from random weights and runs on your computer through the repository's own
Python engine, with no API key or network connection.

- Click the crab in the activity bar, or run **Kani: Open in New Tab**.
- Code blocks are syntax-colored and have **Copy** and **Insert**. Insert replaces
  the selection in the last editor you used, as one undo step. Inserted code is
  never saved or run.
- Calculator and clock answers come from labeled local helpers, not the model.

## Requirements

- A checkout of KelusModelB1 with a trained checkpoint (`checkpoints/b1-v3.best.npz`).
- Python 3.10+ with NumPy.

Open the repository folder in VS Code, or set **Kani: Repository Path**.
Set **Kani: Python Path** if `python3` is not the interpreter with NumPy.

The model is narrow: it knows practiced chat replies, 20 coding concepts, and
short Python and JavaScript snippets. Review generated code before using it.

## Attachments (0.2.0)

Public name: **Kani Seal 3**. Development codename: **Kani B2:10F**.

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
