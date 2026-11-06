"""Write the notebooks from their tested scripts (kaggle/kani_pixels_v3.py, kaggle/kani_advanced.py, kaggle/kani_tpu.py), so they never drift.

    python3 kaggle/make_notebook.py   # kani-pixels-v3.ipynb, kani-pixels-v3-colab.ipynb, kani-advanced.ipynb, kani-advanced-multilingual.ipynb, kani-gpt3-step1.ipynb, kani-gpt3-tpu.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = (HERE / 'kani_pixels_v3.py').read_text(encoding='utf-8')


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(keepends=True)}


ABOUT = '''
An image generator for Kani, trained **from scratch** on free GPUs. No pretrained weights.

- **Model:** a diffusion U-Net (about 78M parameters) with the EDM training recipe and sampler, and classifier-free guidance
- **Subjects:** about 4,000 from Open Images (people, animals, vehicles, buildings, food, places, objects, and styles such as
  drawing, sketch, cartoon and painting), plus anime faces, cat and dog faces, fruits, birds, cars, landscapes and flowers. A picture
  carries several labels, so a prompt can combine them: "a drawing of a cat"
- **Images:** 64×64. Training stops itself before the session limit and continues in the next session.
'''

SOURCES = '''
# Where each dataset comes from (kaggle.com/datasets/<owner>/<name>)
SOURCES = {
    'animefacedataset': 'splcher/animefacedataset',
    'animal-faces': 'andrewmvd/animal-faces',
    'fruits': 'moltean/fruits',
    'celeba-dataset': 'jessicali9530/celeba-dataset',
    'bird-species-classification-220-categories': 'kedarsai/bird-species-classification-220-categories',
    'stanford-car-dataset-by-classes-folder': 'jutrera/stanford-car-dataset-by-classes-folder',
    'landscape-pictures': 'arnaud58/landscape-pictures',
    'flowers-recognition': 'alxmamaev/flowers-recognition',
}
'''

DOWNLOAD = '''
# Downloads every dataset in SOURCES into dataset_root/<name> (public files, no login or Add Input needed).
# A failure prints a "!" line; the next step then stops before any training.
import shutil
import urllib.request
import zipfile
from pathlib import Path
root = Path(SETTINGS['dataset_root'])
root.mkdir(parents=True, exist_ok=True)
saved = list(Path(SETTINGS.get('input_root', '/kaggle/input')).rglob('prepared/manifest.json'))
if saved:
    print('Pictures saved by an earlier session are attached, so nothing to download:', saved[0].parent)
for name in [] if saved else DATASETS:
    target, archive = root / name, root / f'{name}.zip'
    if target.exists():
        continue
    try:
        request = urllib.request.Request(f'https://www.kaggle.com/api/v1/datasets/download/{SOURCES[name]}',
                                         headers={'User-Agent': 'kani-pixels-v3'})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as out:
            shutil.copyfileobj(response, out, 1 << 20)
        with zipfile.ZipFile(archive) as files:
            files.extractall(target)
        print(f'{SOURCES[name]}: {sum(p.is_file() for p in target.rglob("*")):,} files')
    except Exception as error:
        print(f'! {SOURCES[name]}: {error}')
        shutil.rmtree(target, ignore_errors=True)
    finally:
        archive.unlink(missing_ok=True)
'''

DATASETS = '''
# DATASETS: dataset folder name -> how its images are labelled:
#   a fixed label ('anime face'), 'folder' (the image's folder: cat / dog / wild),
#   or 'first-word' (first word of the folder: 'Apple Braeburn 1' -> apple).
DATASETS = {
    'animefacedataset': 'anime face',
    'animal-faces': 'folder',
    'fruits': 'first-word',
    'celeba-dataset': 'human face',
    'bird-species-classification-220-categories': 'bird',
    'stanford-car-dataset-by-classes-folder': 'car',
    'landscape-pictures': 'landscape',
    'flowers-recognition': 'folder',  # daisy, dandelion, rose, sunflower, tulip
}
'''

SETTINGS_TAIL = '''
    'size': 64,             # 32, 64 or 128 pixels; 128 needs about 4x the training for the same quality
    'channels': 128,        # model width; 192 is bigger and slower
    'max_per_class': 20000,
    'class_caps': {'fruits': 500},  # at most this many per fruit: uncapped, fruit would be 60% of training and cats 1%
    'exclude': ['test-multiple_fruits'],  # mixed-fruit pictures; they would become a subject called "test"
    'guidance': 2.0,        # higher follows the subject more strictly, with less variety
}

import json
json.dump(SETTINGS, open('settings.json', 'w'), indent=1)
print(json.dumps(SETTINGS, indent=1))
'''

GPU_CHECK = '''
import torch
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if not gpus:  # stops a run on a laptop or a CPU session, which would take days
    raise SystemExit('No NVIDIA GPU: {hint}')
print('PyTorch', torch.__version__, '| GPUs:', gpus)
'''

DATA = '''
!python kani_pixels_v3.py prepare --settings settings.json
from pathlib import Path
if not Path(SETTINGS.get('data_dir', '/tmp/kani-pixels-data'), 'manifest.json').exists():  # "!" commands do not stop a notebook
    raise RuntimeError('No prepared images, so no training: see the "!" lines above.')
!python kani_pixels_v3.py real --settings settings.json --out real.png
from IPython.display import Image, display
display(Image('real.png'))
'''

RESULTS = '''
from pathlib import Path
grids = sorted(Path(SETTINGS['run_dir'] if 'run_dir' in SETTINGS else 'kani-pixels-v3', 'samples').glob('*.png'))
print(f'{len(grids)} grids; the latest, one row per subject:', grids[-1])
display(Image(str(grids[-1])))
'''

TRY = '''
# 16 images for a prompt: words naming subjects in kani-pixels-v3.json; several words combine
prompt = 'a drawing of a cat'
!python kani_pixels_v3.py sample --settings settings.json --prompt "{prompt}" --count 16 --out try.png
display(Image('try.png'))
'''

TRAIN_NOTE = '''
## Train

One log line every 200 steps (`loss` should fall, then flatten), a checkpoint and a sample grid every
30 minutes. Stops by itself after `hours`, then saves the model for Kani.
'''

kaggle = [
    markdown('# Kani Pixels v3\n' + ABOUT + '''
## Setup (once)

1. **Settings → Accelerator → GPU T4 x2.** Kaggle needs a phone-verified account for GPUs.
2. **Settings → Internet → On.** The notebook downloads everything itself: about 1.2M Open Images pictures (the first session spends
   up to 3 hours on this) and these datasets. If they cannot be reached, it stops before training and prints a `!` line.
3. **Add Input → Your Work → Notebooks → pixelsv3** (your earlier run). This keeps what it already learned. Without it, training
   starts from nothing.

   | Dataset | Subjects it gives |
   |---|---|
   | `splcher/animefacedataset` | anime face |
   | `andrewmvd/animal-faces` | cat, dog, wild |
   | `moltean/fruits` | apple, banana, … (about 70) |
   | `jessicali9530/celeba-dataset` | human face |
   | `kedarsai/bird-species-classification-220-categories` | bird |
   | `jutrera/stanford-car-dataset-by-classes-folder` | car |
   | `arnaud58/landscape-pictures` | landscape |
   | `alxmamaev/flowers-recognition` | daisy, dandelion, rose, sunflower, tulip |

4. **Save Version → Save & Run All (Commit).** It runs in the background for about 11 hours, so you can close the tab.

## Train more in a later session

Open the notebook, **Add Input → Your Work → Notebooks →** this notebook (its latest version), then
**Save & Run All** again. It finds `kani-pixels-v3/checkpoint.pt` and the saved pictures, and continues where it stopped, so later sessions skip
the download. Keep the settings the same. Each session adds about 10 hours of training, and quality keeps improving slowly: expect rough shapes
for a long time with this many subjects.

## Output (the version's Output tab)

- `kani-pixels-v3/kani-pixels-v3.npz` and `kani-pixels-v3.json`: the model for Kani
- `kani-pixels-v3/samples/*.png`: a grid every 30 minutes, one row per prompt in `showcase`
- `kani-pixels-v3/checkpoint.pt`: for continuing training
- `kani-pixels-v3/prepared/`: the pictures, small, so later sessions skip the download

Open Images: the pictures are listed as CC BY 2.0 and the labels are CC BY 4.0; credit them when you ship this model.
Sexual and undressed pictures and gore are left out by their labels, which lowers the chance of such pictures but does not prove there are none.
'''),
    code('import time\n' + SOURCES + DATASETS + '''SETTINGS = {
    'datasets': DATASETS,
    'dataset_root': '/tmp/kani-datasets',
    'open_images': {'per_label': 1000, 'max_images': 1200000, 'min_label_images': 100},
    'prepare_threads': 64,
    'prepare_hours': 3.0,   # stop downloading after this long and train on what arrived
    'session_started': time.time(),
    'hours': 11.0,          # from the start of this session, downloads included (Kaggle stops sessions at 12 hours)
    'showcase': ['person', 'human face', 'anime face', 'cat', 'dog', 'bird', 'horse', 'flower', 'tree', 'mountain', 'beach', 'house', 'car', 'airplane',
                 'food', 'painting', 'drawing + cat', 'cartoon + dog', 'sketch + person', 'painting + mountain'],  # the rows of the sample grids
    'batch': 128,           # across both GPUs''' + SETTINGS_TAIL),
    code(GPU_CHECK.replace('{hint}', 'set Settings → Accelerator → GPU T4 x2')),
    markdown('The training script (the same file as `kaggle/kani_pixels_v3.py` in the KelusModelB1 repository).'),
    code('%%writefile kani_pixels_v3.py\n' + SCRIPT),
    markdown('## Data\nThe first session downloads the pictures, centre-crops and resizes each one (up to 3 hours), and saves them small in the output; '
             'later sessions reload them in minutes. Then it shows real examples, one row per prompt in `showcase`.'),
    code(DOWNLOAD),
    code(DATA),
    markdown(TRAIN_NOTE),
    code('''
gpu_count = max(1, torch.cuda.device_count())
!torchrun --nnodes=1 --nproc_per_node={gpu_count} --master_addr=127.0.0.1 --master_port=29500 kani_pixels_v3.py train --settings settings.json
'''),
    markdown('## Results'),
    code(RESULTS),
    code(TRY),
    markdown('''
## Bring it to Kani

Download `kani-pixels-v3/kani-pixels-v3.npz` and `kani-pixels-v3/kani-pixels-v3.json` from the Output tab,
put both in `KelusModelB1/checkpoints/kani-pixels-v3/`, and Kani can be wired to draw with it.
'''),
]

colab = [
    markdown('# Kani Pixels v3 (Google Colab)\n' + ABOUT + '''
The same model and script as the Kaggle notebook; checkpoints from either one continue in the other.

## Setup (once)

1. **Runtime → Change runtime type → T4 GPU.**
2. A Kaggle API token, used only to download the datasets: on kaggle.com go to **Settings → API → Generate New Token**.
   In Colab, open **Secrets** (the key icon on the left), add a secret named `KAGGLE_API_TOKEN` with the token
   as its value, and switch on notebook access. (An older `kaggle.json` works too: add secrets `KAGGLE_USERNAME`
   and `KAGGLE_KEY` instead.)
3. **Runtime → Run all**, and allow Google Drive access when asked. Progress is saved to
   `MyDrive/KaniPixelsV3`, so nothing is lost when Colab disconnects.

## When Colab disconnects

Free Colab sessions end after at most 12 hours, sooner if the tab sits idle. Just **Runtime → Run all** again:
the prepared images come back from Drive in a minute and training continues from the last checkpoint
(at most 30 minutes are lost).
'''),
    code(SOURCES + DATASETS + '''DRIVE = '/content/drive/MyDrive/KaniPixelsV3'
SETTINGS = {
    'datasets': DATASETS,
    'input_root': '/content/input',
    'data_dir': '/content/kani-pixels-data',
    'run_dir': DRIVE + '/kani-pixels-v3',
    'hours': 11.0,          # training time this session (free Colab sessions end by 12 hours)
    'batch': 64,            # one T4''' + SETTINGS_TAIL),
    code(GPU_CHECK.replace('{hint}', 'Runtime → Change runtime type → T4 GPU')),
    code('''
from google.colab import drive
drive.mount('/content/drive')
'''),
    markdown('The training script (the same file as `kaggle/kani_pixels_v3.py` in the KelusModelB1 repository).'),
    code('%%writefile kani_pixels_v3.py\n' + SCRIPT),
    markdown('''
## Data

The first time, this downloads the datasets from Kaggle, centre-crops and resizes every image (about
15 minutes), and keeps the result in Drive. Later sessions copy it back from Drive instead.
'''),
    code('''
import os, shutil
from pathlib import Path
import kani_pixels_v3

cache, data = Path(DRIVE) / 'data', Path(SETTINGS['data_dir'])
files = ('images.npy', 'labels.npy', 'manifest.json')
key = kani_pixels_v3.prepared_key(kani_pixels_v3.load_settings('settings.json'))
if all((cache / name).exists() for name in files) and json.loads((cache / 'manifest.json').read_text()).get('key') == key:
    data.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy(cache / name, data / name)
    print('prepared images copied from Drive')
else:
    from google.colab import userdata
    for name in ('KAGGLE_API_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):
        try:
            os.environ[name] = userdata.get(name)
        except Exception:  # that secret is not set, or notebook access is off
            pass
    if not (os.environ.get('KAGGLE_API_TOKEN') or os.environ.get('KAGGLE_KEY')):
        raise SystemExit('Add the KAGGLE_API_TOKEN secret (see Setup step 2), then run this cell again.')
    !pip install -q --upgrade kaggle
    for name in DATASETS:
        slug, target = SOURCES[name], Path(SETTINGS['input_root']) / name
        if not target.exists():
            print('downloading', slug)
            !kaggle datasets download -d {slug} -p {target} --unzip -q
    !python kani_pixels_v3.py prepare --settings settings.json
    cache.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy(data / name, cache / name)
    print('prepared images saved to Drive for next time')
'''),
    code(DATA.replace('!python kani_pixels_v3.py prepare --settings settings.json\n', '')),
    markdown(TRAIN_NOTE),
    code('!python kani_pixels_v3.py train --settings settings.json'),
    markdown('## Results'),
    code(RESULTS),
    code(TRY),
    markdown('''
## Bring it to Kani

Download `kani-pixels-v3.npz` and `kani-pixels-v3.json` from `MyDrive/KaniPixelsV3/kani-pixels-v3`,
put both in `KelusModelB1/checkpoints/kani-pixels-v3/`, and Kani can be wired to draw with it.
'''),
]


ADVANCED = (HERE / 'kani_advanced.py').read_text(encoding='utf-8')
TOKENIZER = json.dumps(json.loads((HERE / 'kani_tokenizer.json').read_text(encoding='utf-8')), separators=(',', ':'))

advanced = [
    markdown('''
# Kani Advanced

Kani's own transformer, made bigger and pretrained **from scratch** on Kaggle's free GPUs, with no pretrained weights.

- **Same architecture and tokenizer as Kani** (rotary attention, tied embeddings, Kani's 4,096-token BPE), so the
  result loads straight into Kani's terminal app and VS Code extension.
- **About 205M parameters** (big Kani v1 has 88M): width 1024, 16 layers, 1,024-token context.
- **Better text:** FineWeb-Edu (educational web pages) and Cosmopedia v2 (textbook-style writing), both public.
  Each session downloads new text; the goal is 4 billion tokens (big Kani v1 saw 164 million).
- **What to expect:** roughly GPT-2-class writing and general knowledge after the full run. That is a big step
  up for Kani, and still far from GPT-5, which is trained on vastly more text with far more compute.

## Setup (once)

1. **Settings → Accelerator → GPU T4 x2.**
2. **Settings → Internet → On** (Session options in the right-hand panel). The text is downloaded from Hugging Face.
3. **Save Version → Save & Run All (Commit).** About 10 minutes of downloading and tokenizing, then about 10 hours of
   training. You can close the tab.

## Every next session

**Add Input → Your Work → Notebooks →** this notebook (its latest version), then **Save & Run All** again. It
continues from `kani-advanced/checkpoint.pt` on new text. About 4–5 sessions reach the 4-billion-token goal.

`continue_from_previous` is True, so a session with that input missing stops within a minute with an error
instead of quietly training from zero. For the very first session of a new run, set it to False.

## Output (the version's Output tab)

- `kani-advanced/kani-advanced.npz`: the model, in Kani's checkpoint format
- `kani-advanced/samples.json`: what it writes after a few prompts, to see it improve
- `kani-advanced/checkpoint.pt` and `progress.json`: for continuing
'''),
    code('''
# Model size. Keep these the same in every session of one run.
#   width 768,  layers 12 ->  88M parameters (big Kani v1's size)
#   width 1024, layers 16 -> 205M (default)
#   width 1024, layers 24 -> 306M (largest Kani supports; needs ~1.5x the sessions)
SETTINGS = {
    'width': 1024,
    'layers': 16,
    'heads': 16,
    'context': 1024,
    'target_tokens': 4_000_000_000,  # the whole run; the learning rate winds down over the last 20%
    'hours': 10.0,                   # training time per session (Kaggle stops sessions at 12 hours)
    'micro_batch': 4,                # sequences per GPU at once; lower it if you see "CUDA out of memory"
    # True after the first session: stop at once (instead of training from zero for 10 hours)
    # when this notebook's earlier output is not attached as an input.
    'continue_from_previous': True,
}

import json
json.dump(SETTINGS, open('settings.json', 'w'), indent=1)
print(json.dumps(SETTINGS, indent=1))
'''),
    code('''
import torch, urllib.request
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('PyTorch', torch.__version__, '| GPUs:', gpus or 'none: set Settings → Accelerator → GPU T4 x2')
try:
    urllib.request.urlopen('https://huggingface.co', timeout=15)
    print('Internet: on')
except OSError as error:
    raise SystemExit(f'Cannot reach huggingface.co ({error}). Is Settings → Internet on (Session options panel)?')
'''),
    markdown("Kani's tokenizer (the same file as `data/big-v1/tokenizer.json` in the KelusModelB1 repository) and the training script (`kaggle/kani_advanced.py`)."),
    code('%%writefile kani_tokenizer.json\n' + TOKENIZER),
    code('%%writefile kani_advanced.py\n' + ADVANCED),
    markdown("## This session's text\nDownloads one new FineWeb-Edu file and one Cosmopedia file and turns them into Kani's tokens (about 10 minutes)."),
    code('''
!python kani_advanced.py prepare --settings settings.json
# A failed prepare (for example no earlier checkpoint attached) stops the notebook here, not with errors further down.
import kani_advanced
from pathlib import Path
if not (Path(kani_advanced.load_settings('settings.json')['data_dir']) / 'manifest.json').exists():
    raise RuntimeError('Stopped before training: read the message just above.')
'''),
    markdown('''
## Train

A log line every 20 steps. `loss` and `validation_loss` should fall over the session and across sessions.
The first line shows `gpu_memory_gb`; if a run stops with "CUDA out of memory", set `micro_batch` to 2.
'''),
    code('''
gpu_count = max(1, torch.cuda.device_count())
!torchrun --nnodes=1 --nproc_per_node={gpu_count} --master_addr=127.0.0.1 --master_port=29500 kani_advanced.py train --settings settings.json
'''),
    markdown('## How it writes now'),
    code('''
progress = json.load(open('kani-advanced/progress.json'))
for session in progress['history']:
    print(session)
for prompt, text in json.load(open('kani-advanced/samples.json')).items():
    print('\\n---', text)
'''),
    markdown('''
## Bring it to Kani

Download `kani-advanced/kani-advanced.npz` from the Output tab and put it in `KelusModelB1/checkpoints/`.
It is a raw-text model: it continues text but does not chat yet. Chat tuning (Kani's conversations, web
search and reading data) and the capability tests then run on the Mac, before it replaces the current Kani.
'''),
]



TOKENIZER_MULTILINGUAL = json.dumps(json.loads((HERE / 'kani_tokenizer_multilingual.json').read_text(encoding='utf-8')), separators=(',', ':'))
MULTILINGUAL_INTRO = markdown('''
# Kani Advanced, multilingual

Kani Advanced continued in nine languages: English, Korean, German, French, Spanish, Portuguese, Japanese, and
Simplified and Traditional Chinese. Still **from scratch**, with no pretrained weights.

- **Kani's multilingual tokenizer** (32,768 tokens, BPE pattern version 3): about 4x fewer tokens than the English one
  for Korean, Japanese and Chinese, and 1.7-1.9x fewer for German, French, Spanish and Portuguese.
- **A continuation, not a restart:** the run began from the English Kani Advanced, moved to this tokenizer by
  `kani_advanced.py retokenize` (Modal does this once; see `kaggle/modal_kani_advanced.py`).
- **Text per session:** 40% English (FineWeb-Edu, Cosmopedia v2) and 7.5% of each other language (FineWeb-2), every
  document checked to be in its language's script. Validation holds out text in every language.

## Every session

**Add Input** with the run's latest checkpoint: this notebook's own latest output, or a dataset with the
`kani-multilingual` folder (checkpoint.pt and progress.json) downloaded from Modal. Then **Save & Run All**.
Settings: **GPU T4 x2** and **Internet on**. `continue_from_previous` is True: with no checkpoint attached it stops
within a minute instead of training from zero.

## Output

- `kani-multilingual/kani-advanced.npz`: the model, in Kani's checkpoint format
- `kani-multilingual/samples.json`: a prompt in each language, to see it improve
- `kani-multilingual/checkpoint.pt` and `progress.json`: for continuing
''')
MULTILINGUAL_SETTINGS = '''
SETTINGS = {
    'run_dir': '/kaggle/working/kani-multilingual',
    'tokenizer': 'kani_tokenizer_multilingual.json',
    'width': 1024,
    'layers': 16,
    'heads': 16,
    'context': 1024,
    'target_tokens': 4_500_000_000,  # the whole run; the learning rate winds down over the last 20%
    'hours': 10.0,                   # training time per session (Kaggle stops sessions at 12 hours)
    'micro_batch': 4,                # sequences per GPU at once; lower it if you see "CUDA out of memory"
    'continue_from_previous': True,  # stop at once when the run's checkpoint is not attached
    'sources': [  # 40% English, 7.5% of each other language; FineWeb-2 documents are kept only in their script
        {'repo': 'HuggingFaceFW/fineweb-edu', 'prefix': 'sample/10BT/', 'per_session': 1, 'share': 0.30},
        {'repo': 'HuggingFaceTB/smollm-corpus', 'prefix': 'cosmopedia-v2/', 'per_session': 1, 'share': 0.10},
    ] + [{'repo': 'HuggingFaceFW/fineweb-2', 'prefix': f'data/{code}/train/', 'validation_file': f'data/{code}/test/000_00000.parquet',
          'per_session': 1, 'share': 0.075, **extra}
         for code, extra in [('kor_Hang', {'filter': 'hangul'}), ('deu_Latn', {}), ('fra_Latn', {}), ('spa_Latn', {}), ('por_Latn', {}),
                             ('jpn_Jpan', {'filter': 'kana'}), ('cmn_Hani', {'filter': 'simplified'}),
                             ('cmn_Hani', {'filter': 'traditional', 'offset': 185})]],
}

import json
json.dump(SETTINGS, open('settings.json', 'w'), indent=1)
print(json.dumps(SETTINGS, indent=1))
'''


def multilingual_cells():
    """The English notebook's cells, with the multilingual introduction, settings, tokenizer and folder."""
    cells = [MULTILINGUAL_INTRO]
    for cell in advanced[1:]:
        text = ''.join(cell['source'])
        if text.lstrip().startswith('# Model size.'):
            text = MULTILINGUAL_SETTINGS
        elif text.startswith('%%writefile kani_tokenizer.json'):
            text = '%%writefile kani_tokenizer_multilingual.json\n' + TOKENIZER_MULTILINGUAL
        else:
            text = (text.replace("Kani's tokenizer (the same file as `data/big-v1/tokenizer.json`",
                                 "Kani's multilingual tokenizer (the same file as `kaggle/kani_tokenizer_multilingual.json`")
                        .replace('Downloads one new FineWeb-Edu file and one Cosmopedia file', 'Downloads one new file per language')
                        .replace('kani-advanced/progress.json', 'kani-multilingual/progress.json')
                        .replace('kani-advanced/samples.json', 'kani-multilingual/samples.json')
                        .replace('`kani-advanced/kani-advanced.npz`', '`kani-multilingual/kani-advanced.npz`'))
        cells.append(code(text) if cell['cell_type'] == 'code' else markdown(text))
    return cells


RUNG1_INTRO = markdown('''
# Kani toward GPT-3, step 1

Kani's goal is to climb GPT-3's sizes (`GPT3_GOAL.md`): step 1 is **GPT-3 Small** (average 43.9 over HellaSwag, LAMBADA,
PIQA, Winogrande, ARC-Easy and ARC-Challenge). This notebook continues **Kani Big v2** (205M, from scratch, no
pretrained weights) from 2.77B tokens toward 10B, on English-heavy text: 65% FineWeb-Edu, 25% Cosmopedia v2,
10% code. Every file is new to it (`offset` skips the files the RunPod session read).

- **No test leaks:** every document that shares a run of 13 words with one of the six test sets is dropped before
  training (`contamination.npy`, made by `kani_advanced.py grams`), as GPT-3 did.
- **One Kaggle session (T4 x2, 10 hours):** about 0.58B tokens. After each, the Mac scores
  `kani-advanced/kani-advanced.npz` with `python -m kelus.standard_eval`.

## First session

1. **Add Input → Datasets →** your dataset with the `kani-advanced` folder (`checkpoint.pt`, `progress.json`,
   `contamination.npy`), uploaded from `~/Downloads/kani-rung1-start`.
2. **Settings → Accelerator → GPU T4 x2**, and **Internet on**.
3. **Save Version → Save & Run All (Commit).** You can close the tab.

## Every next session

**Add Input → Your Work → Notebooks →** this notebook (its latest version), and the TPU notebook's
(`kani-gpt3-tpu.ipynb`) if a TPU session ran, then **Save & Run All**. The run continues from whichever attached
checkpoint has seen the most tokens, so the starting dataset can stay attached. Never run both notebooks at once.
`continue_from_previous` and `decontaminate` are True: without a checkpoint, or without `contamination.npy`, it stops
within a minute instead of training wrongly.

## Output

- `kani-advanced/kani-advanced.npz`: the model, in Kani's checkpoint format (download it for the Mac's tests)
- `kani-advanced/checkpoint.pt`, `progress.json`, `contamination.npy`: for continuing
''')
RUNG1_SETTINGS = '''
SETTINGS = {
    'width': 1024,
    'layers': 16,
    'heads': 16,
    'context': 1024,
    'target_tokens': 10_000_000_000,  # the learning rate stays up until 8B, then winds down to 10B
    'session_tokens': 700_000_000,    # a little more than one session trains on, so no window repeats much
    'hours': 10.0,                    # training time per session (Kaggle stops sessions at 12 hours)
    'micro_batch': 4,                 # sequences per GPU at once; lower it if you see "CUDA out of memory"
    'continue_from_previous': True,   # stop at once when no checkpoint is attached
    'decontaminate': True,            # stop at once when contamination.npy is not attached
    'sources': [  # `offset` starts after the files the RunPod session (session 3) read
        {'repo': 'HuggingFaceFW/fineweb-edu', 'prefix': 'sample/100BT/', 'share': 0.65, 'per_session': 1, 'offset': 5},
        {'repo': 'HuggingFaceTB/smollm-corpus', 'prefix': 'cosmopedia-v2/', 'share': 0.25, 'per_session': 2, 'offset': 18},
        {'repo': 'codeparrot/github-code-clean', 'prefix': 'data/', 'column': 'code', 'share': 0.10, 'per_session': 1, 'offset': 5},
    ],
}

import json
json.dump(SETTINGS, open('settings.json', 'w'), indent=1)
print(json.dumps(SETTINGS, indent=1))
'''


def rung1_cells():
    """The English notebook's cells, with the GPT-3 step 1 introduction and settings."""
    cells = [RUNG1_INTRO]
    for cell in advanced[1:]:
        text = ''.join(cell['source'])
        if text.lstrip().startswith('# Model size.'):
            text = RUNG1_SETTINGS
        else:
            text = text.replace('Downloads one new FineWeb-Edu file and one Cosmopedia file',
                                'Downloads new FineWeb-Edu, Cosmopedia and code files, drops any document that shares 13 words with a test set,')
        cells.append(code(text) if cell['cell_type'] == 'code' else markdown(text))
    return cells


TPU = (HERE / 'kani_tpu.py').read_text(encoding='utf-8')

TPU_INTRO = markdown('''
# Kani toward GPT-3, step 1 (TPU)

The same run as `kani-gpt3-step1.ipynb` (Kani Big v2, 205M, from scratch, toward 10B tokens of English-heavy text
with the test sets filtered out), on Kaggle's **TPU v5e-8** instead of two T4s, trained with JAX by `kani_tpu.py`.
TPU hours are a weekly quota of their own (about 20 hours, beside the GPUs' 30), and a TPU session trains several
times more text than a T4 session.

Both notebooks write the same `kani-advanced/` output, and each continues from whichever attached checkpoint has seen
the most tokens, so sessions can switch between them. Never run both at the same time: Kani would split into two
different models.

## Every session

1. **Add Input**: the latest output of this notebook and of the GPU notebook (**Your Work → Notebooks**), and the
   dataset with the starting `kani-advanced` folder (`checkpoint.pt`, `progress.json`, `contamination.npy`).
   Whatever is attached, the newest checkpoint wins; at least one must be there or it stops within a minute.
2. **Settings → Accelerator → TPU v5e-8**, and **Internet on**.
3. **Save Version → Save & Run All (Commit).** About 8 hours (Kaggle stops TPU sessions at 9). You can close the tab.

The log shows `attached inputs`, `continuing from …`, the feeder reading files, then a status line every 20 steps.
`waiting_for_text_seconds` should stay near 0; if it climbs, the TPU is waiting for the tokenizer. If XLA runs out of
memory at the start, set `tpu_micro_batch` to 4.

## Output

- `kani-advanced/kani-advanced.npz`: the model, in Kani's checkpoint format (download it for the Mac's tests)
- `kani-advanced/checkpoint.pt`, `progress.json`, `contamination.npy`: for continuing, on either notebook
''')


def tpu_cells():
    """The step 1 notebook for a TPU: its settings, tokenizer and data, with kani_tpu.py training."""
    settings = (RUNG1_SETTINGS
                .replace("    'session_tokens': 700_000_000,    # a little more than one session trains on, so no window repeats much\n", '')
                .replace("    'hours': 10.0,                    # training time per session (Kaggle stops sessions at 12 hours)",
                         "    'hours': 8.0,                     # the whole session (Kaggle stops TPU sessions at 9 hours)")
                .replace("    'micro_batch': 4,                 # sequences per GPU at once; lower it if you see \"CUDA out of memory\"",
                         "    'tpu_micro_batch': 8,             # sequences per TPU chip at once; set 4 if XLA runs out of memory\n"
                         "    'micro_batch': 4,                 # the GPU notebook's: the validation windows follow it, so both report the same loss"))
    assert "'tpu_micro_batch'" in settings and "'hours': 8.0" in settings and 'session_tokens' not in settings
    return [
        TPU_INTRO,
        code(settings),
        code('''
import importlib.util, subprocess, sys, urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=15)
    print('Internet: on')
except OSError as error:
    raise SystemExit(f'Cannot reach huggingface.co ({error}). Is Settings → Internet on (Session options panel)?')
# PyTorch (CPU) only reads and writes the checkpoints, which are shared with the GPU notebook.
if importlib.util.find_spec('torch') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
missing = [name for name in ('pyarrow', 'huggingface_hub') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
# JAX is checked in a separate process: a notebook that imported JAX itself would hold the TPU, and the trainer
# (also a separate process) could not use it.
check = subprocess.run([sys.executable, '-c', 'import jax; d = jax.devices(); print("JAX", jax.__version__, "|", len(d), "x", d[0].device_kind, d[0].platform)'],
                       capture_output=True, text=True)
print(check.stdout or check.stderr[-2000:])
if 'tpu' not in check.stdout:
    raise SystemExit('No TPU: set Settings → Accelerator → TPU v5e-8.')
'''),
        markdown("Kani's tokenizer (the same file as `data/big-v1/tokenizer.json` in the KelusModelB1 repository), the GPU "
                 "training script (`kaggle/kani_advanced.py`: the tokenizer, text and checkpoint code) and the TPU trainer (`kaggle/kani_tpu.py`)."),
        code('%%writefile kani_tokenizer.json\n' + TOKENIZER),
        code('%%writefile kani_advanced.py\n' + ADVANCED),
        code('%%writefile kani_tpu.py\n' + TPU),
        markdown('''
## Train

The feeder downloads and tokenizes the text while the TPU trains (the first shard takes a few minutes), then a status
line every 20 steps. `loss` and `validation_loss` should fall over the session and across sessions.
'''),
        code('''
!python kani_tpu.py train --settings settings.json
from pathlib import Path
if not Path('kani-advanced/samples.json').exists():
    raise RuntimeError('Training stopped before the end of the session: read the messages above.')
'''),
        *advanced[-3:],
    ]


def write(name, cells, metadata):
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}, **metadata}}
    (HERE / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {HERE / name}: {len(cells)} cells')


write('kani-pixels-v3.ipynb', kaggle, {'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': True,
                                                  'language': 'python', 'sourceType': 'notebook'}})
write('kani-pixels-v3-colab.ipynb', colab, {'accelerator': 'GPU', 'colab': {'gpuType': 'T4', 'provenance': []}})
write('kani-advanced.ipynb', advanced, {'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True, 'isInternetEnabled': True,
                                                   'language': 'python', 'sourceType': 'notebook'}})
write('kani-advanced-multilingual.ipynb', multilingual_cells(), {'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True,
                                                                         'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'}})
write('kani-gpt3-step1.ipynb', rung1_cells(), {'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True,
                                                          'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'}})
write('kani-gpt3-tpu.ipynb', tpu_cells(), {'kaggle': {'isGpuEnabled': False, 'isInternetEnabled': True, 'language': 'python',
                                                      'sourceType': 'notebook'}})
