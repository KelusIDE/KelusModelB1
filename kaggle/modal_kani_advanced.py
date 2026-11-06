"""Kani Advanced on Modal: the Kaggle training script (kaggle/kani_advanced.py), unchanged, on one L4 GPU.

Modal's free Starter plan gives $30 of compute a month: about three 8.5-hour sessions on an L4, at about $9
each (the GPU about $0.80 an hour, plus the CPU and memory beside it), each roughly one Kaggle session.
Set a budget at modal.com/settings so nothing is billed past the free credit. Everything lives in the Modal
volume "kani-advanced":

    /input/kaggle/kani-advanced/   checkpoint.pt and progress.json from Kaggle (uploaded once, below)
    /run/kani-advanced/            the English run: checkpoint.pt, progress.json, kani-advanced.npz, samples.json
    /run/kani-multilingual/        the nine-language run (English, Korean, German, French, Spanish, Portuguese,
                                   Japanese, Simplified and Traditional Chinese), started from the English run
    /data/, /data-multilingual/    each run's prepared text for the session (train.bin, validation.bin, manifest.json)

A session first prepares its text on a CPU machine (cents), then trains on the GPU, which never waits for
tokenizing. Checkpoints are saved every 30 minutes, so a stopped session loses at most that.

Once:
    pip install modal && modal setup                     # account, then log in through the browser
    modal volume create kani-advanced
    modal volume put kani-advanced ~/Downloads/kani-advanced/checkpoint.pt /input/kaggle/kani-advanced/checkpoint.pt
    modal volume put kani-advanced ~/Downloads/kani-advanced/progress.json /input/kaggle/kani-advanced/progress.json

    modal deploy kaggle/modal_kani_advanced.py           # again after changing this file

Each session runs entirely on Modal's side, started rather than watched, so this computer can sleep or go offline
(a `modal run` that waits for the result cancels the session when its connection drops). Follow it at modal.com/apps:
    python kaggle/modal_kani_advanced.py retokenize       # once: the English run's latest checkpoint becomes the
                                                          # multilingual run's start (kaggle/kani_tokenizer_multilingual.json)
    python kaggle/modal_kani_advanced.py start            # the multilingual run; --run english, --hours 8.5 by default
    python kaggle/modal_kani_advanced.py status           # step, tokens, losses, samples (--run english)
    modal volume get kani-advanced /run/kani-multilingual/kani-advanced.npz checkpoints/   # the model, for kelus

Back to Kaggle: download the run's checkpoint.pt and progress.json, upload them as a Kaggle dataset (in a folder
named after the run) and attach it instead of the notebook's own output.
"""
import json
from pathlib import Path
import shutil
import subprocess

import modal

HERE = Path(__file__).resolve().parent
app = modal.App('kani-advanced')
volume = modal.Volume.from_name('kani-advanced', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.8.0', 'numpy==2.3.3', 'pyarrow==21.0.0', 'huggingface_hub==0.35.3')
         .add_local_file(HERE / 'kani_advanced.py', '/app/kani_advanced.py')
         .add_local_file(HERE / 'kani_tokenizer.json', '/app/kani_tokenizer.json')
         .add_local_file(HERE / 'kani_tokenizer_multilingual.json', '/app/kani_tokenizer_multilingual.json'))

VOLUME, LOCAL_DATA = Path('/vol'), Path('/tmp/kani-advanced-data')
DATA_FILES = ('manifest.json', 'train.bin', 'validation.bin')


def fineweb2(code, kind=None, offset=0):
    """A FineWeb-2 language: 7.5% of every session, its own test file held out, documents kept in its script."""
    return {'repo': 'HuggingFaceFW/fineweb-2', 'prefix': f'data/{code}/train/', 'validation_file': f'data/{code}/test/000_00000.parquet',
            'per_session': 1, 'share': 0.075, **({'filter': kind} if kind else {}), **({'offset': offset} if offset else {})}



COMMON = {'input_root': str(VOLUME / 'input'), 'data_dir': str(LOCAL_DATA), 'width': 1024, 'layers': 16, 'heads': 16, 'context': 1024,
          'micro_batch': 4, 'continue_from_previous': True, 'workers': 8}
RUNS = {
    'english': {**COMMON, 'run_dir': str(VOLUME / 'run/kani-advanced'), 'tokenizer': '/app/kani_tokenizer.json',
                'target_tokens': 4_000_000_000, 'kept': str(VOLUME / 'data')},
    'multilingual': {**COMMON, 'run_dir': str(VOLUME / 'run/kani-multilingual'), 'tokenizer': '/app/kani_tokenizer_multilingual.json',
                     'target_tokens': 4_500_000_000, 'kept': str(VOLUME / 'data-multilingual'),
                     'sources': [
                         {'repo': 'HuggingFaceFW/fineweb-edu', 'prefix': 'sample/10BT/', 'per_session': 1, 'share': 0.30},
                         {'repo': 'HuggingFaceTB/smollm-corpus', 'prefix': 'cosmopedia-v2/', 'per_session': 1, 'share': 0.10},
                         fineweb2('kor_Hang', 'hangul'), fineweb2('deu_Latn'), fineweb2('fra_Latn'), fineweb2('spa_Latn'),
                         fineweb2('por_Latn'), fineweb2('jpn_Jpan', 'kana'),
                         fineweb2('cmn_Hani', 'simplified'), fineweb2('cmn_Hani', 'traditional', offset=185)]},
}
PREPARE_CPUS = 8


def write_settings(run, hours):
    path = Path(f'/tmp/{run}-settings.json')
    path.write_text(json.dumps({**{k: v for k, v in RUNS[run].items() if k != 'kept'}, 'hours': hours}, indent=1))
    return str(path)


def script(command, settings, *extra):
    subprocess.run(['python', '/app/kani_advanced.py', command, '--settings', settings, *extra], check=True, cwd='/app')


@app.function(image=image, volumes={str(VOLUME): volume}, cpu=PREPARE_CPUS, memory=32768, timeout=3 * 3600)
def prepare(run: str):
    """Download and tokenize this session's text (about 1.2 billion tokens), kept in the volume for the GPU."""
    kept = Path(RUNS[run]['kept'])
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    if all((kept / name).exists() for name in DATA_FILES):
        shutil.copy(kept / 'manifest.json', LOCAL_DATA / 'manifest.json')
    script('prepare', write_settings(run, 0))
    if (LOCAL_DATA / 'train.bin').exists():
        kept.mkdir(parents=True, exist_ok=True)
        for name in DATA_FILES:
            shutil.copy(LOCAL_DATA / name, kept / name)
        volume.commit()


@app.function(image=image, volumes={str(VOLUME): volume}, gpu='L4', cpu=2, memory=16384, timeout=24 * 3600)
def train(run: str, hours: float):
    """Train for `hours` from the run's latest checkpoint."""
    shutil.copytree(RUNS[run]['kept'], LOCAL_DATA, dirs_exist_ok=True)
    try:
        script('train', write_settings(run, hours))
    finally:
        volume.commit()


@app.function(image=image, volumes={str(VOLUME): volume}, cpu=2, memory=16384, timeout=3600)
def retokenize():
    """The English run's latest checkpoint, moved to the multilingual vocabulary, as the multilingual run's start."""
    target = Path(RUNS['multilingual']['run_dir']) / 'checkpoint.pt'
    if target.exists():
        raise SystemExit(f'{target} exists already: the multilingual run has started')
    script('retokenize', write_settings('multilingual', 0), '--checkpoint', str(Path(RUNS['english']['run_dir']) / 'checkpoint.pt'),
           '--old-tokenizer', RUNS['english']['tokenizer'])
    volume.commit()


@app.function(image=image, volumes={str(VOLUME): volume})
def report(run: str):
    volume.reload()
    run_dir = Path(RUNS[run]['run_dir'])
    lines = []
    for name in ('status.json', 'progress.json', 'samples.json'):
        path = run_dir / name
        lines.append(f'== {name}\n' + (path.read_text() if path.exists() else '(not yet)'))
    return '\n'.join(lines)


@app.function(image=image, volumes={str(VOLUME): volume}, timeout=24 * 3600)
def session(run: str, hours: float):
    """Prepare, then train, on Modal's side: it goes on after this computer sleeps or the terminal closes."""
    prepare.remote(run)
    train.remote(run, hours)
    return report.local(run)


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Start or check a Kani Advanced session on the deployed Modal app.')
    parser.add_argument('command', choices=['start', 'status', 'retokenize'])
    parser.add_argument('--run', choices=sorted(RUNS), default='multilingual')
    parser.add_argument('--hours', type=float, default=8.5, help='training time; 8.5 hours is about $9 of the $30 a month')
    args = parser.parse_args()
    if args.command == 'status':
        print(modal.Function.from_name(app.name, 'report').remote(args.run))
    elif args.command == 'retokenize':
        modal.Function.from_name(app.name, 'retokenize').remote()
        print(modal.Function.from_name(app.name, 'report').remote('multilingual'))
    else:
        if not 0.25 <= args.hours <= 20:
            raise SystemExit('Use --hours between 0.25 and 20 (Modal stops a function after 24 hours).')
        call = modal.Function.from_name(app.name, 'session').spawn(args.run, args.hours)
        print(f'Started a {args.hours}-hour {args.run} session (call {call.object_id}); follow it at modal.com/apps. '
              f'Check: python kaggle/modal_kani_advanced.py status --run {args.run}')


if __name__ == '__main__':
    cli()
