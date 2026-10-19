"""Local, fail-closed Firebase release watcher. See AUTO_DEPLOY.md."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'training-runs' / 'auto-deploy'
FIREBASE = Path('/Users/hashtagpro/.nvm/versions/node/v25.2.1/bin/firebase')
URL = 'https://kelus-b1.web.app'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def tree_digest(directory):
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def command(args, cwd, log, timeout=7200):
    with Path(log).open('a') as output:
        subprocess.run(args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def prepare(checkpoint, data, state=STATE):
    """Explicitly register a finished instruction run; never called by pretraining."""
    from .model import Model
    from .export_web import export
    checkpoint, data = Path(checkpoint).resolve(), Path(data).resolve()
    foundation = ROOT / 'training-runs/b1-foundation'
    status = read(foundation / 'status.json')
    if status.get('state') != 'complete' or status.get('step', 0) < status.get('total_steps', 1):
        raise ValueError('Finish foundation pretraining before preparing an instruction release.')
    identity = digest(checkpoint)
    release = state / 'releases' / identity
    release.mkdir(parents=True, exist_ok=False)
    frozen = release / 'candidate.npz'
    shutil.copyfile(checkpoint, frozen)
    if digest(frozen) != identity:
        raise ValueError('Checkpoint changed while copying; prepare an immutable checkpoint.')
    model, header, _ = Model.load(frozen)
    metadata = header.get('metadata', {})
    if metadata.get('stage') == 'pretraining' or not metadata.get('training_examples') or not metadata.get('parent_sha256'):
        raise ValueError('Candidate must be instruction-trained from our foundation checkpoint.')
    if model.parameter_count != 8691840 or metadata.get('steps', 0) <= 0:
        raise ValueError('Candidate is not the trained 8.69M B1 model.')
    del model
    shutil.copyfile(data, release / 'instruction.jsonl')
    # The instruction trainer fingerprints the raw JSONL bytes.
    if digest(release / 'instruction.jsonl') != metadata.get('data_sha256'):
        raise ValueError('Instruction data does not match the checkpoint.')
    parents = {digest(p) for p in (foundation / 'best.npz', foundation / 'latest.npz') if p.exists()}
    if metadata['parent_sha256'] not in parents:
        raise ValueError('Instruction model does not descend from this completed foundation run.')
    from .code_readiness import CASES, WEBSITE
    probes = {case[1] for case in CASES} | {WEBSITE}
    for line in (release / 'instruction.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('split') != 'validation' and any(m.get('role') == 'user' and m.get('content') in probes for m in record['messages']):
            raise ValueError('Readiness prompt leaked into instruction training.')
    shutil.copytree(ROOT / 'site', release / 'site')
    export(frozen, release / 'site' / 'model')
    config = read(ROOT / 'firebase.json')
    if config['hosting']['site'] != 'kelus-b1':
        raise ValueError('Unexpected Firebase site.')
    config = {'hosting': config['hosting'] | {'public': 'site'}}
    write(release / 'firebase.json', config)
    log = release / 'checks.log'
    command([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], ROOT, log)
    for module, filename in [('code_readiness', 'readiness.json'), ('evaluate', 'regressions.json'), ('benchmark', 'benchmark.json')]:
        args = [sys.executable, '-m', 'kelus.' + module, '--checkpoint', str(frozen), '--out', str(release / filename)]
        if module != 'code_readiness':
            args += ['--data', str(release / 'instruction.jsonl')]
        command(args, ROOT, log)
    # Check the actual export with the frozen browser engine (not just a tiny test model).
    script = (ROOT / 'tests' / 'web_parity.mjs').read_text().replace("'../site/b1.js'", "'./site/b1.js'")
    (release / 'parity.mjs').write_text(script)
    conversations = [[{'role': 'user', 'content': p}] for p in ['hello!', 'Write a Python square function.']]
    write(release / 'cases.json', {'texts': [], 'helpers': [], 'conversations': conversations, 'max_tokens': 24})
    result = subprocess.run([str(FIREBASE.parent / 'node'), str(release / 'parity.mjs'), str(release / 'site/model'), str(release / 'cases.json')],
                            capture_output=True, text=True, check=True, timeout=600)
    from .runtime import prepare_reply
    model, _, _ = Model.load(frozen)
    expected = []
    for messages in conversations:
        retained, omitted, tool = prepare_reply(model, messages, 24)
        expected.append({'text': tool[0] if tool else ''.join(model.generate(retained, max_tokens=24, temperature=0)),
                         'source': tool[1] if tool else 'model', 'omitted': omitted})
    if json.loads(result.stdout)['replies'] != expected:
        raise ValueError('Exported browser engine differs from Python.')
    report = read(release / 'readiness.json')
    # Keep the exact generated text for browser inspection; no generated code is executed here.
    (release / 'website-reply.txt').write_text(report['website']['reply'])
    manifest = {'checkpoint_sha256': identity, 'training_complete': True,
                'stage': 'instruction', 'prepared_at': time.time(),
                'bundle_sha256': tree_digest(release / 'site'),
                'config_sha256': digest(release / 'firebase.json'),
                'reports': {name: digest(release / name) for name in ['readiness.json', 'regressions.json', 'benchmark.json']},
                'unit_tests_passed': True, 'browser_parity_passed': True}
    from .languages import scope_sha256, verification_template
    manifest['language_scope_sha256'] = scope_sha256()
    write(release / 'language-verification.template.json', verification_template(identity))
    write(release / 'release.json', manifest)
    write(state / 'candidate.json', {'release': str(release)})
    print('Candidate prepared. Website interaction evidence is still required:', release)


def gate(release):
    m = read(release / 'release.json')
    identity = digest(release / 'candidate.npz')
    if m['checkpoint_sha256'] != identity or m.get('stage') != 'instruction' or m.get('training_complete') is not True:
        return 'Checkpoint identity or completed instruction training is missing.'
    if tree_digest(release / 'site') != m['bundle_sha256'] or digest(release / 'firebase.json') != m['config_sha256']:
        return 'Prepared website bundle or Firebase config changed; prepare again.'
    if not (m.get('unit_tests_passed') is True and m.get('browser_parity_passed') is True):
        return 'Unit tests or browser engine parity did not pass.'
    for name in ['readiness.json', 'regressions.json', 'benchmark.json']:
        if digest(release / name) != m['reports'][name]:
            return 'Test report changed: ' + name
        report = read(release / name)
        if report.get('sha256', report.get('checkpoint_sha256')) != identity:
            return 'Test report belongs to another checkpoint: ' + name
    ready = read(release / 'readiness.json')
    if ready['python_total'] != 4 or ready['python_passed'] != 4 or not all(ready['website']['structure_checks'].get(k) is True for k in ['html_document','heading','button','inline_css','inline_javascript']):
        return 'New-spec coding or website structure checks failed.'
    regression = read(release / 'regressions.json')
    benchmark = read(release / 'benchmark.json')
    for score, minimum, total in [(regression['scores']['regression'],5,5), (regression['scores']['heldout'],12,15),
                                  (regression['screenshot_sequence'],5,5), (benchmark['overall'],158,199),
                                  (benchmark['context_regression'],179,184)]:
        if score['total'] != total or score['passed'] < minimum:
            return 'Chat/code regressions fell below the released v3 baseline.'
    evidence = release / 'website-verification.json'
    if not evidence.exists():
        return 'Waiting for real browser interaction checks and B1 site preview verification.'
    qa = read(evidence)
    reply_hash = hashlib.sha256(ready['website']['reply'].encode()).hexdigest()
    if qa.get('checkpoint_sha256') != identity or qa.get('bundle_sha256') != m['bundle_sha256'] or qa.get('reply_sha256') != reply_hash:
        return 'Website verification belongs to another model, reply, or site bundle.'
    if not qa.get('tester') or not qa.get('notes') or not all(qa.get('checks', {}).get(k) is True for k in ['counter_starts_at_zero', 'counter_increments_twice', 'heading_and_button_visible', 'b1_preview_loads_and_replies']):
        return 'Website interaction checks have not all passed.'
    from .languages import scope_sha256, verification_problem
    if m.get('language_scope_sha256') != scope_sha256():
        return 'Candidate predates the requested language scope; prepare it again.'
    languages = release / 'language-verification.json'
    if not languages.exists():
        return 'Waiting for verification of the 23 coding languages and 7 chat targets.'
    return verification_problem(read(languages), identity)


def publish(release, log):
    command([str(FIREBASE), 'deploy', '--only', 'hosting', '--project', 'kelus-ide',
             '--config', str(release / 'firebase.json'), '--non-interactive'], release, log, timeout=900)


def verify_live(release):
    expected = read(release / 'site/model/b1.json')
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(URL + '/model/b1.json?release=' + str(time.time_ns()), timeout=60) as response:
                actual = json.load(response)
            with urlopen(URL + '/model/b1.bin?release=' + str(time.time_ns()), timeout=60) as response:
                weights = hashlib.sha256(response.read()).hexdigest()
            if actual['source_checkpoint_sha256'] == expected['source_checkpoint_sha256'] and weights == actual['weights_sha256'] == expected['weights_sha256']:
                return
            raise ValueError('Live model identity does not match the exported candidate.')
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(10)
    raise last_error


def tick(state=STATE, foundation=None, publisher=publish, verifier=verify_live):
    state.mkdir(parents=True, exist_ok=True)
    def status(phase, detail):
        write(state / 'status.json', {'state': phase, 'detail': detail, 'updated_at': time.time(), 'url': URL})
    if not (state / 'enabled').exists():
        return status('disabled', 'Automatic deployment is disabled.')
    with (state / 'watcher.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        foundation = foundation or ROOT / 'training-runs/b1-foundation/status.json'
        try:
            training = read(foundation)
            if training.get('state') != 'complete' or training.get('step', 0) < training.get('total_steps', 1):
                return status('waiting_pretraining', f"Pretraining {training.get('state')}: {training.get('step')}/{training.get('total_steps')}. Instruction training and release tests must follow.")
            if not (state / 'candidate.json').exists():
                return status('waiting_instruction_candidate', 'Pretraining finished. A completed, tested instruction model has not been prepared yet.')
            release = Path(read(state / 'candidate.json')['release']).resolve()
            if release.parent != (state / 'releases').resolve():
                raise ValueError('Candidate must be a prepared local release.')
            reason = gate(release)
            if reason:
                return status('held', reason)
            identity = read(release / 'release.json')['checkpoint_sha256']
            receipt = release / 'deployment.json'
            if receipt.exists():
                previous = read(receipt)
                return status(previous['state'], previous['detail'])
            if not (state / 'enabled').exists():
                return status('disabled', 'Cancelled before publishing.')
            # At-most-once publish. An uncertain network failure requires inspection, not blind retries.
            write(receipt, {'state': 'deployment_started', 'detail': 'Publish started; inspect deploy.log if interrupted.', 'sha256': identity})
            status('deploying', 'Publishing tested model to kelus-b1.web.app.')
            try:
                publisher(release, release / 'deploy.log')
                verifier(release)
            except Exception as error:
                write(receipt, {'state': 'deployment_needs_attention', 'detail': str(error), 'sha256': identity})
                raise
            message = 'Published and verified the live model weights: ' + identity
            write(receipt, {'state': 'deployed', 'detail': message, 'sha256': identity, 'time': time.time()})
            status('deployed', message)
        except Exception as error:
            status('needs_attention', str(error))


def watch():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'process.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        write(STATE / 'process.json', {'pid': os.getpid(), 'started_at': time.time()})
        while (STATE / 'enabled').exists():
            tick()
            if (STATE / 'status.json').exists() and read(STATE / 'status.json').get('state') in ('deployed', 'deployment_needs_attention', 'needs_attention'):
                return
            for _ in range(60):
                if not (STATE / 'enabled').exists():
                    return
                time.sleep(5)


def start():
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / 'enabled').touch()
    # Existing watcher owns this lock; duplicate starters exit harmlessly.
    with (STATE / 'watcher.log').open('a') as log:
        child = subprocess.Popen(['/usr/bin/nice', '-n', '10', sys.executable, '-u', '-m', 'kelus.auto_deploy', '--watch'],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    print('Background watcher started (or already running). Check --status. Launcher PID:', child.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--once', action='store_true')
    group.add_argument('--watch', action='store_true')
    group.add_argument('--start', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--disable', action='store_true')
    group.add_argument('--prepare', metavar='CHECKPOINT')
    parser.add_argument('--data')
    parser.add_argument('--training-complete', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        if not args.training_complete or not args.data:
            parser.error('--prepare requires --training-complete and --data after instruction training finishes')
        prepare(args.prepare, args.data)
    elif args.start:
        start()
    elif args.watch:
        watch()
    elif args.disable:
        (STATE / 'enabled').unlink(missing_ok=True)
        tick()
    elif args.once:
        tick()
    else:
        value = read(STATE / 'status.json') if (STATE / 'status.json').exists() else {'state': 'not_started'}
        with (STATE / 'process.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                value['watcher_running'] = False
            except BlockingIOError:
                value['watcher_running'] = True
        value['enabled'] = (STATE / 'enabled').exists()
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
