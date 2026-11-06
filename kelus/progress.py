"""Live step counts for running training jobs.

    python3 -m kelus.progress                # running jobs in training-runs/, with a progress bar
    python3 -m kelus.progress --full         # with progress bar and validation loss
    python3 -m kelus.progress path/to/run.log

Reads the JSON lines `kelus.train` and `kelus.pretrain` print. Running jobs are
matched by --out/--run to find the step budget. Ctrl+C exits; training is unaffected.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

WIDTH = 28


def running_jobs():
    """Map each running job's log path to its requested step count."""
    try:
        listing = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    jobs = {}
    for line in listing.splitlines():
        steps = re.search(r"--steps\s+(\d+)", line)
        if re.search(r"kelus\.(?:train\b|gpu_train instruct)", line) and (out := re.search(r"--out\s+(\S+)", line)):
            path = Path(out.group(1)).with_suffix('.log')
            budget = int(steps.group(1)) if steps else 6000
            jobs[str(path.resolve())] = budget
            jobs[path.name] = budget
        elif re.search(r"kelus\.(?:pretrain\b|gpu_train pretrain|pixel_dit train)", line) and (run := re.search(r"--run\s+(\S+)", line)):

            jobs[Path(run.group(1)).name + ".log"] = int(steps.group(1)) if steps else 4000
    return jobs


def bar(fraction):
    """A bar with eighth-block precision."""
    eighths = round(max(0.0, min(1.0, fraction)) * WIDTH * 8)
    full, part = divmod(eighths, 8)
    text = "█" * full + ("▏▎▍▌▋▊▉"[part - 1] if part else "")
    return text + "░" * (WIDTH - len(text))


def timing(seconds):
    if seconds is None or seconds < 0:
        return ""
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m left · done ~{time.strftime('%H:%M', time.localtime(time.time() + seconds))}"


def row(name, fraction, step, total, state, eta_seconds=None, loss="", full=False):
    eta = timing(eta_seconds) if state == "running" else ""
    text = f"{name:<18} [{bar(fraction)}] {fraction:6.1%}  step {step:,}/{total:,}  {eta or state}"
    return text + (f"  {loss}" if full and loss else "")


def summary(log, jobs, full=False):
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    budget = jobs.get(str(log.resolve())) or (jobs.get(log.name) if log.name != 'instruction.log' else None)
    known_total = None
    declared_state = None
    for path in [log.parent / 'status.json', log.parent / 'recipe.json']:
        if path.exists():
            try:
                info = json.loads(path.read_text())
                known_total = known_total or info.get('total_steps') or info.get('steps')
                if path.name == 'status.json':
                    declared_state = info.get('state')
            except (ValueError, OSError):
                pass
    live_path = log.with_suffix("") / "status.json"
    if live_path.exists():

        try:
            live = json.loads(live_path.read_text())
        except ValueError:
            live = {}
        if live.get("stage") == "pretraining":
            state = live.get("state", "running")
            if state == "running" and log.name not in jobs:
                state = "stopped (no process)"
            step, total = int(live.get("step", 0)), int(live.get("total_steps") or 1)
            rate = live.get("steps_per_second") or 0
            eta = (total - step) / rate if rate else None
            loss = f"val {live['validation_loss']:.4f}" if live.get("validation_loss") is not None else ""
            return row(log.stem, 1.0 if state == "complete" else step / total, step, total, "done" if state == "complete" else state, eta, loss, full)

    start = max([i for i, line in enumerate(lines) if line.startswith('{"name": ')], default=0)
    lines = lines[start:]
    reports = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "step" in item and "elapsed_seconds" in item:
            reports.append(item)
    interrupted = any('KeyboardInterrupt' in line or line.startswith('Interrupted') for line in lines)
    finished = any(line.startswith('Saved ') for line in lines)
    early_stop = any(line.startswith('Early stop') for line in lines)
    failed = any('Traceback' in line for line in lines) and not interrupted
    if not reports:
        return f"{log.stem:<18} starting…" if budget else None
    first, last = reports[0], reports[-1]
    begun = first["step"] - 1
    total = budget or (max(1, int(known_total)-begun) if known_total else max(1, last['step']-begun))
    done = last["step"] - begun
    state = 'running' if budget else 'stopped' if interrupted or declared_state == 'stopped' else 'failed' if failed else 'early stop' if early_stop else 'done' if finished else 'stopped'
    rate = (done - 1) / max(last["elapsed_seconds"] - first["elapsed_seconds"], 1e-9) if done > 1 else 0
    if state == "running" and rate > 0:
        done = min(total, done + rate * max(0.0, time.time() - log.stat().st_mtime))
    absolute_total = begun + total
    fraction = min(1.0, (begun + done) / absolute_total)
    losses = [r["validation_loss"] for r in reports if r.get("validation_loss") is not None]
    loss = f"val {losses[-1]:.4f} (best {min(losses):.4f})" if losses else ""
    eta = (total - done) / rate if rate > 0 else None
    if not budget and not known_total and state != 'done':
        return f'{log.stem:<18} step {last["step"]:,} · {state} · target unknown' + (f'  {loss}' if full and loss else '')
    return row(log.stem, fraction, int(begun + done), absolute_total, state, eta, loss, full)


TESTS = re.compile(r"-m kelus\.(reading_eval|decide_eval|benchmark|spec_eval|paint_eval|code_readiness)\b.*?--checkpoint\s+([^\s$]+)")


def testing():
    """One line per checkpoint whose evaluations are running."""
    try:
        listing = subprocess.run(["ps", "-Ao", "command"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    found = {}
    for line in listing.splitlines():
        match = TESTS.search(line)
        if match and 'ython' in line.split()[0]:
            found.setdefault(match.group(2), set()).add(match.group(1).replace("_", " "))
    return [f"{'testing':<18} {Path(checkpoint).parent.name}/{Path(checkpoint).stem}: {', '.join(sorted(names))}" for checkpoint, names in sorted(found.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="Training logs (default: training-runs/*.log)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--full", action="store_true", help="Also show a progress bar and validation loss")
    args = parser.parse_args()
    live = sys.stdout.isatty() and not args.once
    try:
        while True:

            logs = [Path(p) for p in args.logs] or sorted(Path("training-runs").glob("*.log"))
            if not logs:
                sys.exit("No training logs found. Run from the KelusModelB1 folder or pass a log path.")
            jobs = running_jobs()
            shown = [log for log in logs if args.logs or log.name in jobs] or \
                [log for log in logs if time.time() - log.stat().st_mtime < 3600] or [max(logs, key=lambda log: log.stat().st_mtime)]
            rows = [line for line in (summary(log, jobs, args.full) for log in shown) if line] + testing()
            if live:
                width = shutil.get_terminal_size().columns - 1
                rows = [row[:width] for row in rows]
                sys.stdout.write("\x1b[H\x1b[2J" + time.strftime("%H:%M:%S") + "  Ctrl+C to close\n\n")
            print("\n".join(rows) or "No training running.", flush=True)
            if not live:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
