"""Live step counts for running training jobs.

    python3 -m kelus.progress                # running jobs in training-runs/
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
        if "kelus.train" in line and (out := re.search(r"--out\s+(\S+)", line)):
            jobs[Path(out.group(1)).with_suffix(".log").name] = int(steps.group(1)) if steps else 6000
        elif "kelus.pretrain" in line and (run := re.search(r"--run\s+(\S+)", line)):
            # Pretraining logs are written next to the run directory: training-runs/<run>.log
            jobs[Path(run.group(1)).name + ".log"] = int(steps.group(1)) if steps else 4000
    return jobs


def summary(log, jobs, full=False):
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    reports = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "step" in item:
            reports.append(item)
    updated = log.stat().st_mtime
    status = log.with_suffix("") / "status.json"  # pretraining rewrites this every step; its log only at checkpoints
    try:
        live = json.loads(status.read_text())
        if reports and live.get("stage") == "pretraining" and live["step"] > reports[-1]["step"]:
            reports.append({**live, "validation_loss": reports[-1]["validation_loss"]})
            updated = status.stat().st_mtime
    except (OSError, ValueError, KeyError):
        pass
    finished = any(line.startswith(("Saved ", "Early stop", "Interrupted")) for line in lines) or (reports and reports[-1].get("state") == "complete")
    failed = any("Traceback" in line or "Error" in line for line in lines)
    if not reports:
        return f"{log.stem:<22} starting…" if log.name in jobs else None
    first, last = reports[0], reports[-1]
    total = jobs.get(log.name)
    best = min(r["validation_loss"] for r in reports if r.get("validation_loss") is not None)
    status = "failed" if failed else "done" if finished else "running" if total else "stopped"
    offset = 1 if first["step"] == 0 else 0  # pretraining reports step 0 first; kelus.train reports step 1
    if total is None:
        total = last["step"] - first["step"] + 1 - offset
    done_steps = last["step"] - first["step"] + 1 - offset
    rate = (done_steps - 1) / max(last["elapsed_seconds"] - first["elapsed_seconds"], 1e-9)
    if status == "running" and rate > 0:
        # Logs arrive every few hundred steps; estimate progress in between.
        done_steps = min(total, done_steps + rate * max(0.0, time.time() - updated))
    fraction = 1.0 if status == "done" else min(1.0, done_steps / total)
    bar = "█" * round(fraction * WIDTH) + "░" * (WIDTH - round(fraction * WIDTH))
    eta = ""
    if status == "running" and rate > 0:
        minutes = int((total - done_steps) / rate // 60)
        eta = f"{minutes // 60}h{minutes % 60:02d}m left"
    step = f"step {int(first['step'] + offset + done_steps - 1)}/{first['step'] + offset + total - 1}"
    if not full:
        return f"{log.stem:<18} {step:<16} {fraction:5.1%}  {eta or status}"
    return (f"{log.stem:<22} [{bar}] {fraction:5.1%}  {step:<16} "
            f"val {last['validation_loss']:.4f} (best {best:.4f})  {status} {eta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="Training logs (default: training-runs/*.log)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--full", action="store_true", help="Also show a progress bar and validation loss")
    args = parser.parse_args()
    live = sys.stdout.isatty() and not args.once
    try:
        while True:
            # Re-read the folder each time so a job that starts later (a chained run) appears.
            logs = [Path(p) for p in args.logs] or sorted(Path("training-runs").glob("*.log"))
            if not logs:
                sys.exit("No training logs found. Run from the KelusModelB1 folder or pass a log path.")
            jobs = running_jobs()
            shown = [log for log in logs if args.logs or log.name in jobs] or \
                [log for log in logs if time.time() - log.stat().st_mtime < 3600] or logs[-1:]
            rows = [row for row in (summary(log, jobs, args.full) for log in shown) if row]
            if live:
                width = shutil.get_terminal_size().columns - 1
                rows = [row[:width] for row in rows]  # never wrap, so the redraw stays in place
                sys.stdout.write("\x1b[H\x1b[2J" + time.strftime("%H:%M:%S") + "  Ctrl+C to close\n\n")
            print("\n".join(rows) or "No training running.", flush=True)
            if not live:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
