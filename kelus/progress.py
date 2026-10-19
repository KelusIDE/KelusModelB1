"""Live progress bars for running training jobs.

    python3 -m kelus.progress                # every log in training-runs/
    python3 -m kelus.progress path/to/run.log

Reads the JSON lines `kelus.train` prints. Running jobs are matched by their
--out path to find the step budget. Ctrl+C exits; training is unaffected.
"""
import argparse
import json
from pathlib import Path
import re
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
        if "kelus.train" not in line:
            continue
        out = re.search(r"--out\s+(\S+)", line)
        steps = re.search(r"--steps\s+(\d+)", line)
        if out:
            jobs[Path(out.group(1)).with_suffix(".log").name] = int(steps.group(1)) if steps else 6000
    return jobs


def summary(log, jobs):
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    reports = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "step" in item:
            reports.append(item)
    finished = any(line.startswith(("Saved ", "Early stop", "Interrupted")) for line in lines)
    failed = any("Traceback" in line or "Error" in line for line in lines)
    if not reports:
        return f"{log.stem:<22} starting…"
    first, last = reports[0], reports[-1]
    total = jobs.get(log.name)
    best = min(r["validation_loss"] for r in reports)
    status = "failed" if failed else "done" if finished else "running" if total else "stopped"
    if total is None:
        total = last["step"] - first["step"] + 1
    done_steps = last["step"] - first["step"] + 1
    rate = (done_steps - 1) / max(last["elapsed_seconds"] - first["elapsed_seconds"], 1e-9)
    if status == "running" and rate > 0:
        # Logs arrive every few hundred steps; estimate progress in between.
        done_steps = min(total, done_steps + rate * max(0.0, time.time() - log.stat().st_mtime))
    fraction = 1.0 if status == "done" else min(1.0, done_steps / total)
    bar = "█" * round(fraction * WIDTH) + "░" * (WIDTH - round(fraction * WIDTH))
    eta = ""
    if status == "running" and rate > 0:
        seconds = (total - done_steps) / rate
        eta = f"ETA {int(seconds // 60)}m{int(seconds % 60):02d}s"
    return (f"{log.stem:<22} [{bar}] {fraction:5.1%}  step {int(first['step'] + done_steps - 1):>6}/{first['step'] + total - 1:<6} "
            f"val {last['validation_loss']:.4f} (best {best:.4f})  {status} {eta}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="Training logs (default: training-runs/*.log)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()
    logs = [Path(p) for p in args.logs] or sorted(Path("training-runs").glob("*.log"))
    if not logs:
        sys.exit("No training logs found. Run from the KelusModelB1 folder or pass a log path.")
    printed = 0
    try:
        while True:
            jobs = running_jobs()
            shown = [log for log in logs if args.logs or log.name in jobs or time.time() - log.stat().st_mtime < 3600]
            rows = [summary(log, jobs) for log in shown or logs]
            if printed and sys.stdout.isatty():
                sys.stdout.write(f"\x1b[{printed}F\x1b[J")
            print("\n".join(rows), flush=True)
            printed = len(rows)
            if args.once or not any(log.name in jobs for log in shown):
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
