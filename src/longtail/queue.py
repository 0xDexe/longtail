"""Run a queue of experiments back to back inside one GPU job.

    python -m longtail.queue experiments/commit2.toml
    python -m longtail.queue --resume results/queue-20261106-2100-123456   # finish an interrupted queue
    python -m longtail.queue experiments/commit2.toml --dry-run            # validate + print commands

Plan format (TOML): a [defaults] table plus one [[run]] table per experiment. Every key
must be a RolloutConfig field, checked before any GPU time is spent.

Each run is a separate subprocess, so GPU memory is fully released between runs and one
crash doesn't kill the queue. All output (including vLLM's engine-core process) is captured
to <queue_dir>/<run>/run.log and mirrored to the console with a [run] prefix. At the end
<queue_dir>/comparison.md compares every run against the first one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from .config import FIELD_NAMES


def load_plan(path: str | Path) -> list[dict]:
    data = tomllib.loads(Path(path).read_text())
    defaults = data.get("defaults", {})
    runs = data.get("run", [])
    if not runs:
        raise ValueError(f"{path}: no [[run]] entries")
    plan, names = [], set()
    for r in runs:
        params = {**defaults, **r}
        name = params.pop("name", None)
        if not name:
            raise ValueError(f"{path}: every [[run]] needs a name")
        if name in names:
            raise ValueError(f"{path}: duplicate run name {name!r}")
        unknown = set(params) - FIELD_NAMES
        if unknown:
            raise ValueError(f"{path}: run {name!r} has unknown keys {sorted(unknown)}")
        names.add(name)
        plan.append({"name": name, "params": params})
    return plan


def to_cli(params: dict) -> list[str]:
    out = []
    for k, v in params.items():
        flag = k.replace("_", "-")
        if isinstance(v, bool):
            out.append(f"--{flag}" if v else f"--no-{flag}")
        else:
            out += [f"--{flag}", str(v)]
    return out


class QueueLog:
    def __init__(self, path: Path):
        self.f = path.open("a", buffering=1)

    def __call__(self, msg: str, prefix: str = "queue") -> None:
        line = f"[{prefix}] {time.strftime('%H:%M:%S')} {msg.rstrip()}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        self.f.write(line)


def _stream(cmd: list[str], log_path: Path, prefix: str, qlog: QueueLog) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "LONGTAIL_QUEUE": "1"}
    with log_path.open("a", buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(f"{time.strftime('%H:%M:%S')} {line}")
            qlog(line, prefix=prefix)
        return proc.wait()


def _opt(s: dict, key: str, fmt: str) -> str:
    return format(s[key], fmt) if key in s else "-"


def comparison_table(queue_dir: Path, plan: list[dict], status: dict) -> str:
    rows = [
        "| run | mode | status | step time (s) | tok/s | slot occupancy | tail-wait | "
        "staleness mean/max | reward | speedup | preemptions/step | KV max | spec acceptance |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    base = None
    for item in plan:
        name = item["name"]
        st = status.get(name, {}).get("state", "pending")
        path = queue_dir / name / "summary.json"
        if not path.exists():
            rows.append(f"| {name} | {item['params'].get('mode', 'barrier')} | {st} | | | | | | | | | | |")
            continue
        s = json.loads(path.read_text())
        base = base or s["wall_s"]
        rows.append(
            f"| {name} | {s['mode']} | {st} | {s['wall_s']:.1f} | {s['tokens_per_s']:.0f} | "
            f"{s['slot_occupancy']:.2f} | {s['tail_wait_fraction']:.2f} | "
            f"{s['staleness_mean']:.2f}/{s['staleness_max']} | {s['mean_reward']:.3f} | "
            f"{base / s['wall_s']:.2f}x | {_opt(s, 'tel_preemptions', '.0f')} | "
            f"{_opt(s, 'tel_kv_usage_max', '.2f')} | {_opt(s, 'tel_spec_acceptance', '.3f')} |"
        )
    return "\n".join(rows) + "\n\nSpeedup = step time of the first completed run / this run's step time.\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="experiment plan (.toml)")
    ap.add_argument("--resume", help="existing queue dir; skips runs already marked done")
    ap.add_argument("--only", help="comma-separated run names to execute")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args(argv)

    if args.resume:
        queue_dir = Path(args.resume)
        plan_path = queue_dir / "plan.toml"
    elif args.plan:
        plan_path = Path(args.plan)
        job = os.environ.get("JOB_ID")
        queue_dir = Path(args.output_dir) / (
            f"queue-{time.strftime('%Y%m%d-%H%M%S')}" + (f"-{job}" if job else "")
        )
    else:
        ap.error("give a plan file or --resume")

    plan = load_plan(plan_path)  # fail fast on typos, before touching the GPU
    if args.only:
        wanted = set(args.only.split(","))
        plan = [p for p in plan if p["name"] in wanted]

    if args.dry_run:
        for item in plan:
            print(item["name"], " ".join(to_cli(item["params"])))
        return 0

    queue_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        shutil.copy(plan_path, queue_dir / "plan.toml")
    status_path = queue_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    qlog = QueueLog(queue_dir / "queue.log")
    qlog(f"queue_dir={queue_dir} runs={[p['name'] for p in plan]}")

    failures = 0
    for item in plan:
        name = item["name"]
        if status.get(name, {}).get("state") == "done":
            qlog(f"skip {name} (already done)")
            continue
        run_dir = queue_dir / name
        run_dir.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "longtail.experiment", "--run-dir", str(run_dir),
               *to_cli(item["params"])]
        start = time.time()
        status[name] = {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": cmd}
        status_path.write_text(json.dumps(status, indent=2))
        qlog(f"start {name}: {' '.join(cmd[3:])}")

        rc = _stream(cmd, run_dir / "run.log", name, qlog)

        status[name].update(state="done" if rc == 0 else "failed", returncode=rc,
                            duration_s=round(time.time() - start, 1))
        status_path.write_text(json.dumps(status, indent=2))
        qlog(f"{'finished' if rc == 0 else 'FAILED'} {name} rc={rc} in {time.time() - start:.0f}s")
        failures += rc != 0

    table = comparison_table(queue_dir, load_plan(plan_path), status)
    (queue_dir / "comparison.md").write_text(table)
    qlog("comparison:\n" + table)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
