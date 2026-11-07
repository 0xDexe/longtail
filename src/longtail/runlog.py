"""Autologging: environment capture and stdout/stderr tee for standalone runs."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
import time

PACKAGES = ("vllm", "torch", "transformers", "datasets")


def _cmd(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def capture_env() -> dict:
    """Everything needed to reproduce or explain a number later."""
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    porcelain = _cmd(["git", "status", "--porcelain"])
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "job_id": os.environ.get("JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": _cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "python": platform.python_version(),
        "packages": versions,
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(porcelain) if porcelain is not None else None,
        "argv": sys.argv,
    }


class _Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, s):
        self.stream.write(s)
        self.logfile.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_output(path) -> None:
    """Mirror this process's Python-level stdout/stderr into `path`.

    Output from vLLM's engine-core subprocess bypasses this; run through the queue
    (python -m longtail.queue) to capture everything, including child processes.
    """
    f = open(path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
