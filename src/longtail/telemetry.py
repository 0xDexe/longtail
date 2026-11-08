"""Engine telemetry: periodically sample vLLM's own metrics during a run.

Writes every sample to telemetry.jsonl (for time-series plots) and summarizes any time
window into per-step fields:

tel_preemptions        sequences evicted from the KV cache and recomputed (counter delta)
tel_kv_usage_mean/max  KV cache utilization, 0..1
tel_running_mean       sequences actually decoding
tel_waiting_mean       sequences queued (admitted by us, not yet scheduled by vLLM)
tel_gen_tokens_per_s   all tokens generated in the window, delivered or not
tel_spec_acceptance    accepted / drafted tokens (speculative decoding only)
tel_spec_tokens_per_draft  accepted tokens per verification (speculative decoding only)
tel_tail_tok_s         generated tokens/s over intervals with <= tail_running sequences
                       decoding: tail throughput, much less sensitive than step time to which
                       sequence happens to be longest

Metric names drift across vLLM releases, so fields are matched by suffix and missing ones
are simply omitted. If the metrics reader can't be imported, telemetry turns itself off and
the run continues.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Iterable

Snapshot = Callable[[], Iterable]  # returns objects with .name, .value and a class name


def _vllm_snapshot() -> Snapshot | None:
    try:
        from vllm.v1.metrics.reader import get_metrics_snapshot
    except Exception as e:  # pragma: no cover - depends on vLLM version
        print(f"telemetry disabled: cannot import vLLM metrics reader ({e})", flush=True)
        return None
    return get_metrics_snapshot


def _strip(name: str) -> str:
    name = name.split(":", 1)[-1]
    return name[: -len("_total")] if name.endswith("_total") else name


class Telemetry:
    def __init__(self, interval: float = 1.0, path: str | Path | None = None,
                 snapshot: Snapshot | None = None, clock: Callable[[], float] = time.perf_counter,
                 tail_running: int = 64):
        self.interval = interval
        self.tail_running = tail_running
        self.clock = clock
        self._snap = snapshot if snapshot is not None else (_vllm_snapshot() if interval > 0 else None)
        self.samples: list[tuple[float, dict[str, float]]] = []
        self.kinds: dict[str, str] = {}
        self._f = open(path, "w") if (path and self._snap) else None
        self._task: asyncio.Task | None = None
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self._snap is not None

    def sample(self) -> None:
        vals: dict[str, float] = {}
        for m in self._snap():
            kind = type(m).__name__
            if kind not in ("Counter", "Gauge"):
                continue
            key = _strip(m.name)
            vals[key] = vals.get(key, 0.0) + float(m.value)  # sum over label sets
            self.kinds[key] = kind
        t = self.clock()
        self.samples.append((t, vals))
        if self._f:
            self._f.write(json.dumps({"t": t, **vals}) + "\n")

    async def _loop(self) -> None:
        while True:
            try:
                self.sample()
            except Exception as e:
                if not self._warned:
                    print(f"telemetry sample failed: {e}", flush=True)
                    self._warned = True
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self.enabled:
            try:
                self.sample()
            except Exception:
                pass
        if self._f:
            self._f.close()
            self._f = None

    def current(self) -> dict | None:
        """Fresh reading for the admission controller: kv, preemptions (total), waiting, running."""
        if not self.enabled:
            return None
        self.sample()
        v = self.samples[-1][1]
        names = {
            "kv": self._find("kv_cache_usage_perc", "gpu_cache_usage_perc"),
            "preemptions": self._find("num_preemptions"),
            "waiting": self._find("num_requests_waiting"),
            "running": self._find("num_requests_running"),
        }
        if names["kv"] is None:
            return None
        return {k: v.get(n, 0.0) for k, n in names.items() if n}

    def _find(self, *suffixes: str) -> str | None:
        for suf in suffixes:
            for name in self.kinds:
                if name.endswith(suf):
                    return name
        return None

    def window(self, t0: float, t1: float) -> dict:
        """Summary of [t0, t1]. Counters: last sample <= t1 minus last sample <= t0.
        Gauges: mean / max over samples inside the window."""
        if not self.samples or t1 <= t0:
            return {}
        before = None
        end = None
        inside = []
        for t, v in self.samples:
            if t <= t0:
                before = v
            if t <= t1:
                end = v
            if t0 <= t <= t1:
                inside.append(v)
        if end is None:
            return {}
        before = before if before is not None else (inside[0] if inside else end)

        def delta(name):
            return end.get(name, 0.0) - before.get(name, 0.0) if name else None

        def gauge(name):
            xs = [v[name] for v in inside if name in v] if name else []
            return (sum(xs) / len(xs), max(xs)) if xs else None

        out: dict[str, float] = {}
        pre = self._find("num_preemptions")
        if pre:
            out["tel_preemptions"] = delta(pre)
        kv = gauge(self._find("kv_cache_usage_perc", "gpu_cache_usage_perc"))
        if kv:
            out["tel_kv_usage_mean"], out["tel_kv_usage_max"] = kv
        run = gauge(self._find("num_requests_running"))
        if run:
            out["tel_running_mean"] = run[0]
        wait = gauge(self._find("num_requests_waiting"))
        if wait:
            out["tel_waiting_mean"] = wait[0]
        gen = self._find("generation_tokens")
        if gen:
            out["tel_gen_tokens_per_s"] = delta(gen) / (t1 - t0)
        run_name = self._find("num_requests_running")
        if gen and run_name:
            pts = [(t, v) for t, v in self.samples if t0 <= t <= t1]
            tok = dt = 0.0
            for (ta, va), (tb, vb) in zip(pts, pts[1:]):
                if va.get(run_name, 1e9) <= self.tail_running:
                    tok += vb.get(gen, 0.0) - va.get(gen, 0.0)
                    dt += tb - ta
            if dt > 0:
                out["tel_tail_tok_s"] = tok / dt
                out["tel_tail_time_frac"] = dt / (t1 - t0)
        acc = self._find("spec_decode_num_accepted_tokens")
        drafted = self._find("spec_decode_num_draft_tokens")
        drafts = self._find("spec_decode_num_drafts")
        if acc and drafted and delta(drafted):
            out["tel_spec_acceptance"] = delta(acc) / delta(drafted)
        if acc and drafts and delta(drafts):
            out["tel_spec_tokens_per_draft"] = delta(acc) / delta(drafts)
        return out
