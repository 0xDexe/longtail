"""Per-request traces, engine concurrency tracking, and step-level metrics.

A "step" is one batch handed to the trainer: `prompts_per_step` complete groups.
Metrics are comparable across scheduling modes:

wall_s              time to deliver one batch (barrier: until the slowest sequence finishes)
tokens_per_s        output tokens delivered / wall_s
avg_concurrency     time-averaged number of sequences in flight during the step
slot_occupancy      avg_concurrency / capacity. 1.0 = engine always full; low = idle slots
tail_wait_fraction  (t_last - t_p90) / wall_s: share of the step spent on the slowest 10%
staleness_*         delivered_step - submit_step (policy versions behind). Barrier is always 0;
                    refill delivers some sequences generated under an older policy.

Slot occupancy is not GPU utilization: NVML "utilization" reads ~100% even at batch 1.
The real cost is that decode at small batch is memory-bound, so throughput collapses
as the batch drains.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, fields
from typing import Callable


@dataclass
class RequestTrace:
    request_id: str
    prompt_idx: int
    sample_idx: int
    submit_step: int
    prompt_tokens: int
    output_tokens: int
    submit_t: float
    finish_t: float
    finish_reason: str  # "stop" | "length"
    reward: float
    delivered_step: int = -1
    # Only filled when save_tokens is on; written to tokens.jsonl, not traces.jsonl.
    prompt_token_ids: list[int] | None = field(default=None, repr=False)
    token_ids: list[int] | None = field(default=None, repr=False)

    @property
    def latency_s(self) -> float:
        return self.finish_t - self.submit_t

    def to_dict(self) -> dict:
        skip = {"prompt_token_ids", "token_ids"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}

    def tokens_dict(self) -> dict:
        return {
            "request_id": self.request_id, "delivered_step": self.delivered_step,
            "submit_step": self.submit_step, "prompt_idx": self.prompt_idx,
            "sample_idx": self.sample_idx, "finish_reason": self.finish_reason,
            "prompt_token_ids": self.prompt_token_ids, "token_ids": self.token_ids,
        }


class ConcurrencyMeter:
    """Piecewise-constant record of sequences in flight; integrates over any window."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.active = 0
        self.peak = 0
        self._events: list[tuple[float, int]] = []  # (time, active count after the change)

    def add(self, delta: int, t: float | None = None) -> None:
        self.active += delta
        self.peak = max(self.peak, self.active)
        self._events.append((self.clock() if t is None else t, self.active))

    def average(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        level = 0
        for t, a in self._events:
            if t <= t0:
                level = a
            else:
                break
        total, cur = 0.0, t0
        for t, a in self._events:
            if t <= t0:
                continue
            if t >= t1:
                break
            total += level * (t - cur)
            cur, level = t, a
        total += level * (t1 - cur)
        return total / (t1 - t0)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def step_stats(
    traces: list[RequestTrace],
    step: int,
    step_start: float,
    step_end: float,
    avg_concurrency: float,
    capacity: int,
) -> dict:
    if not traces:
        raise ValueError("no traces")
    n = len(traces)
    wall = step_end - step_start
    finish = [max(0.0, t.finish_t - step_start) for t in traces]
    lengths = [float(t.output_tokens) for t in traces]
    stale = [step - t.submit_step for t in traces]
    t_p50, t_p90, t_last = percentile(finish, 50), percentile(finish, 90), max(finish)
    occupancy = avg_concurrency / capacity if capacity else 0.0

    return {
        "step": step,
        "n_seqs": n,
        "wall_s": wall,
        "output_tokens": int(sum(lengths)),
        "tokens_per_s": sum(lengths) / wall if wall > 0 else 0.0,
        "len_p50": percentile(lengths, 50),
        "len_p90": percentile(lengths, 90),
        "len_p99": percentile(lengths, 99),
        "len_max": max(lengths),
        "len_max_over_p50": max(lengths) / max(percentile(lengths, 50), 1.0),
        "avg_concurrency": avg_concurrency,
        "capacity": capacity,
        "slot_occupancy": occupancy,
        "idle_slot_fraction": 1.0 - occupancy,
        "tail_wait_fraction": (t_last - t_p90) / wall if wall > 0 else 0.0,
        "t50_fraction": t_p50 / wall if wall > 0 else 0.0,
        "truncation_rate": sum(t.finish_reason == "length" for t in traces) / n,
        "mean_reward": sum(t.reward for t in traces) / n,
        "staleness_mean": sum(stale) / n,
        "staleness_max": max(stale),
    }


SUMMARY_KEYS = (
    "wall_s", "tokens_per_s", "avg_concurrency", "slot_occupancy", "tail_wait_fraction",
    "len_p50", "len_p99", "truncation_rate", "mean_reward", "staleness_mean",
)


def summarize(steps: list[dict], skip: int = 1) -> dict:
    """Mean of per-step metrics, excluding the first `skip` steps (ramp-up)."""
    if not steps:
        raise ValueError("no steps")
    use = steps[skip:] or steps
    out = {"steps_total": len(steps), "steps_used": len(use)}
    for k in SUMMARY_KEYS:
        out[k] = sum(s[k] for s in use) / len(use)
    out["staleness_max"] = max(s["staleness_max"] for s in use)
    tel_keys = sorted({k for s in use for k in s if k.startswith(("tel_", "adm_cap"))})
    for k in tel_keys:
        xs = [s[k] for s in use if k in s]
        out[k] = sum(xs) / len(xs)
    return out


def format_stats(s: dict) -> str:
    return (
        f"step {s['step']:>3} | wall {s['wall_s']:7.1f}s | {s['tokens_per_s']:8.0f} tok/s | "
        f"len p50/p99/max {s['len_p50']:5.0f}/{s['len_p99']:5.0f}/{s['len_max']:5.0f} | "
        f"occupancy {s['slot_occupancy']:.2f} | tail-wait {s['tail_wait_fraction']:.2f} | "
        f"stale {s['staleness_mean']:.2f}/{s['staleness_max']} | "
        f"trunc {s['truncation_rate']:.2f} | reward {s['mean_reward']:.2f}"
        + (f" | preempt {s['tel_preemptions']:.0f}" if "tel_preemptions" in s else "")
        + (f" | kv max {s['tel_kv_usage_max']:.2f}" if "tel_kv_usage_max" in s else "")
        + (f" | cap {s['adm_cap_mean']:.0f}" if "adm_cap_mean" in s else "")
        + (f" | tail tok/s {s['tel_tail_tok_s']:.0f}" if "tel_tail_tok_s" in s else "")
        + (f" | spec acc {s['tel_spec_acceptance']:.2f}" if "tel_spec_acceptance" in s else "")
        + (f" | spec tok/draft {s['tel_spec_tokens_per_draft']:.2f}" if "tel_spec_tokens_per_draft" in s else "")
    )
