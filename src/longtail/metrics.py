"""Per-request traces and step-level straggler metrics.

Definitions (all per rollout step, t=0 at first submit):

wall_s                 time until the *last* sequence finishes; the trainer waits this long
slot_occupancy         sum(active time of each sequence) / (n_seqs * wall_s).
                       1.0 = every batch slot busy the whole step; low = slots idle waiting on stragglers
idle_slot_fraction     1 - slot_occupancy
tail_wait_fraction     (t_last - t_p90) / wall_s: share of the step spent waiting on the slowest 10%
t50_fraction           t_p50 / wall_s: when half the sequences are already done

Note: slot occupancy is *not* GPU utilization. NVML "utilization" stays near 100%
even at batch size 1; the real cost is that decode at small batch is memory-bound,
so throughput collapses as the batch drains. Later commits measure that directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class RequestTrace:
    request_id: str
    step: int
    prompt_idx: int
    sample_idx: int
    prompt_tokens: int
    output_tokens: int
    submit_t: float
    finish_t: float
    finish_reason: str  # "stop" | "length"
    reward: float

    @property
    def latency_s(self) -> float:
        return self.finish_t - self.submit_t

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def step_stats(traces: list[RequestTrace], step_start: float, step_end: float) -> dict:
    if not traces:
        raise ValueError("no traces")
    n = len(traces)
    wall = step_end - step_start
    finish = [t.finish_t - step_start for t in traces]
    lengths = [float(t.output_tokens) for t in traces]
    active = sum(t.latency_s for t in traces)
    occupancy = active / (n * wall) if wall > 0 else 0.0
    t_p50, t_p90, t_last = percentile(finish, 50), percentile(finish, 90), max(finish)

    return {
        "n_seqs": n,
        "wall_s": wall,
        "output_tokens": int(sum(lengths)),
        "tokens_per_s": sum(lengths) / wall if wall > 0 else 0.0,
        "len_p50": percentile(lengths, 50),
        "len_p90": percentile(lengths, 90),
        "len_p99": percentile(lengths, 99),
        "len_max": max(lengths),
        "len_max_over_p50": max(lengths) / max(percentile(lengths, 50), 1.0),
        "slot_occupancy": occupancy,
        "idle_slot_fraction": 1.0 - occupancy,
        "tail_wait_fraction": (t_last - t_p90) / wall if wall > 0 else 0.0,
        "t50_fraction": t_p50 / wall if wall > 0 else 0.0,
        "truncation_rate": sum(t.finish_reason == "length" for t in traces) / n,
        "mean_reward": sum(t.reward for t in traces) / n,
    }


def format_stats(step: int, s: dict) -> str:
    return (
        f"step {step:>3} | wall {s['wall_s']:7.1f}s | {s['tokens_per_s']:8.0f} tok/s | "
        f"len p50/p99/max {s['len_p50']:5.0f}/{s['len_p99']:5.0f}/{s['len_max']:5.0f} | "
        f"occupancy {s['slot_occupancy']:.2f} | tail-wait {s['tail_wait_fraction']:.2f} | "
        f"trunc {s['truncation_rate']:.2f} | reward {s['mean_reward']:.2f}"
    )
