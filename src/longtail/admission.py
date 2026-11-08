"""KV-aware admission control for the refill scheduler.

A fixed in-flight cap is only right for one output-length regime: cap 384 was best at a 4k
limit and thrashed the KV cache at 16k (5,604 preemptions per step). This controller sets
the cap from vLLM's own signals instead, AIMD-style:

  preemptions since last tick   -> multiplicative decrease (cap *= decrease), then hold
  KV usage >= kv_high           -> step down one group
  KV usage <  kv_target, nothing waiting, and the cap is what limits us
                                -> step up one group

Lowering the cap never aborts running sequences; the scheduler just stops admitting new
groups until in-flight drops below it. Steps are whole groups (samples_per_prompt).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdmissionController:
    cap: int
    cap_min: int
    cap_max: int
    step: int
    kv_target: float = 0.85
    kv_high: float = 0.95
    decrease: float = 0.7
    hold_ticks: int = 4  # after a decrease, ticks to wait before increasing again
    _last_pre: float | None = field(default=None, repr=False)
    _hold: int = field(default=0, repr=False)
    decisions: dict = field(default_factory=lambda: {"up": 0, "down": 0, "cut": 0, "hold": 0})

    def update(self, kv: float, preemptions_total: float, waiting: float, active: int) -> int:
        if self._last_pre is None:
            self._last_pre = preemptions_total
        new_pre = preemptions_total - self._last_pre
        self._last_pre = preemptions_total

        if new_pre > 0:
            self.cap = max(self.cap_min, int(self.cap * self.decrease) // self.step * self.step)
            self._hold = self.hold_ticks
            self.decisions["cut"] += 1
        elif kv >= self.kv_high:
            self.cap = max(self.cap_min, self.cap - self.step)
            self.decisions["down"] += 1
        elif self._hold > 0:
            self._hold -= 1
            self.decisions["hold"] += 1
        elif kv < self.kv_target and waiting <= 0 and active >= self.cap - self.step:
            self.cap = min(self.cap_max, self.cap + self.step)
            self.decisions["up"] += 1
        return self.cap
