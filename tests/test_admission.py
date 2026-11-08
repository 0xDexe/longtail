import asyncio

from longtail.admission import AdmissionController
from longtail.config import RolloutConfig
from tests.test_engine_fake import FakeMixin, make_group_factory
from longtail.engine import RefillRolloutEngine


def ctl(cap=64):
    return AdmissionController(cap=cap, cap_min=8, cap_max=256, step=8, kv_target=0.85, kv_high=0.95)


def test_grows_when_kv_has_room_and_cap_binds():
    c = ctl()
    assert c.update(kv=0.5, preemptions_total=0, waiting=0, active=64) == 72
    assert c.update(kv=0.5, preemptions_total=0, waiting=0, active=72) == 80


def test_does_not_grow_when_cap_is_not_the_limit():
    c = ctl()
    assert c.update(kv=0.5, preemptions_total=0, waiting=0, active=20) == 64  # few in flight: no reason to grow
    assert c.update(kv=0.5, preemptions_total=0, waiting=3, active=64) == 64  # vLLM already queueing


def test_cuts_on_preemption_then_holds():
    c = ctl(cap=200)
    c.update(kv=0.9, preemptions_total=10, waiting=0, active=200)  # first reading sets the baseline
    cap = c.update(kv=0.99, preemptions_total=25, waiting=0, active=200)
    assert cap == 136  # int(200 * 0.7) = 140, rounded down to a whole group of 8
    for _ in range(c.hold_ticks):
        assert c.update(kv=0.5, preemptions_total=25, waiting=0, active=136) == 136
    assert c.update(kv=0.5, preemptions_total=25, waiting=0, active=136) == 144


def test_steps_down_near_full_and_respects_bounds():
    c = ctl(cap=16)
    assert c.update(kv=0.97, preemptions_total=0, waiting=0, active=16) == 8
    assert c.update(kv=0.97, preemptions_total=0, waiting=0, active=8) == 8  # never below one group
    c = ctl(cap=256)
    assert c.update(kv=0.1, preemptions_total=0, waiting=0, active=256) == 256  # never above cap_max


class FakeAdaptive(FakeMixin, RefillRolloutEngine):
    pass


def test_adaptive_refill_settles_below_kv_limit():
    """Simulated KV cache holds 40 sequences; above that, vLLM would preempt. The controller
    starts at 8, must grow toward the limit, and must not sit far above it."""
    cfg = RolloutConfig(mode="refill", prompts_per_step=4, samples_per_prompt=4, admission="adaptive",
                        max_inflight_seqs=128, adaptive_start=8, admission_interval=0.002, kv_target=0.85)
    preempt = {"total": 0}

    async def go():
        eng = FakeAdaptive(cfg, lambda text, idx: 1.0)

        def signal():
            active = eng.meter.active
            if active > 40:
                preempt["total"] += active - 40
            return {"kv": min(1.0, active / 40), "preemptions": preempt["total"], "waiting": 0}

        eng.signal_fn = signal
        caps = []
        async for step, traces, t0, t1 in eng.run(12, make_group_factory()):
            assert len(traces) == cfg.sequences_per_step
            caps.append(eng.cap)
        return eng, caps

    eng, caps = asyncio.run(go())
    assert eng.controller.decisions["up"] > 0
    assert max(caps) <= 56  # never far past the 40-sequence KV limit
    assert caps[-1] >= 24  # grew well beyond the starting cap of 8
    assert eng.meter.peak <= 56 + cfg.samples_per_prompt


def test_adaptive_config_validation():
    import pytest

    with pytest.raises(ValueError):
        RolloutConfig(mode="barrier", admission="adaptive").validate()
    with pytest.raises(ValueError):
        RolloutConfig(mode="refill", admission="adaptive", telemetry_interval=0).validate()
    RolloutConfig(mode="refill", admission="adaptive").validate()
