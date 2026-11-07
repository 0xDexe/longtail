"""Scheduling logic of both engines, tested on CPU with a fake LLM whose output lengths are heavy-tailed."""

import asyncio
import itertools
import time
from types import SimpleNamespace

from longtail.config import RolloutConfig
from longtail.engine import BarrierRolloutEngine, RefillRolloutEngine, RolloutRequest

TICK = 0.0005  # seconds per fake token


def fake_length(request_id: str) -> int:
    return 200 if int(request_id[1:]) % 16 == 0 else 10  # 1 in 16 is a straggler


class FakeLLM:
    def __init__(self):
        self.max_concurrent = 0
        self._live = 0

    async def generate(self, prompt, params, request_id):
        self._live += 1
        self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            n = fake_length(request_id)
            await asyncio.sleep(n * TICK)
            yield SimpleNamespace(
                finished=True, prompt_token_ids=[0] * 5,
                outputs=[SimpleNamespace(text="\\boxed{1}", token_ids=[0] * n, finish_reason="stop")],
            )
        finally:
            self._live -= 1

    def shutdown(self):
        pass


class FakeMixin:
    def _make_llm(self):
        return FakeLLM()

    def _params(self, seed, max_tokens):
        return None


class FakeBarrier(FakeMixin, BarrierRolloutEngine):
    pass


class FakeRefill(FakeMixin, RefillRolloutEngine):
    pass


CFG = RolloutConfig(prompts_per_step=4, samples_per_prompt=4)  # 16 seqs/step


def make_group_factory():
    counter = itertools.count(1)
    prompt_ids = itertools.count()

    def make_group(version):
        p = next(prompt_ids)
        return [
            RolloutRequest(f"r{next(counter)}", version, p, k, "prompt", 0)
            for k in range(CFG.samples_per_prompt)
        ]

    return make_group


def test_barrier_waits_for_stragglers():
    async def go():
        eng = FakeBarrier(CFG, lambda text, idx: 1.0)
        mg = make_group_factory()
        reqs = [r for _ in range(CFG.prompts_per_step) for r in mg(0)]
        t0 = time.perf_counter()
        traces = await eng.generate_batch(reqs)
        t1 = time.perf_counter()
        return traces, eng.meter.average(t0, t1) / len(reqs), eng.meter.active

    traces, occupancy, active = asyncio.run(go())
    assert len(traces) == 16 and active == 0
    assert all(t.reward == 1.0 for t in traces)
    assert occupancy < 0.35  # mostly idle waiting on the 200-token straggler


def test_refill_keeps_engine_full_and_bounds_capacity():
    async def go():
        eng = FakeRefill(CFG, lambda text, idx: 1.0)
        steps = []
        async for step, traces, t0, t1 in eng.run(4, make_group_factory()):
            steps.append((step, traces, eng.meter.average(t0, t1) / eng.capacity))
        return eng, steps

    eng, steps = asyncio.run(go())
    assert [s for s, _, _ in steps] == [0, 1, 2, 3]
    for step, traces, _ in steps:
        assert len(traces) == CFG.sequences_per_step
        by_prompt = {}
        for t in traces:
            by_prompt.setdefault(t.prompt_idx, []).append(t)
            assert t.delivered_step == step and t.submit_step <= step
        assert all(len(g) == CFG.samples_per_prompt for g in by_prompt.values())  # whole groups only
    assert eng.llm.max_concurrent <= eng.capacity
    assert eng.meter.peak <= eng.capacity
    assert eng.meter.active == 0 and not eng._tasks  # in-flight work cancelled cleanly
    steady = [occ for _, _, occ in steps[1:]]
    assert sum(steady) / len(steady) > 0.6


def test_refill_surfaces_errors_instead_of_hanging():
    class Boom(FakeRefill):
        def _params(self, seed, max_tokens):
            raise RuntimeError("boom")

    async def go():
        eng = Boom(CFG, lambda text, idx: 1.0)
        async for _ in eng.run(1, make_group_factory()):
            pass

    try:
        asyncio.run(asyncio.wait_for(go(), timeout=5))
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
