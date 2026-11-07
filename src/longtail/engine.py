"""Rollout engines on top of vLLM's async engine.

BarrierRolloutEngine  how most RL frameworks generate today: submit the whole batch,
                      wait for *every* sequence, hand the batch to the trainer.
RefillRolloutEngine   keeps the engine full: as soon as a whole group's worth of slots
                      frees up, a new group starts. A step is emitted once
                      `prompts_per_step` complete groups exist, so long sequences
                      straddle step boundaries instead of blocking them (bounded staleness).

Each sample is its own request (not SamplingParams(n=G)) so every sequence gets its own
finish timestamp; prefix caching keeps the shared prompt cheap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from .config import RolloutConfig
from .metrics import ConcurrencyMeter, RequestTrace

RewardFn = Callable[[str, int], float]  # (completion_text, prompt_idx) -> reward
GroupFactory = Callable[[int], "list[RolloutRequest]"]  # policy_version -> one group of requests


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    submit_step: int  # policy version the sequence is generated under
    prompt_idx: int
    sample_idx: int
    prompt: str
    seed: int


class RolloutEngineBase:
    def __init__(self, cfg: RolloutConfig, reward_fn: RewardFn):
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.meter = ConcurrencyMeter()
        self.llm = self._make_llm()

    # --- vLLM-specific pieces (overridden in tests with a fake engine) ---
    def _make_llm(self):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        args = AsyncEngineArgs(
            model=self.cfg.model,
            dtype=self.cfg.dtype,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_model_len=self.cfg.max_model_len,
            max_num_seqs=self.cfg.max_num_seqs,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            seed=self.cfg.seed,
        )
        return AsyncLLMEngine.from_engine_args(args)

    def _params(self, seed: int, max_tokens: int):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        return SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_tokens,
            seed=seed,
            output_kind=RequestOutputKind.FINAL_ONLY,  # no per-token streaming overhead
        )

    # --- shared machinery ---
    def _on_sequence_done(self) -> None:
        pass

    async def _run_one(self, req: RolloutRequest, max_tokens: int) -> RequestTrace:
        """Caller must have done meter.add(+1) for this request before scheduling it."""
        try:
            submit_t = time.perf_counter()
            final = None
            async for out in self.llm.generate(
                req.prompt, self._params(req.seed, max_tokens), request_id=req.request_id
            ):
                final = out
            finish_t = time.perf_counter()
            if final is None or not final.finished:
                raise RuntimeError(f"{req.request_id} did not finish")
            completion = final.outputs[0]
            return RequestTrace(
                request_id=req.request_id,
                prompt_idx=req.prompt_idx,
                sample_idx=req.sample_idx,
                submit_step=req.submit_step,
                prompt_tokens=len(final.prompt_token_ids or []),
                output_tokens=len(completion.token_ids),
                submit_t=submit_t,
                finish_t=finish_t,
                finish_reason=str(completion.finish_reason),
                reward=self.reward_fn(completion.text, req.prompt_idx),
            )
        finally:
            self.meter.add(-1)
            self._on_sequence_done()

    async def warmup(self, requests: list[RolloutRequest], max_tokens: int = 64) -> None:
        """CUDA graphs, allocator, prefix cache. Not recorded."""
        self.meter.add(len(requests))
        await asyncio.gather(*(self._run_one(r, max_tokens) for r in requests))

    def shutdown(self) -> None:
        self.llm.shutdown()


class BarrierRolloutEngine(RolloutEngineBase):
    async def generate_batch(
        self, requests: list[RolloutRequest], max_tokens: int | None = None
    ) -> list[RequestTrace]:
        """Submit everything at once; return only when the slowest sequence is done."""
        mt = max_tokens or self.cfg.max_new_tokens
        self.meter.add(len(requests))
        return list(await asyncio.gather(*(self._run_one(r, mt) for r in requests)))


class RefillRolloutEngine(RolloutEngineBase):
    def __init__(self, cfg: RolloutConfig, reward_fn: RewardFn):
        super().__init__(cfg, reward_fn)
        self.capacity = cfg.max_inflight_seqs or cfg.sequences_per_step
        self._make_group: GroupFactory | None = None
        self._version = 0
        self._stopping = True
        self._done: asyncio.Queue | None = None
        self._tasks: set[asyncio.Task] = set()

    def _fill(self) -> None:
        if self._stopping or self._make_group is None:
            return
        g = self.cfg.samples_per_prompt
        while self.meter.active + g <= self.capacity:
            reqs = self._make_group(self._version)
            self.meter.add(len(reqs))
            task = asyncio.create_task(self._run_group(reqs))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_group(self, reqs: list[RolloutRequest]) -> None:
        try:
            traces = await asyncio.gather(
                *(self._run_one(r, self.cfg.max_new_tokens) for r in reqs)
            )
            self._done.put_nowait(list(traces))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface failures to run() instead of hanging it
            self._done.put_nowait(e)

    def _on_sequence_done(self) -> None:
        self._fill()

    async def run(
        self, num_steps: int, make_group: GroupFactory
    ) -> AsyncIterator[tuple[int, list[RequestTrace], float, float]]:
        """Yields (step, traces, step_start, step_end). The policy version advances after
        each yield; in the full RL loop this is where the trainer updates the weights."""
        self._make_group = make_group
        self._done = asyncio.Queue()
        self._version = 0
        self._stopping = False
        step_start = time.perf_counter()
        self._fill()
        try:
            for step in range(num_steps):
                groups: list[list[RequestTrace]] = []
                while len(groups) < self.cfg.prompts_per_step:
                    item = await self._done.get()
                    if isinstance(item, Exception):
                        raise item
                    groups.append(item)
                step_end = time.perf_counter()
                traces = [t for grp in groups for t in grp]
                for t in traces:
                    t.delivered_step = step
                yield step, traces, step_start, step_end
                self._version += 1
                step_start = step_end
        finally:
            self._stopping = True
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
