"""Synchronous-barrier rollout engine on top of vLLM's async engine.

This mirrors how most RL frameworks generate rollouts today: submit the whole
batch, wait for *every* sequence to finish, hand the batch to the trainer.
Each sample is its own request (instead of SamplingParams(n=G)) so we get a
per-sequence finish timestamp; prefix caching keeps the shared prompt cheap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind

from .config import RolloutConfig
from .metrics import RequestTrace


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    step: int
    prompt_idx: int
    sample_idx: int
    prompt: str
    seed: int


class BarrierRolloutEngine:
    def __init__(self, cfg: RolloutConfig, reward_fn: Callable[[str, int], float]):
        """reward_fn(completion_text, prompt_idx) -> float"""
        self.cfg = cfg
        self.reward_fn = reward_fn
        args = AsyncEngineArgs(
            model=cfg.model,
            dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            max_num_seqs=cfg.max_num_seqs,
            enable_prefix_caching=cfg.enable_prefix_caching,
            seed=cfg.seed,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    def _params(self, seed: int, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_tokens,
            seed=seed,
            output_kind=RequestOutputKind.FINAL_ONLY,  # no per-token streaming overhead
        )

    async def _run_one(self, req: RolloutRequest, max_tokens: int) -> RequestTrace:
        submit_t = time.perf_counter()
        final = None
        async for out in self.engine.generate(
            req.prompt, self._params(req.seed, max_tokens), request_id=req.request_id
        ):
            final = out
        finish_t = time.perf_counter()
        assert final is not None and final.finished, f"{req.request_id} did not finish"

        completion = final.outputs[0]
        return RequestTrace(
            request_id=req.request_id,
            step=req.step,
            prompt_idx=req.prompt_idx,
            sample_idx=req.sample_idx,
            prompt_tokens=len(final.prompt_token_ids or []),
            output_tokens=len(completion.token_ids),
            submit_t=submit_t,
            finish_t=finish_t,
            finish_reason=str(completion.finish_reason),
            reward=self.reward_fn(completion.text, req.prompt_idx),
        )

    async def generate_batch(
        self, requests: list[RolloutRequest], max_tokens: int | None = None
    ) -> list[RequestTrace]:
        """Submit everything at once; return only when the slowest sequence is done."""
        mt = max_tokens or self.cfg.max_new_tokens
        return list(await asyncio.gather(*(self._run_one(r, mt) for r in requests)))

    def shutdown(self) -> None:
        self.engine.shutdown()
