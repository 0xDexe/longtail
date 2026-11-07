"""Baseline: synchronous-barrier rollouts. Measures how much of each step is lost to stragglers.

Usage:
    python scripts/run_baseline.py --num-steps 5
    python scripts/run_baseline.py --prompts-per-step 8 --num-steps 1 --max-new-tokens 512  # smoke test
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from longtail.config import build_arg_parser, config_from_args
from longtail.data import PromptSampler, build_prompt, load_problems
from longtail.engine import BarrierRolloutEngine, RolloutRequest
from longtail.metrics import format_stats, step_stats
from longtail.reward import correctness_reward


def make_requests(cfg, tokenizer, problems, step: int) -> list[RolloutRequest]:
    reqs = []
    for p in problems:
        prompt = build_prompt(tokenizer, p)
        for k in range(cfg.samples_per_prompt):
            reqs.append(
                RolloutRequest(
                    request_id=f"s{step}-p{p.idx}-k{k}",
                    step=step,
                    prompt_idx=p.idx,
                    sample_idx=k,
                    prompt=prompt,
                    seed=cfg.seed * 1_000_003 + step * 10_007 + p.idx * 31 + k,
                )
            )
    return reqs


async def main() -> None:
    cfg = config_from_args(build_arg_parser(__doc__).parse_args())
    run_dir = Path(cfg.output_dir) / f"baseline-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    problems = load_problems(cfg)
    gold = {p.idx: p.answer for p in problems}
    sampler = PromptSampler(problems, cfg.prompts_per_step, seed=cfg.seed)

    engine = BarrierRolloutEngine(cfg, reward_fn=lambda text, idx: correctness_reward(text, gold[idx]))
    try:
        # Warmup: CUDA graphs, allocator, prefix cache. Not recorded.
        warm = make_requests(cfg, tokenizer, problems[: cfg.warmup_prompts], step=-1)
        await engine.generate_batch(warm, max_tokens=64)

        with (run_dir / "traces.jsonl").open("w") as ftr, (run_dir / "steps.jsonl").open("w") as fst:
            for step in range(cfg.num_steps):
                reqs = make_requests(cfg, tokenizer, sampler.next_batch(), step)
                t0 = time.perf_counter()
                traces = await engine.generate_batch(reqs)
                t1 = time.perf_counter()

                stats = {"step": step, **step_stats(traces, t0, t1)}
                for t in traces:
                    ftr.write(json.dumps({**t.to_dict(), "finish_offset_s": t.finish_t - t0}) + "\n")
                fst.write(json.dumps(stats) + "\n")
                ftr.flush(), fst.flush()
                print(format_stats(step, stats), flush=True)
    finally:
        engine.shutdown()
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
