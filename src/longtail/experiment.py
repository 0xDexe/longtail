"""Run one rollout experiment (barrier or refill) and write its results.

    python -m longtail.experiment --mode barrier --num-steps 5
    python -m longtail.experiment --mode refill --run-dir results/my-run

Writes to the run directory:
    config.json   resolved configuration
    env.json      host, GPU, package versions, git commit
    run.log       console output (full capture when launched via longtail.queue)
    steps.jsonl   per-step metrics
    traces.jsonl  per-sequence timing, length, reward, staleness
    summary.json  steady-state averages (first `summary_skip_steps` excluded)
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
from pathlib import Path

from .config import RolloutConfig, build_arg_parser, config_from_args
from .data import Problem, PromptSampler, build_prompt, load_problems
from .engine import BarrierRolloutEngine, RefillRolloutEngine, RolloutRequest
from .metrics import RequestTrace, format_stats, step_stats, summarize
from .reward import correctness_reward
from .runlog import capture_env, tee_output
from .telemetry import Telemetry


class StepWriter:
    def __init__(self, run_dir: Path, save_tokens: bool = False):
        self.traces_f = (run_dir / "traces.jsonl").open("w")
        self.steps_f = (run_dir / "steps.jsonl").open("w")
        self.tokens_f = (run_dir / "tokens.jsonl").open("w") if save_tokens else None
        self.steps: list[dict] = []

    def record(self, traces: list[RequestTrace], step: int, t0: float, t1: float,
               avg_concurrency: float, capacity: int, extra: dict | None = None) -> dict:
        stats = {**step_stats(traces, step, t0, t1, avg_concurrency, capacity), **(extra or {})}
        for t in traces:
            self.traces_f.write(json.dumps(t.to_dict()) + "\n")
            if self.tokens_f is not None:
                self.tokens_f.write(json.dumps(t.tokens_dict()) + "\n")
        self.steps_f.write(json.dumps(stats) + "\n")
        self.traces_f.flush()
        self.steps_f.flush()
        self.steps.append(stats)
        print(format_stats(stats), flush=True)
        return stats

    def close(self) -> None:
        self.traces_f.close()
        self.steps_f.close()
        if self.tokens_f is not None:
            self.tokens_f.close()


async def run_experiment(cfg: RolloutConfig, run_dir: Path) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    problems = load_problems(cfg)
    gold = {p.idx: p.answer for p in problems}
    sampler = PromptSampler(problems, cfg.prompts_per_step, seed=cfg.seed)
    counter = itertools.count()

    def make_group(problem: Problem, version: int) -> list[RolloutRequest]:
        prompt = build_prompt(tokenizer, problem)
        reqs = []
        for k in range(cfg.samples_per_prompt):
            n = next(counter)
            reqs.append(RolloutRequest(
                request_id=f"r{n}", submit_step=version, prompt_idx=problem.idx,
                sample_idx=k, prompt=prompt, seed=cfg.seed * 1_000_003 + n,
            ))
        return reqs

    def reward_fn(text: str, idx: int) -> float:
        return correctness_reward(text, gold[idx])

    warm = [r for p in problems[: cfg.warmup_prompts] for r in make_group(p, -1)]
    writer = StepWriter(run_dir, save_tokens=cfg.save_tokens)
    tel: Telemetry | None = None
    try:
        if cfg.mode == "barrier":
            engine = BarrierRolloutEngine(cfg, reward_fn)
            try:
                await engine.warmup(warm)
                print("warmup done", flush=True)
                tel = Telemetry(cfg.telemetry_interval, run_dir / "telemetry.jsonl")
                tel.start()
                for step in range(cfg.num_steps):
                    reqs = [r for p in sampler.next_batch() for r in make_group(p, step)]
                    t0 = time.perf_counter()
                    traces = await engine.generate_batch(reqs)
                    t1 = time.perf_counter()
                    for t in traces:
                        t.delivered_step = step
                    writer.record(traces, step, t0, t1, engine.meter.average(t0, t1), len(reqs),
                                  tel.window(t0, t1))
            finally:
                if tel:
                    await tel.stop()
                engine.shutdown()
        else:
            engine = RefillRolloutEngine(cfg, reward_fn)
            try:
                await engine.warmup(warm)
                print("warmup done", flush=True)
                tel = Telemetry(cfg.telemetry_interval, run_dir / "telemetry.jsonl")
                tel.start()
                async for step, traces, t0, t1 in engine.run(
                    cfg.num_steps, lambda v: make_group(sampler.next_problem(), v)
                ):
                    writer.record(traces, step, t0, t1, engine.meter.average(t0, t1), engine.capacity,
                                  tel.window(t0, t1))
            finally:
                if tel:
                    await tel.stop()
                engine.shutdown()
    finally:
        writer.close()

    summary = {"mode": cfg.mode, **summarize(writer.steps, cfg.summary_skip_steps)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary: " + json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = build_arg_parser(__doc__)
    parser.add_argument("--run-dir", default=None, help="default: <output_dir>/<mode>-<timestamp>")
    args = parser.parse_args()
    cfg = config_from_args(args)

    run_dir = Path(args.run_dir or Path(cfg.output_dir) / f"{cfg.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("LONGTAIL_QUEUE"):  # the queue already captures everything
        tee_output(run_dir / "run.log")

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    env = capture_env()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(f"run_dir={run_dir} mode={cfg.mode} gpu={env['gpus']} commit={env['git_commit']}", flush=True)

    asyncio.run(run_experiment(cfg, run_dir))
    print(f"results -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
