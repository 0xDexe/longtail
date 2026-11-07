"""Run configuration for rollout experiments.

Every field is exposed as a CLI flag (``--prompts-per-step 32``) and as a key in
experiment plan files (``experiments/*.toml``), so a run is reproducible from its
saved ``config.json`` alone.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass

MODES = ("barrier", "refill")


@dataclass
class RolloutConfig:
    # Scheduling strategy: "barrier" (wait for whole batch) or "refill" (keep engine full)
    mode: str = "barrier"

    # Model / engine
    model: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 6144
    max_num_seqs: int = 512
    enable_prefix_caching: bool = True

    # Data
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"

    # Rollout shape (GRPO-style: G samples per prompt)
    prompts_per_step: int = 64
    samples_per_prompt: int = 8
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0

    # Refill mode: max sequences in flight (0 = sequences_per_step, same as barrier)
    max_inflight_seqs: int = 0

    # Speculative decoding, passed to vLLM's speculative_config ("" = off).
    # "ngram" = vLLM's prompt-lookup drafting from each sequence's own context.
    spec_method: str = ""
    spec_k: int = 8  # draft tokens per step
    spec_min_n: int = 2  # n-gram match length range (ngram method)
    spec_max_n: int = 4

    # Engine telemetry: sample vLLM's metrics every N seconds (0 = off)
    telemetry_interval: float = 1.0

    # Experiment
    num_steps: int = 5
    summary_skip_steps: int = 1  # steps excluded from summary.json (warm-up / ramp)
    warmup_prompts: int = 8
    seed: int = 0
    output_dir: str = "results"
    save_tokens: bool = False  # write tokens.jsonl (prompt + output token ids) for offline analysis

    @property
    def sequences_per_step(self) -> int:
        return self.prompts_per_step * self.samples_per_prompt

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        cap = self.max_inflight_seqs or self.sequences_per_step
        if cap < self.samples_per_prompt:
            raise ValueError("max_inflight_seqs must fit at least one full group")
        if self.spec_method and self.spec_k <= 0:
            raise ValueError("spec_k must be > 0 when spec_method is set")
        if self.spec_min_n > self.spec_max_n:
            raise ValueError("spec_min_n must be <= spec_max_n")

    def speculative_config(self) -> dict | None:
        if not self.spec_method:
            return None
        spec = {"method": self.spec_method, "num_speculative_tokens": self.spec_k}
        if self.spec_method == "ngram":
            spec.update(prompt_lookup_min=self.spec_min_n, prompt_lookup_max=self.spec_max_n)
        return spec

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


FIELD_NAMES = {f.name for f in dataclasses.fields(RolloutConfig)}


def build_arg_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(RolloutConfig):
        flag = "--" + f.name.replace("_", "-")
        if str(f.type) == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            caster = {"int": int, "float": float, "str": str}[str(f.type)]
            parser.add_argument(flag, type=caster, default=f.default)
    return parser


def config_from_args(args: argparse.Namespace) -> RolloutConfig:
    cfg = RolloutConfig(**{k: v for k, v in vars(args).items() if k in FIELD_NAMES})
    cfg.validate()
    return cfg
