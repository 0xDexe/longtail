"""Run configuration for rollout experiments.

Every field is exposed as a CLI flag (``--prompts-per-step 32``) so experiments
are reproducible from the saved ``config.json`` alone.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class RolloutConfig:
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

    # Experiment
    num_steps: int = 5
    warmup_prompts: int = 8
    seed: int = 0
    output_dir: str = "results"

    @property
    def sequences_per_step(self) -> int:
        return self.prompts_per_step * self.samples_per_prompt

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_arg_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for f in dataclasses.fields(RolloutConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            caster = {"int": int, "float": float, "str": str}[str(f.type)]
            parser.add_argument(flag, type=caster, default=f.default)
    return parser


def config_from_args(args: argparse.Namespace) -> RolloutConfig:
    names = {f.name for f in dataclasses.fields(RolloutConfig)}
    return RolloutConfig(**{k: v for k, v in vars(args).items() if k in names})
