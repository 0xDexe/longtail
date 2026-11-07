"""Run configuration for rollout experiments.

Every field is exposed as a CLI flag (``--prompts-per-step 32``) and as a key in
experiment plan files (``experiments/*.toml``), so a run is reproducible from its
saved ``config.json`` alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
    spec_min_n: int = 2  # n-gram match length range (ngram method and our drafter)
    spec_max_n: int = 4
    # "custom_class" = longtail.drafter.SiblingProposer (needs runner_v2 = 0)
    draft_sources: str = "both"  # self | sibling | both
    draft_gate: int = 0  # draft only while <= this many sequences are decoding (0 = always)
    spec_extra: str = ""  # JSON merged into speculative_config, e.g. a batch-size schedule
    runner_v2: int = -1  # vLLM model runner: -1 = vLLM default, 0 = V1, 1 = V2

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
        if self.draft_sources not in ("self", "sibling", "both"):
            raise ValueError("draft_sources must be self|sibling|both")
        if self.spec_method == "custom_class" and self.runner_v2 == 1:
            raise ValueError("custom_class drafters need vLLM's V1 runner (runner_v2 = 0)")
        if self.runner_v2 not in (-1, 0, 1):
            raise ValueError("runner_v2 must be -1, 0 or 1")
        if self.spec_extra:
            json.loads(self.spec_extra)  # fail fast on bad JSON

    def speculative_config(self) -> dict | None:
        if not self.spec_method:
            return None
        spec = {"method": self.spec_method, "num_speculative_tokens": self.spec_k}
        if self.spec_method in ("ngram", "ngram_gpu"):
            spec.update(prompt_lookup_min=self.spec_min_n, prompt_lookup_max=self.spec_max_n)
        if self.spec_method == "custom_class":
            spec["model"] = "longtail.drafter.SiblingProposer"
        if self.spec_extra:
            spec.update(json.loads(self.spec_extra))
        return spec

    def engine_env(self) -> dict[str, str]:
        """Environment for the vLLM engine-core process (set before the engine starts)."""
        env = {
            "LONGTAIL_DRAFT_SOURCES": self.draft_sources,
            "LONGTAIL_DRAFT_GATE": str(self.draft_gate),
            "LONGTAIL_DRAFT_MIN_N": str(self.spec_min_n),
            "LONGTAIL_DRAFT_MAX_N": str(self.spec_max_n),
        }
        if self.runner_v2 in (0, 1):
            env["VLLM_USE_V2_MODEL_RUNNER"] = str(self.runner_v2)
        return env

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
