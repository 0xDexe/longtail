"""GSM8K prompt loading and per-step prompt sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RolloutConfig

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "Put the final numeric answer inside \\boxed{}."
)


@dataclass(frozen=True)
class Problem:
    idx: int
    question: str
    answer: str  # normalized ground-truth answer


def parse_gsm8k_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <answer>'."""
    return solution.split("####")[-1].strip().replace(",", "")


def load_problems(cfg: RolloutConfig) -> list[Problem]:
    from datasets import load_dataset  # heavy import, keep lazy

    ds = load_dataset(cfg.dataset, cfg.dataset_config, split=cfg.split)
    return [
        Problem(idx=i, question=row["question"], answer=parse_gsm8k_answer(row["answer"]))
        for i, row in enumerate(ds)
    ]


def build_prompt(tokenizer, problem: Problem) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class PromptSampler:
    """Deterministic epoch-based sampler: shuffles once per epoch, never repeats within one."""

    def __init__(self, problems: list[Problem], batch_size: int, seed: int = 0):
        if batch_size > len(problems):
            raise ValueError("batch_size larger than dataset")
        self.problems = problems
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self._order: list[int] = []
        self._pos = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self._order = list(range(len(self.problems)))
        self.rng.shuffle(self._order)
        self._pos = 0

    def next_batch(self) -> list[Problem]:
        if self._pos + self.batch_size > len(self._order):
            self._reshuffle()
        ids = self._order[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return [self.problems[i] for i in ids]
