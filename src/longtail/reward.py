"""Verifiable correctness reward for GSM8K-style answers."""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize(ans: str) -> str | None:
    s = ans.strip().replace(",", "").replace("$", "").rstrip(".").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if x.is_integer():
        return str(int(x))
    return str(round(x, 6))


def extract_answer(completion: str) -> str | None:
    """Return the model's final answer, or None if there isn't a usable one.

    A reasoning trace that opened <think> but never closed it was truncated
    mid-thought; we don't scavenge numbers out of it.
    """
    text = completion
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]

    boxed = _BOXED.findall(text)
    if boxed:
        return normalize(boxed[-1])
    nums = _NUMBER.findall(text)
    return normalize(nums[-1]) if nums else None


def correctness_reward(completion: str, gold: str) -> float:
    pred = extract_answer(completion)
    return 1.0 if pred is not None and pred == normalize(gold) else 0.0
