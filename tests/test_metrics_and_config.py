import pytest

from longtail.config import RolloutConfig, build_arg_parser, config_from_args
from longtail.data import PromptSampler, Problem
from longtail.metrics import RequestTrace, percentile, step_stats


def _trace(i, finish, tokens, reason="stop", reward=1.0):
    return RequestTrace(
        request_id=f"r{i}", step=0, prompt_idx=i, sample_idx=0, prompt_tokens=10,
        output_tokens=tokens, submit_t=0.0, finish_t=finish, finish_reason=reason, reward=reward,
    )


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 90) == pytest.approx(9.0)
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([], 50)


def test_no_stragglers_means_full_occupancy():
    traces = [_trace(i, 10.0, 100) for i in range(8)]
    s = step_stats(traces, 0.0, 10.0)
    assert s["slot_occupancy"] == pytest.approx(1.0)
    assert s["tail_wait_fraction"] == pytest.approx(0.0)


def test_single_straggler_dominates_step():
    # 9 sequences finish at t=1s, one runs to t=10s.
    traces = [_trace(i, 1.0, 100) for i in range(9)] + [_trace(9, 10.0, 1000, reason="length", reward=0.0)]
    s = step_stats(traces, 0.0, 10.0)
    assert s["slot_occupancy"] == pytest.approx((9 * 1 + 10) / (10 * 10))
    assert s["idle_slot_fraction"] == pytest.approx(0.81)
    assert s["t50_fraction"] == pytest.approx(0.1)
    assert s["truncation_rate"] == pytest.approx(0.1)
    assert s["mean_reward"] == pytest.approx(0.9)
    assert s["len_max_over_p50"] == pytest.approx(10.0)
    assert s["tokens_per_s"] == pytest.approx(190.0)


def test_cli_roundtrip():
    args = build_arg_parser().parse_args(
        ["--prompts-per-step", "4", "--temperature", "0.7", "--no-enable-prefix-caching"]
    )
    cfg = config_from_args(args)
    assert cfg.prompts_per_step == 4 and cfg.temperature == 0.7
    assert cfg.enable_prefix_caching is False
    assert cfg.sequences_per_step == 4 * RolloutConfig().samples_per_prompt


def test_sampler_no_repeats_within_epoch():
    problems = [Problem(i, f"q{i}", str(i)) for i in range(10)]
    sampler = PromptSampler(problems, batch_size=3, seed=1)
    seen = [p.idx for _ in range(3) for p in sampler.next_batch()]
    assert len(seen) == len(set(seen)) == 9
