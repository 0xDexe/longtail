import pytest

from longtail.config import RolloutConfig, build_arg_parser, config_from_args
from longtail.data import Problem, PromptSampler
from longtail.metrics import ConcurrencyMeter, RequestTrace, percentile, step_stats, summarize


def _trace(i, finish, tokens, reason="stop", reward=1.0, submit_step=0):
    return RequestTrace(
        request_id=f"r{i}", prompt_idx=i, sample_idx=0, submit_step=submit_step, prompt_tokens=10,
        output_tokens=tokens, submit_t=0.0, finish_t=finish, finish_reason=reason, reward=reward,
    )


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 90) == pytest.approx(9.0)
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([], 50)


def test_concurrency_meter():
    m = ConcurrencyMeter()
    m.add(4, t=0.0)
    m.add(-2, t=1.0)
    m.add(-2, t=3.0)
    assert m.average(0, 4) == pytest.approx((4 * 1 + 2 * 2) / 4)
    assert m.average(1, 3) == pytest.approx(2.0)
    assert m.average(0.5, 1.5) == pytest.approx(3.0)
    assert m.average(5, 6) == 0.0
    assert m.peak == 4 and m.active == 0


def test_single_straggler_dominates_step():
    # 9 sequences finish at t=1s, one runs to t=10s -> avg concurrency (9*1 + 10)/10 = 1.9
    traces = [_trace(i, 1.0, 100) for i in range(9)] + [_trace(9, 10.0, 1000, reason="length", reward=0.0)]
    s = step_stats(traces, step=0, step_start=0.0, step_end=10.0, avg_concurrency=1.9, capacity=10)
    assert s["slot_occupancy"] == pytest.approx(0.19)
    assert s["t50_fraction"] == pytest.approx(0.1)
    assert s["tail_wait_fraction"] == pytest.approx((10 - percentile([1.0] * 9 + [10.0], 90)) / 10)
    assert s["truncation_rate"] == pytest.approx(0.1)
    assert s["mean_reward"] == pytest.approx(0.9)
    assert s["len_max_over_p50"] == pytest.approx(10.0)
    assert s["tokens_per_s"] == pytest.approx(190.0)
    assert s["staleness_mean"] == 0 and s["staleness_max"] == 0


def test_staleness():
    traces = [_trace(0, 1.0, 10, submit_step=3), _trace(1, 1.0, 10, submit_step=1)]
    s = step_stats(traces, step=3, step_start=0.0, step_end=1.0, avg_concurrency=2, capacity=2)
    assert s["staleness_mean"] == pytest.approx(1.0)
    assert s["staleness_max"] == 2


def test_summarize_skips_rampup():
    base = step_stats([_trace(0, 1.0, 10)], 0, 0.0, 1.0, 1, 1)
    steps = [{**base, "wall_s": 100.0}, {**base, "wall_s": 10.0}, {**base, "wall_s": 20.0}]
    s = summarize(steps, skip=1)
    assert s["wall_s"] == pytest.approx(15.0) and s["steps_used"] == 2


def test_cli_roundtrip_and_validation():
    args = build_arg_parser().parse_args(
        ["--mode", "refill", "--prompts-per-step", "4", "--temperature", "0.7", "--no-enable-prefix-caching"]
    )
    cfg = config_from_args(args)
    assert cfg.mode == "refill" and cfg.prompts_per_step == 4 and cfg.temperature == 0.7
    assert cfg.enable_prefix_caching is False
    assert cfg.sequences_per_step == 4 * RolloutConfig().samples_per_prompt
    with pytest.raises(ValueError):
        config_from_args(build_arg_parser().parse_args(["--mode", "async"]))


def test_sampler_no_repeats_within_epoch():
    problems = [Problem(i, f"q{i}", str(i)) for i in range(10)]
    sampler = PromptSampler(problems, batch_size=3, seed=1)
    seen = [p.idx for _ in range(3) for p in sampler.next_batch()]
    assert len(seen) == len(set(seen)) == 9
