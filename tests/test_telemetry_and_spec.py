import pytest

from longtail.config import RolloutConfig
from longtail.metrics import format_stats, summarize
from longtail.specsim import repeat_fraction, repetition_report
from longtail.telemetry import Telemetry


class Counter:
    def __init__(self, name, value):
        self.name, self.value = name, value


class Gauge(Counter):
    pass


class Histogram(Counter):
    pass


def make_fake(state):
    def snap():
        return [
            Counter("vllm:num_preemptions", state["pre"]),
            Counter("vllm:generation_tokens", state["gen"]),
            Counter("vllm:spec_decode_num_draft_tokens", state["drafted"]),
            Counter("vllm:spec_decode_num_accepted_tokens", state["acc"]),
            Counter("vllm:spec_decode_num_drafts", state["drafts"]),
            Gauge("vllm:kv_cache_usage_perc", state["kv"]),
            Gauge("vllm:num_requests_running", state["run"]),
            Gauge("vllm:num_requests_waiting", 0.0),
            Histogram("vllm:e2e_request_latency_seconds", 123.0),  # ignored
        ]
    return snap


def test_telemetry_window_deltas_and_gauges():
    state = dict(pre=5, gen=1000, drafted=0, acc=0, drafts=0, kv=0.2, run=10)
    now = [0.0]
    tel = Telemetry(1.0, snapshot=make_fake(state), clock=lambda: now[0])
    tel.sample()  # t=0
    for t, kv, run in [(1, 0.6, 200), (2, 1.0, 250), (3, 0.8, 100)]:
        now[0] = t
        state.update(pre=state["pre"] + 3, gen=state["gen"] + 500, drafted=state["drafted"] + 80,
                     acc=state["acc"] + 20, drafts=state["drafts"] + 10, kv=kv, run=run)
        tel.sample()
    w = tel.window(0.5, 3.0)
    assert w["tel_preemptions"] == 9
    assert w["tel_gen_tokens_per_s"] == pytest.approx(1500 / 2.5)
    assert w["tel_kv_usage_max"] == pytest.approx(1.0)
    assert w["tel_kv_usage_mean"] == pytest.approx((0.6 + 1.0 + 0.8) / 3)
    assert w["tel_running_mean"] == pytest.approx(550 / 3)
    assert w["tel_spec_acceptance"] == pytest.approx(0.25)
    assert w["tel_spec_tokens_per_draft"] == pytest.approx(2.0)
    assert "e2e_request_latency_seconds" not in tel.kinds


def test_telemetry_disabled_is_harmless():
    tel = Telemetry(0, snapshot=None)
    assert not tel.enabled and tel.window(0, 1) == {}


def test_tel_fields_flow_into_summary_and_format():
    base = {"step": 1, "wall_s": 10.0, "tokens_per_s": 1.0, "avg_concurrency": 1, "slot_occupancy": 1,
            "tail_wait_fraction": 0, "len_p50": 1, "len_p99": 1, "len_max": 1, "truncation_rate": 0,
            "mean_reward": 1, "staleness_mean": 0, "staleness_max": 0}
    steps = [{**base, "tel_preemptions": 100.0}, {**base, "tel_preemptions": 4.0, "tel_kv_usage_max": 0.9},
             {**base, "tel_preemptions": 6.0, "tel_kv_usage_max": 0.7}]
    s = summarize(steps, skip=1)
    assert s["tel_preemptions"] == 5.0 and s["tel_kv_usage_max"] == pytest.approx(0.8)
    assert "preempt 4" in format_stats(steps[1])


def test_speculative_config():
    assert RolloutConfig().speculative_config() is None
    spec = RolloutConfig(spec_method="ngram", spec_k=6, spec_min_n=3, spec_max_n=5).speculative_config()
    assert spec == {"method": "ngram", "num_speculative_tokens": 6, "prompt_lookup_min": 3, "prompt_lookup_max": 5}
    with pytest.raises(ValueError):
        RolloutConfig(spec_method="ngram", spec_k=0).validate()


def test_repetition():
    assert repeat_fraction(list(range(100))) == 0.0
    assert repeat_fraction([1, 2, 3] * 100) > 0.95
    loop = list(range(500)) + [7, 8, 9, 10] * 700
    rep = repetition_report([list(range(i, i + 300)) for i in range(9)] + [loop])
    assert rep["looping_sequences"] == 1
    assert rep["repeat_frac_longest10pct"] > 0.8 > rep["repeat_frac_rest"]


def test_tail_throughput_and_current():
    state = dict(pre=0, gen=0, drafted=0, acc=0, drafts=0, kv=0.5, run=200)
    now = [0.0]
    tel = Telemetry(1.0, snapshot=make_fake(state), clock=lambda: now[0], tail_running=64)
    tel.sample()
    # 2 s of bulk (200 running, 1000 tok/s), then 2 s of tail (30 running, 100 tok/s)
    for t, run, rate in [(1, 200, 1000), (2, 30, 1000), (3, 30, 100), (4, 30, 100)]:
        now[0] = t
        state.update(gen=state["gen"] + rate, run=run)
        tel.sample()
    w = tel.window(0, 4)
    assert w["tel_tail_tok_s"] == pytest.approx(100.0)  # intervals starting with <= 64 running
    assert w["tel_tail_time_frac"] == pytest.approx(0.5)
    cur = tel.current()
    assert cur["kv"] == 0.5 and cur["running"] == 30 and "preemptions" in cur
