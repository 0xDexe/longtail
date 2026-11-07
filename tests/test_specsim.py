import json
import random

from longtail.specsim import NgramIndex, SimConfig, baseline_active, run, simulate_group

CFG = SimConfig(k=6, min_n=2, max_n=3, tail_active=10**9)


def test_index_respects_availability():
    idx = NgramIndex([1, 2, 3, 1, 2, 4, 5], 2, 2)
    assert idx.lookup((1, 2), avail=7, k=3) == [4, 5]  # latest occurrence
    assert idx.lookup((1, 2), avail=5, k=3) == [3, 1, 2]  # second (1,2) has no known continuation yet
    assert idx.lookup((1, 2), avail=2, k=3) == []  # continuation not generated yet
    assert idx.lookup((9, 9), avail=7, k=3) == []


def test_baseline_active():
    assert baseline_active([1, 3, 3]) == [3, 2, 2]


def test_no_speculation_is_one_token_per_step():
    rnd = random.Random(0)
    gens = [[rnd.randrange(50) for _ in range(n)] for n in (30, 50, 80)]
    r = simulate_group([[1, 2]] * 3, gens, "none", "always", CFG, baseline_active([30, 50, 80]))
    assert r["tokens"] == r["seq_steps"] == 160 and r["makespan"] == 80


def test_sibling_lead_bounds_the_gain():
    """B is A without A's 40-token preamble: B's content runs 40 tokens ahead. A can draft from
    B until it catches up, then they move in lockstep and nothing is left to draft. So the
    gain equals the content lead, and the straggler finishes with its sibling."""
    rnd = random.Random(1)
    body = [rnd.randrange(1000, 5000) for _ in range(300)]
    preamble = [rnd.randrange(5000, 6000) for _ in range(40)]
    a, b = preamble + body, list(body)
    prompts = [[1, 2, 3]] * 2
    active = baseline_active([len(a), len(b)])
    none = simulate_group(prompts, [a, b], "none", "always", CFG, active)
    self_only = simulate_group(prompts, [a, b], "self", "always", CFG, active)
    sib = simulate_group(prompts, [a, b], "sibling", "always", CFG, active)
    assert self_only["accepted"] == 0  # random text: no self-repetition
    assert none["makespan"] == 340 and sib["makespan"] == 300
    assert sib["accepted"] == 40
    assert sib["tokens"] == none["tokens"]  # same text, fewer steps


def test_finished_sibling_carries_the_straggler():
    """A long preamble means B finishes while A is still mid-body: A then copies from B's
    complete text, large drafts every step. This is the tail case the method targets."""
    rnd = random.Random(4)
    body = [rnd.randrange(1000, 5000) for _ in range(300)]
    preamble = [rnd.randrange(5000, 6000) for _ in range(250)]
    a, b = preamble + body, list(body)
    active = baseline_active([len(a), len(b)])
    none = simulate_group([[1]] * 2, [a, b], "none", "always", CFG, active)
    sib = simulate_group([[1]] * 2, [a, b], "sibling", "always", CFG, active)
    assert none["makespan"] == 550
    # A copies its whole 300-token body at ~k+1 tokens/step (~43 steps after its preamble),
    # finishing before B; the batch now ends when B does.
    assert sib["makespan"] == 300
    assert sib["accepted"] == 250 and sib["draft_steps"] < 50


def test_advance_never_exceeds_k_plus_one():
    seq = [7] * 100  # maximally repetitive
    r = simulate_group([[7, 7]], [seq], "self", "always", CFG, baseline_active([100]))
    assert r["tokens"] == 100
    assert r["seq_steps"] >= 100 / (CFG.k + 1)


def test_tail_policy_off_when_threshold_zero():
    rnd = random.Random(2)
    body = [rnd.randrange(100) for _ in range(120)]
    gens = [body[10:], body]
    cfg = SimConfig(k=6, min_n=2, max_n=3, tail_active=0)
    active = baseline_active([len(g) for g in gens])
    r = simulate_group([[1]] * 2, gens, "sibling", "tail", cfg, active)
    assert r["drafted"] == 0 and r["tokens"] == r["seq_steps"]


def test_run_end_to_end(tmp_path):
    rnd = random.Random(3)
    path = tmp_path / "tokens.jsonl"
    with path.open("w") as f:
        for p in range(3):
            body = [rnd.randrange(100, 900) for _ in range(200)]
            for k in range(4):
                gen = [rnd.randrange(900, 999) for _ in range(rnd.randrange(0, 60))] + body[: rnd.randrange(80, 200)]
                f.write(json.dumps(dict(request_id=f"r{p}{k}", delivered_step=0, submit_step=0, prompt_idx=p,
                                        sample_idx=k, finish_reason="stop", prompt_token_ids=[1, 2, p],
                                        token_ids=gen)) + "\n")
    report = run(path, SimConfig(k=6, min_n=2, max_n=3, tail_active=4), workers=1)
    res = report["results"]
    assert report["n_sequences"] == 12 and report["n_groups"] == 3
    assert res["none/always"]["tokens_per_step"] == 1.0 and res["none/always"]["makespan_reduction"] == 0.0
    assert res["sibling/always"]["tokens_per_step"] > 1.05
    assert res["sibling/always"]["makespan_reduction"] > 0
    assert res["both/always"]["tokens_per_step"] >= res["self/always"]["tokens_per_step"]
