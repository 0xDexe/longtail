"""The online drafter must reproduce the offline simulator exactly, so specsim numbers
predict what the real engine will see."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from longtail.config import RolloutConfig
from longtail.drafter import SiblingProposer
from longtail.specsim import SimConfig, baseline_active, simulate_group


def drive(proposer: SiblingProposer, gens: list[list[int]], ids: list[str]) -> dict:
    """Run sequences through the proposer in lockstep, like barrier decoding in vLLM."""
    t = [0] * len(gens)
    finish = [0] * len(gens)
    out = dict(steps=0, accepted=0, drafted=0)
    width = max(map(len, gens)) + 1
    s = 0
    while any(t[i] < len(g) for i, g in enumerate(gens)):
        live = [i for i, g in enumerate(gens) if t[i] < len(g)]
        tok = np.zeros((len(live), width), dtype=np.int32)
        for row, i in enumerate(live):
            tok[row, : t[i]] = gens[i][: t[i]]
        drafts = proposer.propose_batch(
            None, [ids[i] for i in live], tok, np.array([t[i] for i in live]), [[1]] * len(live)
        )
        for row, i in enumerate(live):
            d, g = drafts[row], gens[i]
            acc = 0
            while acc < len(d) and t[i] + acc < len(g) and d[acc] == g[t[i] + acc]:
                acc += 1
            t[i] += min(acc + 1, len(g) - t[i])
            out["steps"] += 1
            out["accepted"] += acc
            out["drafted"] += len(d)
            if t[i] >= len(g):
                finish[i] = s + 1
        s += 1
    out["makespan"] = max(finish)
    return out


def structured_group(seed: int, g: int = 6) -> list[list[int]]:
    """Siblings that share reasoning chunks at different offsets, plus some self-repetition."""
    rnd = random.Random(seed)
    chunks = [[rnd.randrange(100, 400) for _ in range(rnd.randrange(20, 60))] for _ in range(12)]
    gens = []
    for _ in range(g):
        seq = []
        for _ in range(rnd.randrange(4, 14)):
            seq += rnd.choice(chunks) if rnd.random() < 0.7 else [rnd.randrange(400, 900) for _ in range(15)]
        gens.append(seq)
    return gens


@pytest.mark.parametrize("sources", ["self", "sibling", "both"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_offline_simulator(sources, seed):
    gens = structured_group(seed)
    cfg = SimConfig(k=8, min_n=2, max_n=4, tail_active=10**9)
    sim = simulate_group([[] for _ in gens], gens, sources, "always", cfg, baseline_active(list(map(len, gens))))
    prop = SiblingProposer(k=8, sources=sources, gate=0, min_n=2, max_n=4)
    online = drive(prop, gens, [f"g7-r{i}" for i in range(len(gens))])
    assert online["steps"] == sim["seq_steps"]
    assert online["accepted"] == sim["accepted"]
    assert online["makespan"] == sim["makespan"]


def test_siblings_beat_self_on_shared_reasoning():
    gens = structured_group(3)
    ids = [f"g1-r{i}" for i in range(len(gens))]
    self_only = drive(SiblingProposer(k=8, sources="self", gate=0), gens, ids)
    both = drive(SiblingProposer(k=8, sources="both", gate=0), gens, ids)
    assert both["accepted"] > self_only["accepted"]


def test_gate_blocks_drafts_in_bulk_phase():
    prop = SiblingProposer(k=8, sources="both", gate=2)
    tok = np.array([[1, 2, 3, 1, 2], [1, 2, 3, 1, 2], [1, 2, 3, 1, 2]], dtype=np.int32)
    drafts = prop.propose_batch(None, ["g0-r0", "g0-r1", "g0-r2"], tok, np.array([5, 5, 5]), [[2]] * 3)
    assert drafts == [[], [], []] and prop.stats["gated_steps"] == 1
    drafts = prop.propose_batch(None, ["g0-r0", "g0-r1"], tok[:2], np.array([5, 5]), [[2]] * 2)
    assert drafts[0] == [3, 1, 2]  # "1 2" seen earlier, followed by "3 1 2"


def test_untagged_ids_get_self_drafting_only():
    prop = SiblingProposer(k=4, sources="both", gate=0)
    tok = np.array([[5, 6, 7, 8, 9, 0], [1, 2, 7, 8, 0, 0]], dtype=np.int32)
    drafts = prop.propose_batch(None, ["a", "b"], tok, np.array([5, 4]), [[1], [1]])
    assert drafts == [[], []]  # "7 8" exists in a, but a and b are not siblings


def test_partial_prefill_and_preemption():
    prop = SiblingProposer(k=4, sources="self", gate=0)
    tok = np.array([[1, 2, 1, 2, 0, 0]], dtype=np.int32)
    assert prop.propose_batch(None, ["g0-r0"], tok, np.array([4]), [[]]) == [[]]  # prefilling: no draft
    assert prop.propose_batch(None, ["g0-r0"], tok, np.array([4]), [[2]]) == [[1, 2]]
    prop.propose_batch(None, ["g0-r0"], tok, np.array([2]), [[2]])  # shorter context: mirror resets
    assert prop.seqs["g0-r0"].tokens == [1, 2]


def test_finished_groups_are_evicted():
    prop = SiblingProposer(k=4, sources="both", gate=0, evict_after=10)
    tok = np.array([[1, 2, 3]], dtype=np.int32)
    prop.propose_batch(None, ["g0-r0"], tok, np.array([3]), [[3]])
    for _ in range(600):
        prop.propose_batch(None, [], np.zeros((0, 3), dtype=np.int32), np.array([]), [])
    assert not prop.seqs and not prop.groups


def test_propose_finds_input_batch_by_shape():
    prop = SiblingProposer(k=4, sources="self", gate=0)
    batch = SimpleNamespace(req_ids=["g0-r0"], token_ids_cpu=np.array([[1, 2, 1, 2]], dtype=np.int32),
                            num_tokens_no_spec=np.array([4]))
    assert prop.propose(4, batch, [[2]], None) == [[1, 2]]
    with pytest.raises(TypeError):
        prop.propose(4, [[2]], None)


def test_config_wiring():
    cfg = RolloutConfig(spec_method="custom_class", runner_v2=0, draft_sources="sibling", draft_gate=64,
                        spec_extra='{"foo": 1}')
    cfg.validate()
    spec = cfg.speculative_config()
    assert spec["model"] == "longtail.drafter.SiblingProposer" and spec["foo"] == 1
    env = cfg.engine_env()
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0" and env["LONGTAIL_DRAFT_GATE"] == "64"
    with pytest.raises(ValueError):
        RolloutConfig(spec_method="custom_class", runner_v2=1).validate()
    with pytest.raises(ValueError):
        RolloutConfig(spec_extra="{bad").validate()
