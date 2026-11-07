import json
import sys
from pathlib import Path

import pytest

from longtail import queue

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_plans_are_valid():
    for plan in (ROOT / "experiments").glob("*.toml"):
        runs = queue.load_plan(plan)
        assert runs and all("mode" in r["params"] for r in runs)


def test_defaults_merge_and_override(tmp_path):
    p = tmp_path / "plan.toml"
    p.write_text('[defaults]\nnum_steps = 3\nmode = "barrier"\n[[run]]\nname = "a"\n[[run]]\nname = "b"\nmode = "refill"\n')
    plan = queue.load_plan(p)
    assert plan[0]["params"] == {"num_steps": 3, "mode": "barrier"}
    assert plan[1]["params"]["mode"] == "refill"


def test_typo_rejected_before_running(tmp_path):
    p = tmp_path / "plan.toml"
    p.write_text('[[run]]\nname = "a"\nprompt_per_step = 3\n')
    with pytest.raises(ValueError, match="unknown keys"):
        queue.load_plan(p)


def test_to_cli():
    assert queue.to_cli({"num_steps": 3, "enable_prefix_caching": False, "mode": "refill"}) == [
        "--num-steps", "3", "--no-enable-prefix-caching", "--mode", "refill"
    ]


def test_queue_continues_after_failure_and_resumes(tmp_path, monkeypatch):
    """Swap the experiment subprocess for a tiny script: run 'a' succeeds, 'b' fails."""
    plan = tmp_path / "plan.toml"
    plan.write_text('[[run]]\nname = "a"\nmode = "barrier"\n[[run]]\nname = "b"\nmode = "refill"\n')
    fake = tmp_path / "fake_exp.py"
    fake.write_text(
        "import json, sys, pathlib\n"
        "d = pathlib.Path(sys.argv[sys.argv.index('--run-dir') + 1])\n"
        "mode = sys.argv[sys.argv.index('--mode') + 1]\n"
        "print('hello from', mode)\n"
        "if mode == 'refill': sys.exit(3)\n"
        "s = dict(mode=mode, wall_s=10.0, tokens_per_s=5.0, slot_occupancy=0.5, tail_wait_fraction=0.1,\n"
        "         staleness_mean=0.0, staleness_max=0, mean_reward=0.8)\n"
        "(d / 'summary.json').write_text(json.dumps(s))\n"
    )
    real_stream = queue._stream

    def fake_stream(cmd, log_path, prefix, qlog):
        return real_stream([sys.executable, str(fake), *cmd[3:]], log_path, prefix, qlog)

    monkeypatch.setattr(queue, "_stream", fake_stream)
    out = tmp_path / "results"
    assert queue.main([str(plan), "--output-dir", str(out)]) == 1

    qdir = next(out.iterdir())
    status = json.loads((qdir / "status.json").read_text())
    assert status["a"]["state"] == "done" and status["b"]["state"] == "failed"
    assert "hello from barrier" in (qdir / "a" / "run.log").read_text()
    assert "| a | barrier | done | 10.0 |" in (qdir / "comparison.md").read_text()

    # resume re-runs only the failed run
    assert queue.main(["--resume", str(qdir)]) == 1
    assert "skip a (already done)" in (qdir / "queue.log").read_text()
