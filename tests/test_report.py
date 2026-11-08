import json
import random

import pytest

from longtail import report


def fake_queue(root, name="queue-x"):
    q = root / name
    runs = {"barrier": dict(mode="barrier", max_num_seqs=512, wall=200.0, pre=300),
            "refill-adaptive": dict(mode="refill", admission="adaptive", max_inflight_seqs=512, wall=100.0, pre=2)}
    (q).mkdir(parents=True)
    (q / "status.json").write_text(json.dumps({n: {"state": "done"} for n in runs}))
    rnd = random.Random(0)
    for name, r in runs.items():
        d = q / name
        d.mkdir()
        cfg = {"mode": r["mode"], "max_num_seqs": r.get("max_num_seqs", 512), "max_inflight_seqs": r.get("max_inflight_seqs", 0),
               "admission": r.get("admission", "fixed"), "summary_skip_steps": 1, "max_new_tokens": 16384, "spec_method": ""}
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "env.json").write_text(json.dumps({"gpus": "NVIDIA L40S, 46068 MiB, 610\nNVIDIA L40S"}))
        steps = [{"step": i, "wall_s": r["wall"] * (3 if i == 0 else 1) + rnd.uniform(-5, 5),
                  "tokens_per_s": 1e6 / r["wall"], "tel_preemptions": r["pre"], "mean_reward": 0.9}
                 for i in range(4)]
        if r["mode"] == "refill":
            for s in steps:
                s["adm_cap_mean"] = 300.0
        (d / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps))
        tel = [{"t": 100.0 + i, "num_requests_running": 200 - i, "kv_cache_usage_perc": 0.5} for i in range(30)]
        (d / "telemetry.jsonl").write_text("\n".join(json.dumps(x) for x in tel))
    return q


def test_tables_skip_warmup_and_compute_speedup(tmp_path):
    q = report.load_queue(fake_queue(tmp_path))
    barrier, adaptive = q["runs"]
    assert barrier["steps_used"] == 3  # step 0 (warm-up, 3x slower) excluded
    assert barrier["stats"]["wall_s"][0] == pytest.approx(200, abs=6)
    assert adaptive["setup"] == "refill, adaptive<= 512"
    table = report.queue_table(q)
    assert "2.0" in table.splitlines()[-1].split("|")[-2]  # ~2x speedup vs first run
    assert "mean cap" in table and q["gpu"] == "NVIDIA L40S, 46068 MiB, 610"


def test_main_writes_summary_and_charts(tmp_path):
    pytest.importorskip("matplotlib")
    qdir = fake_queue(tmp_path)
    out = tmp_path / "report"
    report.main([str(qdir), "--out", str(out)])
    assert (out / "summary.md").read_text().count("| barrier |") == 1
    for suffix in ("throughput", "preemptions", "timeline"):
        assert (out / f"queue-x_{suffix}.png").stat().st_size > 5000
