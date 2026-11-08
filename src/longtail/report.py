"""Turn queue directories (results/queue-* or archived data/run*) into tables and charts.

    python -m longtail.report results/queue-A results/queue-B data/run2-capacity-sweep-l40s --out report

Writes to --out:
    summary.md                 one table per queue: mean +/- std over measured steps
    summary.json               the same numbers, machine-readable
    <queue>_throughput.png     tok/s per run with step-to-step spread
    <queue>_preemptions.png    KV preemptions per step per run (if telemetry exists)
    <queue>_timeline.png       sequences decoding and KV-cache usage over time (if telemetry exists)

Charts need matplotlib (`uv pip install matplotlib`); tables don't.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Reference categorical palette (fixed order, light mode) and text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

METRICS = {  # key in steps.jsonl -> (label, format)
    "wall_s": ("step time (s)", ".1f"),
    "tokens_per_s": ("tok/s", ".0f"),
    "tel_tail_tok_s": ("tail tok/s", ".0f"),
    "tel_preemptions": ("preemptions/step", ".0f"),
    "tel_kv_usage_max": ("KV max", ".2f"),
    "adm_cap_mean": ("mean cap", ".0f"),
    "tel_spec_acceptance": ("spec acceptance", ".3f"),
    "mean_reward": ("reward", ".3f"),
}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def setup_label(cfg: dict) -> str:
    parts = [cfg.get("mode", "?")]
    if cfg.get("mode") == "refill":
        if cfg.get("admission") == "adaptive":
            parts.append(f"adaptive<= {cfg.get('max_inflight_seqs') or '?'}")
        else:
            parts.append(f"cap {cfg.get('max_inflight_seqs') or cfg.get('max_num_seqs')}")
    else:
        parts.append(f"cap {cfg.get('max_num_seqs')}")
    if cfg.get("spec_method"):
        spec = cfg["spec_method"]
        if spec == "custom_class":
            spec = f"sibling-drafter[{cfg.get('draft_sources')}, gate {cfg.get('draft_gate')}]"
        parts.append(spec + (" +schedule" if "per_batch_size" in (cfg.get("spec_extra") or "") else ""))
    return ", ".join(parts)


def load_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    steps = _jsonl(run_dir / "steps.jsonl")
    env = json.loads((run_dir / "env.json").read_text()) if (run_dir / "env.json").exists() else {}
    skip = cfg.get("summary_skip_steps", 1)
    used = steps[skip:] or steps
    stats = {}
    for key in METRICS:
        xs = [s[key] for s in used if key in s]
        if xs:
            stats[key] = _mean_std(xs)
    return {
        "name": run_dir.name, "dir": str(run_dir), "config": cfg, "env": env, "setup": setup_label(cfg),
        "steps_used": len(used), "stats": stats, "telemetry": run_dir / "telemetry.jsonl",
    }


def load_queue(qdir: Path) -> dict:
    status = json.loads((qdir / "status.json").read_text()) if (qdir / "status.json").exists() else {}
    order = list(status) or sorted(p.name for p in qdir.iterdir() if p.is_dir())
    runs = [r for name in order if (r := load_run(qdir / name)) and r["stats"]]
    gpu = next((r["env"].get("gpus") for r in runs if r["env"].get("gpus")), None)
    limit = runs[0]["config"].get("max_new_tokens") if runs else None
    return {"name": qdir.name, "dir": str(qdir), "runs": runs, "gpu": (gpu or "").split("\n")[0],
            "max_new_tokens": limit}


def queue_table(q: dict) -> str:
    keys = [k for k in METRICS if any(k in r["stats"] for r in q["runs"])]
    head = ["run", "setup", "steps"] + [METRICS[k][0] for k in keys] + ["speedup"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    base = q["runs"][0]["stats"]["wall_s"][0] if q["runs"] and "wall_s" in q["runs"][0]["stats"] else None
    for r in q["runs"]:
        cells = [r["name"], r["setup"], str(r["steps_used"])]
        for k in keys:
            if k in r["stats"]:
                m, sd = r["stats"][k]
                fmt = METRICS[k][1]
                cells.append(f"{m:{fmt}} ± {sd:{fmt}}" if r["steps_used"] > 1 else f"{m:{fmt}}")
            else:
                cells.append("-")
        w = r["stats"].get("wall_s")
        cells.append(f"{base / w[0]:.2f}x" if base and w else "-")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------- charts

def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _bar_chart(plt, q: dict, key: str, title: str, ylabel: str, out: Path) -> Path | None:
    runs = [r for r in q["runs"] if key in r["stats"]]
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(runs)), 3.6), facecolor=SURFACE)
    means = [r["stats"][key][0] for r in runs]
    sds = [r["stats"][key][1] for r in runs]
    xs = range(len(runs))
    ax.bar(xs, means, width=0.55, color=SERIES[0], yerr=sds, capsize=3,
           error_kw={"ecolor": TEXT_2, "elinewidth": 1})
    top = max(m + s for m, s in zip(means, sds)) or 1
    for x, m in zip(xs, means):
        ax.text(x, m + 0.02 * top, f"{m:,.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(list(xs), [r["name"] for r in runs], rotation=15, ha="right")
    _style(ax, title, ylabel)
    ax.set_ylim(0, top * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def _timeline_chart(plt, q: dict, out: Path) -> Path | None:
    series = []
    for r in q["runs"]:
        rows = _jsonl(Path(r["telemetry"]))
        if len(rows) < 2:
            continue
        run_key = next((k for k in rows[-1] if k.endswith("num_requests_running")), None)
        kv_key = next((k for k in rows[-1] if k.endswith(("kv_cache_usage_perc", "gpu_cache_usage_perc"))), None)
        if not run_key or not kv_key:
            continue
        t0 = rows[0]["t"]
        series.append((r["name"], [(x["t"] - t0) / 60 for x in rows],
                       [x.get(run_key, 0) for x in rows], [x.get(kv_key, 0) for x in rows]))
    if not series:
        return None
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, facecolor=SURFACE)
    for i, (name, t, running, kv) in enumerate(series[: len(SERIES)]):
        a1.plot(t, running, color=SERIES[i], linewidth=1.4, label=name)
        a2.plot(t, kv, color=SERIES[i], linewidth=1.4, label=name)
    _style(a1, f"{q['name']}: sequences decoding", "sequences")
    _style(a2, "KV-cache usage", "fraction of KV cache")
    a1.set_ylim(bottom=0)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("minutes since run start", color=TEXT_2, fontsize=10)
    a1.legend(frameon=False, fontsize=9, labelcolor=TEXT, ncol=min(4, len(series)), loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def make_charts(queues: list[dict], out: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping charts (uv pip install matplotlib)")
        return []
    made = []
    for q in queues:
        limit = f", {q['max_new_tokens'] // 1024}k limit" if q["max_new_tokens"] else ""
        made += [p for p in (
            _bar_chart(plt, q, "tokens_per_s", f"Rollout throughput{limit}", "tokens/s (mean ± sd over steps)",
                       out / f"{q['name']}_throughput.png"),
            _bar_chart(plt, q, "tel_preemptions", f"KV-cache preemptions{limit}", "preemptions per step",
                       out / f"{q['name']}_preemptions.png"),
            _timeline_chart(plt, q, out / f"{q['name']}_timeline.png"),
        ) if p]
    return made


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queues", nargs="+", help="queue directories (results/queue-* or data/run*)")
    ap.add_argument("--out", default="report")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    queues = [q for d in args.queues if (q := load_queue(Path(d)))["runs"]]
    md = ["# longtail-rl results\n"]
    for q in queues:
        md.append(f"## {q['name']}\n")
        md.append(f"GPU: {q['gpu'] or 'unknown'} · output limit: {q['max_new_tokens']} tokens · "
                  f"mean ± sd over measured steps; speedup is step time vs the first run\n")
        md.append(queue_table(q) + "\n")
    (out / "summary.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps(
        [{**q, "runs": [{k: v for k, v in r.items() if k != "telemetry"} for r in q["runs"]]} for q in queues],
        indent=2, default=str))
    charts = make_charts(queues, out)
    print("\n".join(md))
    print(f"-> {out}/summary.md, summary.json, {len(charts)} charts")


if __name__ == "__main__":
    main()
