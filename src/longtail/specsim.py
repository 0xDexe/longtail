"""Offline evaluation of sibling-draft speculative decoding on recorded GRPO rollouts.

    python -m longtail.specsim results/queue-.../collect/tokens.jsonl
    python -m longtail.specsim tokens.jsonl --k 8 --min-n 2 --max-n 5 --tail-active 64

Idea
----
GRPO samples G completions of the same prompt. Siblings reason about the same problem and
reuse a lot of phrasing, often at different offsets (one sibling restates the problem
first, another starts computing immediately). A sequence can draft its next k tokens by
finding its last n tokens in a sibling's text and copying what followed, then verify all
k in one forward pass, the same mechanism as vLLM's n-gram (prompt-lookup) speculation,
but with siblings as an extra draft source. No draft model is needed.

Why offline exact-match replay is a valid estimate
--------------------------------------------------
With a deterministic draft token x, speculative sampling accepts x with probability
min(1, p(x)/q(x)) = p(x), the target probability. Plain sampling produces exactly x with
the same probability p(x). Chained over positions, the probability that the recorded
sample matches the first j draft tokens equals the probability that speculative decoding
accepts j tokens. So replaying recorded rollouts and counting exact matches is an unbiased
estimate of acceptance (for pure temperature sampling, top_p = 1, as recorded here).

Simulation model
----------------
Barrier batches: every sequence in a step starts at engine step 0 and each engine step
advances every unfinished sequence by (accepted + 1) tokens (the +1 is the token the
verifier samples itself). A sibling's draftable text at engine step s is whatever it had
generated at the start of s, so siblings are only useful when their *content* is ahead.

Sources: "self"     own prompt + generated text (vLLM's existing n-gram method)
         "sibling"  generated text of the other G-1 samples of the same prompt
         "both"     either; take the longest available continuation
Policies: "always"  speculate every step
          "tail"    speculate only while few sequences are still running in the batch
                    (decode is memory-bound there, so verifying k extra tokens is ~free)

Reported per configuration:
  tokens_per_step       generated tokens / sequence-steps (1.0 = no speculation)
  tail_tokens_per_step  same, restricted to tail engine steps
  acceptance            accepted draft tokens / drafted tokens
  draft_rate            share of sequence-steps that had any draft
  makespan_reduction    1 - (engine steps until the last sequence of the batch finishes)
                            / (longest sequence length). This is the straggler win.

Repetition report: share of each sequence whose 8-gram already appeared earlier in the same
sequence, for the longest 10% vs the rest, plus how many sequences end in a loop (>80% of
their final 2,048 tokens repeated). Tells how much of the tail, and of self-drafting's gain,
comes from looping text.
"""

from __future__ import annotations

import argparse
import bisect
import json
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SOURCES = ("none", "self", "sibling", "both")
POLICIES = ("always", "tail")


@dataclass(frozen=True)
class SimConfig:
    k: int = 8  # max draft tokens per step
    min_n: int = 2
    max_n: int = 4
    tail_active: int = 64  # "tail" = baseline batch has <= this many unfinished sequences


class NgramIndex:
    """Positions of every n-gram (min_n..max_n) in a token list, ascending."""

    def __init__(self, tokens: list[int], min_n: int, max_n: int):
        self.tokens = tokens
        self.pos: dict[int, dict[tuple, list[int]]] = {}
        for n in range(min_n, max_n + 1):
            d: dict[tuple, list[int]] = defaultdict(list)
            for p in range(len(tokens) - n + 1):
                d[tuple(tokens[p : p + n])].append(p)
            self.pos[n] = d

    def lookup(self, key: tuple, avail: int, k: int) -> list[int]:
        """Continuation after the latest occurrence of `key` fully inside tokens[:avail]
        that has at least one known continuation token (so the key itself, if it sits at
        the very end of the available text, is excluded)."""
        plist = self.pos[len(key)].get(key)
        if not plist:
            return []
        i = bisect.bisect_right(plist, avail - len(key) - 1) - 1
        if i < 0:
            return []
        start = plist[i] + len(key)
        return self.tokens[start : min(start + k, avail)]


def repeat_fraction(tokens: list[int], n: int = 8) -> float:
    """Share of n-grams that already occurred earlier in the same token list."""
    if len(tokens) <= n:
        return 0.0
    seen: set[tuple] = set()
    rep = 0
    for i in range(n, len(tokens) + 1):
        g = tuple(tokens[i - n : i])
        rep += g in seen
        seen.add(g)
    return rep / (len(tokens) - n + 1)


def repetition_report(gens: list[list[int]], n: int = 8, loop_window: int = 2048,
                      loop_threshold: float = 0.8) -> dict:
    by_len = sorted(gens, key=len)
    cut = max(1, len(by_len) // 10)
    longest, rest = by_len[-cut:], by_len[:-cut] or by_len
    looping = [g for g in gens if len(g) >= loop_window and repeat_fraction(g[-loop_window:], n) > loop_threshold]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "repeat_frac_longest10pct": mean([repeat_fraction(g, n) for g in longest]),
        "repeat_frac_rest": mean([repeat_fraction(g, n) for g in rest]),
        "min_len_longest10pct": len(longest[0]),
        "looping_sequences": len(looping),
        "looping_token_share": sum(map(len, looping)) / max(1, sum(map(len, gens))),
    }


def baseline_active(lengths: list[int]) -> list[int]:
    """active[s] = number of sequences still running at engine step s without speculation."""
    horizon = max(lengths) if lengths else 0
    counts = [0] * (horizon + 1)
    for n in lengths:
        counts[n] += 1  # finishes after step n-1
    active, alive = [0] * horizon, len(lengths)
    for s in range(horizon):
        alive -= counts[s]
        active[s] = alive
    return active


def simulate_group(
    prompts: list[list[int]],
    gens: list[list[int]],
    source: str,
    policy: str,
    cfg: SimConfig,
    active: list[int],
    self_idx: list[NgramIndex] | None = None,
    sib_idx: list[NgramIndex] | None = None,
) -> dict:
    g = len(gens)
    use_self = source in ("self", "both")
    use_sib = source in ("sibling", "both")
    if use_self and self_idx is None:
        self_idx = [NgramIndex(prompts[i] + gens[i], cfg.min_n, cfg.max_n) for i in range(g)]
    if use_sib and sib_idx is None:
        sib_idx = [NgramIndex(gens[i], cfg.min_n, cfg.max_n) for i in range(g)]
    full = [prompts[i] + gens[i] for i in range(g)]

    t = [0] * g
    finish_step = [0] * g
    out = dict(seq_steps=0, tokens=0, tail_seq_steps=0, tail_tokens=0,
               drafted=0, accepted=0, draft_steps=0)
    s = 0
    while True:
        running = [i for i in range(g) if t[i] < len(gens[i])]
        if not running:
            break
        in_tail = s >= len(active) or active[s] <= cfg.tail_active
        speculate = source != "none" and (policy == "always" or in_tail)
        snap = list(t)
        for i in running:
            draft: list[int] = []
            if speculate:
                ctx_len = len(prompts[i]) + snap[i]
                for n in range(cfg.max_n, cfg.min_n - 1, -1):
                    if ctx_len < n:
                        continue
                    key = tuple(full[i][ctx_len - n : ctx_len])
                    if use_self:
                        c = self_idx[i].lookup(key, ctx_len, cfg.k)
                        if len(c) > len(draft):
                            draft = c
                    if use_sib:
                        for j in range(g):
                            if j != i:
                                c = sib_idx[j].lookup(key, snap[j], cfg.k)
                                if len(c) > len(draft):
                                    draft = c
                    if draft:
                        break
            seq = gens[i]
            acc = 0
            while acc < len(draft) and t[i] + acc < len(seq) and draft[acc] == seq[t[i] + acc]:
                acc += 1
            adv = min(acc + 1, len(seq) - t[i])
            t[i] += adv
            out["seq_steps"] += 1
            out["tokens"] += adv
            out["drafted"] += len(draft)
            out["accepted"] += acc
            out["draft_steps"] += bool(draft)
            if in_tail:
                out["tail_seq_steps"] += 1
                out["tail_tokens"] += adv
            if t[i] >= len(seq):
                finish_step[i] = s + 1
        s += 1
    out["makespan"] = max(finish_step) if finish_step else 0
    return out


def _sim_group_all(args) -> dict:
    """Worker: one group, every (source, policy) config, indices built once."""
    prompts, gens, cfg, active, batch_key = args
    self_idx = [NgramIndex(prompts[i] + gens[i], cfg.min_n, cfg.max_n) for i in range(len(gens))]
    sib_idx = [NgramIndex(gens[i], cfg.min_n, cfg.max_n) for i in range(len(gens))]
    res = {}
    for source in SOURCES:
        for policy in POLICIES if source != "none" else ("always",):
            res[f"{source}/{policy}"] = simulate_group(
                prompts, gens, source, policy, cfg, active, self_idx, sib_idx
            )
    return {"batch": batch_key, "results": res}


def load_groups(path: str | Path) -> dict:
    """{delivered_step: {prompt_idx: [(prompt_ids, gen_ids), ...]}}"""
    batches: dict = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("token_ids") is None:
                continue
            batches[r["delivered_step"]][r["prompt_idx"]].append(
                (r["sample_idx"], r["prompt_token_ids"], r["token_ids"])
            )
    return batches


def run(path: str | Path, cfg: SimConfig, workers: int = 0) -> dict:
    batches = load_groups(path)
    jobs = []
    for step, groups in sorted(batches.items()):
        lengths = [len(gen) for grp in groups.values() for _, _, gen in grp]
        active = baseline_active(lengths)
        for grp in groups.values():
            grp = sorted(grp)
            jobs.append(([p for _, p, _ in grp], [gen for _, _, gen in grp], cfg, active, step))
    if not jobs:
        raise ValueError(f"{path}: no token data (was save_tokens on?)")

    if workers == 1:
        outs = [_sim_group_all(j) for j in jobs]
    else:
        with mp.Pool(workers or None) as pool:
            outs = pool.map(_sim_group_all, jobs, chunksize=1)

    agg: dict[str, dict] = {}
    makespans: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for o in outs:
        for name, r in o["results"].items():
            a = agg.setdefault(name, defaultdict(int))
            for key, v in r.items():
                if key != "makespan":
                    a[key] += v
            makespans[name][o["batch"]] = max(makespans[name][o["batch"]], r["makespan"])

    base = makespans["none/always"]
    summary = {}
    for name, a in agg.items():
        reductions = [1 - makespans[name][b] / base[b] for b in base if base[b]]
        summary[name] = {
            "tokens_per_step": a["tokens"] / a["seq_steps"],
            "tail_tokens_per_step": a["tail_tokens"] / a["tail_seq_steps"] if a["tail_seq_steps"] else 1.0,
            "acceptance": a["accepted"] / a["drafted"] if a["drafted"] else 0.0,
            "draft_rate": a["draft_steps"] / a["seq_steps"],
            "mean_draft_len": a["drafted"] / a["draft_steps"] if a["draft_steps"] else 0.0,
            "makespan_reduction": sum(reductions) / len(reductions) if reductions else 0.0,
            "seq_steps": a["seq_steps"],
        }
    return {
        "config": cfg.__dict__,
        "repetition": repetition_report([gen for j in jobs for gen in j[1]]),
        "n_batches": len(base),
        "n_groups": len(jobs),
        "n_sequences": sum(len(j[1]) for j in jobs),
        "results": summary,
    }


def format_table(report: dict) -> str:
    rows = [
        f"{report['n_sequences']} sequences, {report['n_groups']} groups, {report['n_batches']} batches; "
        f"k={report['config']['k']}, n={report['config']['min_n']}..{report['config']['max_n']}, "
        f"tail = <= {report['config']['tail_active']} running\n",
        "| source / policy | tokens/step | tail tokens/step | acceptance | draft rate | "
        "mean draft len | makespan reduction |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(report["results"].items()):
        rows.append(
            f"| {name} | {r['tokens_per_step']:.3f} | {r['tail_tokens_per_step']:.3f} | "
            f"{r['acceptance']:.3f} | {r['draft_rate']:.3f} | {r['mean_draft_len']:.2f} | "
            f"{r['makespan_reduction'] * 100:.1f}% |"
        )
    rep = report.get("repetition")
    if rep:
        rows.append(
            f"\nRepetition (8-gram seen earlier in same sequence): longest 10% (>= "
            f"{rep['min_len_longest10pct']} tokens) {rep['repeat_frac_longest10pct']:.1%}, rest "
            f"{rep['repeat_frac_rest']:.1%}; {rep['looping_sequences']} sequences end in a loop "
            f"({rep['looping_token_share']:.1%} of all tokens)."
        )
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokens", help="tokens.jsonl from a run with save_tokens = true")
    ap.add_argument("--k", type=int, default=SimConfig.k)
    ap.add_argument("--min-n", type=int, default=SimConfig.min_n)
    ap.add_argument("--max-n", type=int, default=SimConfig.max_n)
    ap.add_argument("--tail-active", type=int, default=SimConfig.tail_active)
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out", default=None, help="default: specsim_k{k}_n{min}-{max}.json next to tokens")
    args = ap.parse_args(argv)

    cfg = SimConfig(k=args.k, min_n=args.min_n, max_n=args.max_n, tail_active=args.tail_active)
    report = run(args.tokens, cfg, args.workers)
    table = format_table(report)
    print(table)
    out = Path(args.out) if args.out else Path(args.tokens).with_name(
        f"specsim_k{cfg.k}_n{cfg.min_n}-{cfg.max_n}_tail{cfg.tail_active}.json"
    )
    out.write_text(json.dumps(report, indent=2))
    out.with_suffix(".md").write_text(table)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
