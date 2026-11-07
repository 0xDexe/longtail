# longtail-rl

**Straggler-aware rollout generation for RL post-training of reasoning models.**

In RL fine-tuning (GRPO, PPO), most wall-clock time goes to *generating* rollouts, not training.
Reasoning traces have a heavy-tailed length distribution: most finish in a few hundred tokens, a few
run to the token limit. Because the trainer waits for the whole batch, the slowest 1% of sequences
set the step time, and the GPU decodes a shrinking, memory-bound batch for most of the step.

This project measures that cost and then removes it.

## Status: commit 1 — baseline measurement

A synchronous-barrier rollout harness (how most RL frameworks generate today) on vLLM, with
per-sequence timing and a verifiable GSM8K reward.

| Metric | Meaning |
|---|---|
| `wall_s` | time until the last sequence finishes (what the trainer waits) |
| `slot_occupancy` | sum of per-sequence active time / (n_seqs × wall_s); low = slots idle waiting on stragglers |
| `tail_wait_fraction` | share of the step spent after 90% of sequences were already done |
| `t50_fraction` | when half the batch is done, as a fraction of the step |
| `len_max_over_p50` | how heavy the length tail is |
| `truncation_rate` | sequences that hit `max_new_tokens` |

## Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                                   # CPU-only unit tests

# smoke test (~1 min on one GPU)
python scripts/run_baseline.py --prompts-per-step 8 --num-steps 1 --max-new-tokens 512

# baseline: 64 prompts x 8 samples = 512 sequences/step
python scripts/run_baseline.py --num-steps 5
```

Outputs go to `results/baseline-<timestamp>/`: `config.json`, `steps.jsonl` (per-step metrics),
`traces.jsonl` (per-sequence timing, length, reward).

## Roadmap

1. ✅ Baseline barrier rollouts + straggler metrics
2. Throughput-over-time measurement (tokens/s as the batch drains) + plots
3. Continuous refill: keep the batch full by starting the next prompts as sequences finish
4. Partial rollouts: cap per-step generation, pause long traces, resume next step
5. Minimal GRPO trainer + in-flight weight sync to the inference engine; measure sync latency
6. Off-policy staleness from 3–5 and its effect on learning curves
7. Speculative decoding under policy drift: acceptance-rate decay across training steps

## Environment

Record the exact versions you ran with (`pip freeze | grep -E "vllm|torch|transformers"`) and the GPU model
in every results write-up; vLLM's async API changes between releases.
