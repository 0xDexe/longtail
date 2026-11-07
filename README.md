# longtail-rl

**Straggler-aware rollout generation for RL post-training of reasoning models.**

In RL fine-tuning (GRPO, PPO), most wall-clock time goes to *generating* rollouts, not training.
Reasoning traces have a heavy-tailed length distribution: most finish in a few hundred tokens, a few
run to the token limit. When the trainer waits for the whole batch, the slowest 1% of sequences set
the step time, and the GPU decodes a shrinking, memory-bound batch for most of the step.

This project measures that cost and then removes it.

## Scheduling modes

| Mode | Behavior |
|---|---|
| `barrier` | Submit the whole batch, wait for every sequence (how most RL frameworks work today) |
| `refill` | Keep the engine full: start a new GRPO group whenever a group's worth of slots frees up. A step is emitted once `prompts_per_step` complete groups exist; long sequences carry over into later steps with bounded, measured staleness |

## Metrics (per step, comparable across modes)

| Metric | Meaning |
|---|---|
| `wall_s` | time to deliver one batch of complete groups to the trainer |
| `tokens_per_s` | output tokens delivered / wall_s |
| `slot_occupancy` | time-averaged sequences in flight / capacity (1.0 = engine always full) |
| `tail_wait_fraction` | share of the step spent after 90% of its sequences were done |
| `staleness_mean/max` | policy versions between submitting and delivering a sequence (barrier = 0) |
| `len_p50/p99/max`, `truncation_rate`, `mean_reward` | rollout distribution; checks refill doesn't bias toward short answers |

## Running on a cluster (SGE / BU SCC)

```bash
# login node, once
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                                 # CPU-only: metrics, reward, queue, and both schedulers on a fake LLM
python scripts/prefetch.py             # download weights + dataset before the GPU job

# GPU: one job runs a whole experiment queue
qsub -v PLAN=experiments/smoke.toml jobs/queue.qsub   # ~5 min sanity check
qsub jobs/queue.qsub                                   # experiments/commit2.toml: barrier vs refill
qstat -u $USER
```

Each queue writes `results/queue-<timestamp>-<jobid>/`:

```
plan.toml  status.json  queue.log  comparison.md
<run>/  config.json  env.json  run.log  steps.jsonl  traces.jsonl  summary.json
```

Runs execute as separate processes (GPU memory fully released between runs; one crash doesn't stop
the queue). If the job hits its time limit, `qsub -v RESUME=results/queue-... jobs/queue.qsub`
re-runs only the unfinished runs. Plans are validated before any GPU time is spent: a typo in a
key fails immediately.

To add an experiment, add a `[[run]]` block to a plan. Any `RolloutConfig` field can be set.

## Roadmap

1. ✅ Baseline barrier rollouts + straggler metrics
2. ✅ Experiment queue + autologging; continuous-refill scheduler with staleness tracking
3. Throughput-over-time measurement (tokens/s as the batch drains) + plots
4. Partial rollouts: cap per-step generation, pause long traces, resume next step
5. Minimal GRPO trainer + in-flight weight sync to the inference engine; measure sync latency
6. Effect of staleness from 2 and 4 on learning curves
7. Speculative decoding under policy drift: acceptance-rate decay across training steps
