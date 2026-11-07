# Archived results

Every number in the write-up comes from one of these directories. Each run directory holds
`config.json` (exact settings), `env.json` (GPU, driver, package versions, git commit),
`steps.jsonl` (per-step metrics), `traces.jsonl` (per-sequence timing, length, reward),
`summary.json` and `run.log`. Each queue directory adds `plan.toml`, `status.json`,
`queue.log`, `comparison.md` and `job.log` (the scheduler's output file).

Raw `tokens.jsonl` files (50-100 MB) are not committed; they stay on cluster storage at
`/projectnb/mlresearch/amartya/longtail/results/`.

Hardware for all runs: 1x NVIDIA L40S (46 GB), BU SCC, vLLM 0.33.0, Qwen/Qwen3-1.7B (bf16),
GSM8K train split, 64 prompts x 8 samples per step, temperature 1.0, top_p 1.0.

| Dir | Job | Date | What it shows |
|---|---|---|---|
| `run1-barrier-vs-refill-l40s` | 8490351 | 2026-11-06 | 4k limit, 512 in flight: refill fills slots (occupancy 0.47 -> 0.99) but throughput is flat (3,935 vs 3,909 tok/s) |
| `run2-capacity-sweep-l40s` | 8492374 | 2026-11-06 | Capping in-flight sequences doubles throughput (barrier cap 256: 8,017 tok/s); refill cap 384 is best at 96.1 s/step, 2.41x vs run1 barrier |
| `run3-collect-16k-l40s` | 8494471 | 2026-11-07 | 16k limit, 1,024 rollouts recorded: occupancy 0.30, tail is 45% of each step; specsim tables for sibling-draft speculation |
| `run4-spec-validate-16k-l40s` | 8503433 | 2026-11-07 | 16k limit: vLLM n-gram accepts ~0.65 tokens/step vs specsim's 0.643 (simulator validated); always-on n-gram cuts tail-wait 0.43 -> 0.24 but step time ~unchanged (1.02x); refill cap 384 hits 5,604 preemptions/step |
