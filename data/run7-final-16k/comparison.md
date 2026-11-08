| run | mode | status | step time (s) | tok/s | slot occupancy | tail-wait | staleness mean/max | reward | speedup | preemptions/step | KV max | spec acceptance | tail tok/s | mean cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| barrier-cap256 | barrier | done | 364.9 | 2725 | 0.29 | 0.46 | 0.00/0 | 0.936 | 1.00x | 48 | 0.90 | - | 758 | - |
| barrier-cap256-ngram-sched | barrier | done | 295.9 | 3355 | 0.34 | 0.34 | 0.00/0 | 0.934 | 1.23x | 68 | 0.90 | 0.120 | 1288 | - |
| refill-fixed384 | refill | done | 340.2 | 2864 | 0.99 | 0.22 | 0.90/4 | 0.936 | 1.07x | 5225 | 1.00 | - | - | - |
| refill-adaptive | refill | done | 335.4 | 2967 | 0.40 | 0.19 | 0.55/3 | 0.931 | 1.09x | 821 | 1.00 | - | - | 62 |

Speedup = step time of the first completed run / this run's step time.
