| run | mode | status | step time (s) | tok/s | slot occupancy | tail-wait | staleness mean/max | reward | speedup | preemptions/step | KV max | spec acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| barrier-cap256 | barrier | done | 406.0 | 2620 | 0.28 | 0.43 | 0.00/0 | 0.920 | 1.00x | 310 | 0.95 | - |
| barrier-cap256-ngram | barrier | done | 398.4 | 2663 | 0.40 | 0.24 | 0.00/0 | 0.919 | 1.02x | 50 | 0.92 | 0.083 |
| refill-cap384 | refill | done | 406.5 | 2458 | 0.99 | 0.21 | 0.87/3 | 0.918 | 1.00x | 5604 | 1.00 | - |
| refill-cap384-ngram | refill | done | 426.9 | 2388 | 0.99 | 0.19 | 0.86/3 | 0.921 | 0.95x | 1770 | 1.00 | 0.084 |

Speedup = step time of the first completed run / this run's step time.
