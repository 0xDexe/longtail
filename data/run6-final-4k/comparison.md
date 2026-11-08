| run | mode | status | step time (s) | tok/s | slot occupancy | tail-wait | staleness mean/max | reward | speedup | preemptions/step | KV max | spec acceptance | tail tok/s | mean cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| barrier-512 | barrier | done | 222.8 | 4014 | 0.47 | 0.12 | 0.00/0 | 0.876 | 1.00x | 390 | 1.00 | - | 1362 | - |
| refill-fixed384 | refill | done | 222.1 | 3952 | 0.99 | 0.17 | 0.86/2 | 0.878 | 1.00x | 2649 | 1.00 | - | - | - |
| refill-adaptive | refill | done | 215.4 | 4120 | 0.56 | 0.21 | 0.71/2 | 0.875 | 1.03x | 761 | 1.00 | - | - | 89 |

Speedup = step time of the first completed run / this run's step time.
