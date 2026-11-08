| run | mode | status | step time (s) | tok/s | slot occupancy | tail-wait | staleness mean/max | reward | speedup |
|---|---|---|---|---|---|---|---|---|---|
| barrier-cap256 | barrier | done | 112.7 | 8017 | 0.46 | 0.19 | 0.00/0 | 0.864 | 1.00x |
| refill-cap128 | refill | done | 117.9 | 7706 | 0.97 | 0.16 | 0.31/1 | 0.873 | 0.96x |
| refill-cap256 | refill | done | 103.7 | 8962 | 0.99 | 0.20 | 0.64/2 | 0.860 | 1.09x |
| refill-cap384 | refill | done | 96.1 | 9697 | 0.99 | 0.22 | 0.93/2 | 0.846 | 1.17x |

Speedup = step time of the first completed run / this run's step time.
