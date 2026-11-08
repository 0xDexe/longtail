| run | mode | status | step time (s) | tok/s | slot occupancy | tail-wait | staleness mean/max | reward | speedup | preemptions/step | KV max | spec acceptance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1-nospec | barrier | done | 365.7 | 2730 | 0.31 | 0.41 | 0.00/0 | 0.935 | 1.00x | 127 | 0.91 | - |
| v1-self-gated | barrier | done | 361.5 | 2803 | 0.37 | 0.30 | 0.00/0 | 0.932 | 1.01x | 81 | 0.93 | 0.183 |
| v1-both-gated | barrier | done | 350.8 | 2893 | 0.38 | 0.29 | 0.00/0 | 0.930 | 1.04x | 105 | 0.92 | 0.196 |
| v1-both-always | barrier | done | 335.5 | 3014 | 0.40 | 0.29 | 0.00/0 | 0.930 | 1.09x | 89 | 0.92 | 0.206 |
| v1-suffix | barrier | failed | | | | | | | | | | |
| v2-ngram-sched | barrier | done | 288.9 | 3470 | 0.36 | 0.30 | 0.00/0 | 0.934 | 1.27x | 149 | 0.91 | 0.121 |

Speedup = step time of the first completed run / this run's step time.
