1024 sequences, 128 groups, 2 batches; k=8, n=2..4, tail = <= 64 running

| source / policy | tokens/step | tail tokens/step | acceptance | draft rate | mean draft len | makespan reduction |
|---|---|---|---|---|---|---|
| both/always | 2.144 | 2.216 | 0.205 | 0.709 | 7.86 | 56.0% |
| both/tail | 1.084 | 2.264 | 0.187 | 0.056 | 7.99 | 42.5% |
| none/always | 1.000 | 1.000 | 0.000 | 0.000 | 0.00 | 0.0% |
| self/always | 1.643 | 1.950 | 0.165 | 0.490 | 7.94 | 50.8% |
| self/tail | 1.073 | 1.964 | 0.178 | 0.051 | 7.98 | 39.8% |
| sibling/always | 1.983 | 1.756 | 0.186 | 0.673 | 7.85 | 43.9% |
| sibling/tail | 1.072 | 1.929 | 0.143 | 0.063 | 7.99 | 32.5% |
