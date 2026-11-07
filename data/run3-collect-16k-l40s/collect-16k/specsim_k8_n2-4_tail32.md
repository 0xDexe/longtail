1024 sequences, 128 groups, 2 batches; k=8, n=2..4, tail = <= 32 running

| source / policy | tokens/step | tail tokens/step | acceptance | draft rate | mean draft len | makespan reduction |
|---|---|---|---|---|---|---|
| both/always | 2.144 | 2.486 | 0.205 | 0.709 | 7.86 | 56.0% |
| both/tail | 1.049 | 2.243 | 0.183 | 0.034 | 7.99 | 37.0% |
| none/always | 1.000 | 1.000 | 0.000 | 0.000 | 0.00 | 0.0% |
| self/always | 1.643 | 2.006 | 0.165 | 0.490 | 7.94 | 50.8% |
| self/tail | 1.044 | 1.969 | 0.177 | 0.031 | 7.98 | 34.8% |
| sibling/always | 1.983 | 1.674 | 0.186 | 0.673 | 7.85 | 43.9% |
| sibling/tail | 1.041 | 1.877 | 0.135 | 0.038 | 7.99 | 28.1% |
