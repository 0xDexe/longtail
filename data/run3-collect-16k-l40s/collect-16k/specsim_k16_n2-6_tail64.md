1024 sequences, 128 groups, 2 batches; k=16, n=2..6, tail = <= 64 running

| source / policy | tokens/step | tail tokens/step | acceptance | draft rate | mean draft len | makespan reduction |
|---|---|---|---|---|---|---|
| both/always | 2.235 | 2.303 | 0.115 | 0.695 | 15.45 | 57.6% |
| both/tail | 1.087 | 2.372 | 0.103 | 0.053 | 15.92 | 43.7% |
| none/always | 1.000 | 1.000 | 0.000 | 0.000 | 0.00 | 0.0% |
| self/always | 1.670 | 2.015 | 0.090 | 0.481 | 15.43 | 52.4% |
| self/tail | 1.076 | 2.028 | 0.098 | 0.049 | 15.74 | 41.1% |
| sibling/always | 2.055 | 1.789 | 0.104 | 0.660 | 15.44 | 45.3% |
| sibling/tail | 1.075 | 2.002 | 0.078 | 0.060 | 15.96 | 33.3% |
