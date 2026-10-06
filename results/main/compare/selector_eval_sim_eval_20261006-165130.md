| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 54.6% | +0.0 pp | 42.4 | 33.4 | 187.1 | +0.0% | 296.4 | +0.0% | 7.73 | `simulator` |
| 100 | continuum | 54.6% | +0.0 pp | 42.4 | 33.4 | 187.1 | +0.0% | 296.4 | +0.0% | 7.73 | `simulator` |
| 100 | warmup_only | 55.9% | +1.3 pp | 42.5 | 32.9 | 190.7 | +0.3% | 296.6 | +0.0% | 7.73 | `simulator` |
| 100 | eviction_only | 57.7% | +3.1 pp | 41.5 | 33.0 | 191.9 | -2.0% | 295.6 | -0.3% | 7.74 | `simulator` |
| 100 | cachescout | 57.9% | +3.3 pp | 42.2 | 33.3 | 204.6 | -0.5% | 296.2 | -0.1% | 7.73 | `simulator` |
| 100 | cachescout_literal | 54.8% | +0.2 pp | 42.7 | 33.4 | 193.7 | +0.8% | 296.8 | +0.1% | 7.73 | `simulator` |
| 100 | no_prediction | 57.7% | +3.0 pp | 41.7 | 32.9 | 188.1 | -1.6% | 295.8 | -0.2% | 7.74 | `simulator` |
| 150 | vanilla | 65.1% | +0.0 pp | 32.1 | 27.0 | 66.8 | +0.0% | 286.2 | +0.0% | 7.73 | `simulator` |
| 150 | continuum | 65.1% | +0.0 pp | 32.1 | 27.0 | 66.8 | +0.0% | 286.2 | +0.0% | 7.73 | `simulator` |
| 150 | warmup_only | 65.8% | +0.7 pp | 31.9 | 27.0 | 66.4 | -0.8% | 286.0 | -0.1% | 7.73 | `simulator` |
| 150 | eviction_only | 67.7% | +2.5 pp | 31.2 | 27.4 | 59.8 | -2.8% | 285.3 | -0.3% | 7.74 | `simulator` |
| 150 | cachescout | 67.7% | +2.6 pp | 31.2 | 27.4 | 59.8 | -2.8% | 285.3 | -0.3% | 7.74 | `simulator` |
| 150 | cachescout_literal | 64.1% | -1.0 pp | 32.5 | 27.3 | 67.5 | +1.1% | 286.5 | +0.1% | 7.74 | `simulator` |
| 150 | no_prediction | 68.2% | +3.1 pp | 31.0 | 27.6 | 56.9 | -3.4% | 285.1 | -0.4% | 7.74 | `simulator` |
| 200 | vanilla | 69.0% | +0.0 pp | 30.8 | 26.1 | 60.8 | +0.0% | 284.9 | +0.0% | 7.74 | `simulator` |
| 200 | continuum | 69.0% | +0.0 pp | 30.8 | 26.1 | 60.8 | +0.0% | 284.9 | +0.0% | 7.74 | `simulator` |
| 200 | warmup_only | 69.3% | +0.3 pp | 30.7 | 26.1 | 60.0 | -0.4% | 284.7 | -0.0% | 7.74 | `simulator` |
| 200 | eviction_only | 70.0% | +1.0 pp | 30.4 | 26.1 | 59.8 | -1.1% | 284.5 | -0.1% | 7.74 | `simulator` |
| 200 | cachescout | 69.9% | +1.0 pp | 30.4 | 26.1 | 59.8 | -1.1% | 284.5 | -0.1% | 7.74 | `simulator` |
| 200 | cachescout_literal | 68.0% | -0.9 pp | 31.1 | 26.2 | 63.4 | +1.0% | 285.2 | +0.1% | 7.74 | `simulator` |
| 200 | no_prediction | 70.5% | +1.6 pp | 30.2 | 26.1 | 56.9 | -1.8% | 284.3 | -0.2% | 7.74 | `simulator` |
