| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 56.0% | +0.0 pp | 99.3 | 75.8 | 414.0 | +0.0% | 413.8 | +0.0% | 6.52 | `results/main/selector_eval_s3/vanilla/b100_eval/result.json` |
| 100 | eviction_only | 59.7% | +3.7 pp | 94.7 | 72.6 | 426.2 | -4.6% | 400.1 | -3.3% | 6.54 | `results/main/selector_eval_s3/eviction_only/b100_eval/result.json` |
| 100 | cachescout | 59.5% | +3.5 pp | 97.6 | 73.0 | 428.3 | -1.7% | 408.7 | -1.2% | 6.52 | `results/main/selector_eval_s3/cachescout/b100_eval/result.json` |
| 150 | vanilla | 65.5% | +0.0 pp | 73.5 | 69.6 | 159.5 | +0.0% | 377.9 | +0.0% | 6.54 | `results/main/selector_eval_s3/vanilla/b150_eval/result.json` |
| 150 | eviction_only | 67.9% | +2.4 pp | 72.1 | 68.4 | 133.4 | -1.9% | 371.5 | -1.7% | 6.54 | `results/main/selector_eval_s3/eviction_only/b150_eval/result.json` |
| 150 | cachescout | 67.6% | +2.1 pp | 71.9 | 69.1 | 140.0 | -2.2% | 371.1 | -1.8% | 6.53 | `results/main/selector_eval_s3/cachescout/b150_eval/result.json` |
| 200 | vanilla | 70.0% | +0.0 pp | 71.3 | 68.1 | 137.6 | +0.0% | 371.1 | +0.0% | 6.54 | `results/main/selector_eval_s3/vanilla/b200_eval/result.json` |
| 200 | eviction_only | 71.3% | +1.3 pp | 71.3 | 69.2 | 124.0 | -0.0% | 372.2 | +0.3% | 6.54 | `results/main/selector_eval_s3/eviction_only/b200_eval/result.json` |
| 200 | cachescout | 71.2% | +1.2 pp | 71.2 | 68.4 | 142.0 | -0.2% | 372.3 | +0.3% | 6.54 | `results/main/selector_eval_s3/cachescout/b200_eval/result.json` |
