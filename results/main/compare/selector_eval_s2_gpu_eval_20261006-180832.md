| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 58.2% | +0.0 pp | 81.7 | 73.6 | 252.3 | +0.0% | 397.2 | +0.0% | 5.22 | `results/main/selector_eval_s2/vanilla/b100_eval/result.json` |
| 100 | eviction_only | 60.5% | +2.4 pp | 80.4 | 73.4 | 256.3 | -1.6% | 396.5 | -0.2% | 5.22 | `results/main/selector_eval_s2/eviction_only/b100_eval/result.json` |
| 100 | cachescout | 60.7% | +2.6 pp | 76.7 | 70.7 | 209.2 | -6.1% | 383.5 | -3.4% | 5.23 | `results/main/selector_eval_s2/cachescout/b100_eval/result.json` |
| 150 | vanilla | 67.3% | +0.0 pp | 70.5 | 67.5 | 120.4 | +0.0% | 374.6 | +0.0% | 5.23 | `results/main/selector_eval_s2/vanilla/b150_eval/result.json` |
| 150 | eviction_only | 69.2% | +1.9 pp | 70.5 | 68.0 | 113.4 | -0.0% | 372.8 | -0.5% | 5.22 | `results/main/selector_eval_s2/eviction_only/b150_eval/result.json` |
| 150 | cachescout | 69.1% | +1.8 pp | 71.3 | 68.6 | 124.3 | +1.2% | 379.8 | +1.4% | 5.23 | `results/main/selector_eval_s2/cachescout/b150_eval/result.json` |
| 200 | vanilla | 72.5% | +0.0 pp | 67.9 | 65.8 | 112.0 | +0.0% | 367.6 | +0.0% | 5.23 | `results/main/selector_eval_s2/vanilla/b200_eval/result.json` |
| 200 | eviction_only | 73.2% | +0.7 pp | 69.4 | 67.0 | 112.2 | +2.1% | 376.4 | +2.4% | 5.23 | `results/main/selector_eval_s2/eviction_only/b200_eval/result.json` |
| 200 | cachescout | 73.3% | +0.8 pp | 67.7 | 64.6 | 122.6 | -0.3% | 366.6 | -0.3% | 5.24 | `results/main/selector_eval_s2/cachescout/b200_eval/result.json` |
