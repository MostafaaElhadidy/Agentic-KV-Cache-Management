| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 54.0% | +0.0 pp | 96.8 | 74.7 | 357.3 | +0.0% | 408.0 | +0.0% | 7.63 | `results/main/selector_eval/vanilla/b100_eval/result.json` |
| 100 | eviction_only | 55.9% | +1.9 pp | 96.1 | 74.1 | 369.2 | -0.7% | 400.0 | -2.0% | 7.65 | `results/main/selector_eval/eviction_only/b100_eval/result.json` |
| 100 | cachescout | 55.8% | +1.8 pp | 98.1 | 76.5 | 336.5 | +1.4% | 411.9 | +1.0% | 7.63 | `results/main/selector_eval/cachescout/b100_eval/result.json` |
| 100 | cachescout_literal | 54.4% | +0.4 pp | 94.3 | 74.4 | 338.1 | -2.5% | 406.6 | -0.3% | 7.63 | `results/main/selector_eval/cachescout_literal/b100_eval/result.json` |
| 100 | no_prediction | 56.1% | +2.1 pp | 109.7 | 78.9 | 396.2 | +13.4% | 433.7 | +6.3% | 7.61 | `results/main/selector_eval/no_prediction/b100_eval/result.json` |
| 150 | vanilla | 64.1% | +0.0 pp | 74.5 | 69.4 | 154.3 | +0.0% | 380.5 | +0.0% | 7.64 | `results/main/selector_eval/vanilla/b150_eval/result.json` |
| 150 | eviction_only | 66.5% | +2.5 pp | 74.2 | 70.5 | 140.1 | -0.4% | 379.6 | -0.2% | 7.64 | `results/main/selector_eval/eviction_only/b150_eval/result.json` |
| 150 | cachescout | 66.8% | +2.8 pp | 74.1 | 71.0 | 141.4 | -0.5% | 384.6 | +1.1% | 7.63 | `results/main/selector_eval/cachescout/b150_eval/result.json` |
| 150 | cachescout_literal | 64.1% | +0.1 pp | 75.7 | 71.2 | 172.9 | +1.6% | 383.2 | +0.7% | 7.63 | `results/main/selector_eval/cachescout_literal/b150_eval/result.json` |
| 150 | no_prediction | 66.8% | +2.8 pp | 73.7 | 70.4 | 140.3 | -1.0% | 377.8 | -0.7% | 7.64 | `results/main/selector_eval/no_prediction/b150_eval/result.json` |
| 200 | vanilla | 68.0% | +0.0 pp | 72.3 | 68.7 | 140.9 | +0.0% | 376.6 | +0.0% | 7.65 | `results/main/selector_eval/vanilla/b200_eval/result.json` |
| 200 | eviction_only | 69.6% | +1.6 pp | 71.6 | 69.3 | 128.8 | -0.9% | 374.8 | -0.5% | 7.64 | `results/main/selector_eval/eviction_only/b200_eval/result.json` |
| 200 | cachescout | 69.7% | +1.6 pp | 72.1 | 69.1 | 139.3 | -0.3% | 375.9 | -0.2% | 7.65 | `results/main/selector_eval/cachescout/b200_eval/result.json` |
| 200 | cachescout_literal | 67.1% | -1.0 pp | 74.3 | 71.1 | 140.9 | +2.7% | 382.7 | +1.6% | 7.64 | `results/main/selector_eval/cachescout_literal/b200_eval/result.json` |
| 200 | no_prediction | 69.8% | +1.8 pp | 71.9 | 70.1 | 133.0 | -0.5% | 377.4 | +0.2% | 7.64 | `results/main/selector_eval/no_prediction/b200_eval/result.json` |
