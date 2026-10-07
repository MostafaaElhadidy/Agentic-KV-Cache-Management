| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 54.0% | +0.0 pp | 96.8 | 74.7 | 357.3 | +0.0% | 408.0 | +0.0% | 7.63 | `results/main/selector_eval/vanilla/b100_eval/result.json` |
| 100 | continuum | 54.0% | +0.0 pp | 100.0 | 75.8 | 380.5 | +3.3% | 412.9 | +1.2% | 7.62 | `results/main/selector_eval/continuum/b100_eval/result.json` |
| 100 | warmup_only | 54.9% | +0.8 pp | 95.9 | 74.3 | 352.0 | -0.9% | 408.3 | +0.1% | 7.63 | `results/main/selector_eval/warmup_only/b100_eval/result.json` |
| 100 | eviction_only | 55.9% | +1.9 pp | 96.1 | 74.1 | 369.2 | -0.7% | 400.0 | -2.0% | 7.65 | `results/main/selector_eval/eviction_only/b100_eval/result.json` |
| 100 | cachescout | 55.8% | +1.8 pp | 98.1 | 76.5 | 336.5 | +1.4% | 411.9 | +1.0% | 7.63 | `results/main/selector_eval/cachescout/b100_eval/result.json` |
| 150 | vanilla | 64.1% | +0.0 pp | 74.5 | 69.4 | 154.3 | +0.0% | 380.5 | +0.0% | 7.64 | `results/main/selector_eval/vanilla/b150_eval/result.json` |
| 150 | continuum | 64.2% | +0.1 pp | 74.3 | 70.3 | 146.0 | -0.3% | 384.8 | +1.1% | 7.63 | `results/main/selector_eval/continuum/b150_eval/result.json` |
| 150 | warmup_only | 65.1% | +1.1 pp | 74.0 | 69.9 | 160.2 | -0.7% | 378.9 | -0.4% | 7.64 | `results/main/selector_eval/warmup_only/b150_eval/result.json` |
| 150 | eviction_only | 66.5% | +2.5 pp | 74.2 | 70.5 | 140.1 | -0.4% | 379.6 | -0.2% | 7.64 | `results/main/selector_eval/eviction_only/b150_eval/result.json` |
| 150 | cachescout | 66.8% | +2.8 pp | 74.1 | 71.0 | 141.4 | -0.5% | 384.6 | +1.1% | 7.63 | `results/main/selector_eval/cachescout/b150_eval/result.json` |
| 200 | vanilla | 68.0% | +0.0 pp | 72.3 | 68.7 | 140.9 | +0.0% | 376.6 | +0.0% | 7.65 | `results/main/selector_eval/vanilla/b200_eval/result.json` |
| 200 | continuum | 68.0% | -0.1 pp | 74.7 | 71.1 | 132.7 | +3.4% | 386.2 | +2.6% | 7.64 | `results/main/selector_eval/continuum/b200_eval/result.json` |
| 200 | warmup_only | 68.6% | +0.5 pp | 71.8 | 69.1 | 135.3 | -0.7% | 373.0 | -0.9% | 7.65 | `results/main/selector_eval/warmup_only/b200_eval/result.json` |
| 200 | eviction_only | 69.6% | +1.6 pp | 71.6 | 69.3 | 128.8 | -0.9% | 374.8 | -0.5% | 7.64 | `results/main/selector_eval/eviction_only/b200_eval/result.json` |
| 200 | cachescout | 69.7% | +1.6 pp | 72.1 | 69.1 | 139.3 | -0.3% | 375.9 | -0.2% | 7.65 | `results/main/selector_eval/cachescout/b200_eval/result.json` |
