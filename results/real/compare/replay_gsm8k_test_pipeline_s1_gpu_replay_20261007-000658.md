| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 16.9% | +0.0 pp | 3254.5 | 3275.2 | 8040.2 | +0.0% | 4677.6 | +0.0% | 1.44 | `results/real/replay_gsm8k_test_pipeline_s1/vanilla/b100_replay/result.json` |
| 100 | cachescout | 20.8% | +3.9 pp | 3111.5 | 3161.3 | 7454.0 | -4.4% | 4482.6 | -4.2% | 1.47 | `results/real/replay_gsm8k_test_pipeline_s1/cachescout/b100_replay/result.json` |
| 150 | vanilla | 30.7% | +0.0 pp | 488.4 | 135.2 | 2993.7 | +0.0% | 1868.6 | +0.0% | 1.66 | `results/real/replay_gsm8k_test_pipeline_s1/vanilla/b150_replay/result.json` |
| 150 | cachescout | 37.4% | +6.7 pp | 478.4 | 119.5 | 3100.7 | -2.1% | 1849.7 | -1.0% | 1.66 | `results/real/replay_gsm8k_test_pipeline_s1/cachescout/b150_replay/result.json` |
| 200 | vanilla | 36.8% | +0.0 pp | 171.1 | 98.8 | 1161.4 | +0.0% | 1537.8 | +0.0% | 1.66 | `results/real/replay_gsm8k_test_pipeline_s1/vanilla/b200_replay/result.json` |
| 200 | cachescout | 44.9% | +8.1 pp | 165.1 | 96.6 | 1078.6 | -3.5% | 1552.7 | +1.0% | 1.66 | `results/real/replay_gsm8k_test_pipeline_s1/cachescout/b200_replay/result.json` |
