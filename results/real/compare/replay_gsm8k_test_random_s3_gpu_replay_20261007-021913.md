| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 28.0% | +0.0 pp | 245.2 | 101.8 | 1617.3 | +0.0% | 1283.0 | +0.0% | 1.20 | `results/real/replay_gsm8k_test_random_s3/vanilla/b100_replay/result.json` |
| 100 | cachescout | 31.1% | +3.1 pp | 237.8 | 99.7 | 1668.2 | -3.0% | 1313.1 | +2.3% | 1.20 | `results/real/replay_gsm8k_test_random_s3/cachescout/b100_replay/result.json` |
| 150 | vanilla | 41.7% | +0.0 pp | 104.6 | 80.9 | 485.8 | +0.0% | 1165.3 | +0.0% | 1.25 | `results/real/replay_gsm8k_test_random_s3/vanilla/b150_replay/result.json` |
| 150 | cachescout | 45.4% | +3.7 pp | 97.1 | 84.2 | 373.5 | -7.1% | 1137.7 | -2.4% | 1.26 | `results/real/replay_gsm8k_test_random_s3/cachescout/b150_replay/result.json` |
| 200 | vanilla | 46.7% | +0.0 pp | 89.3 | 79.0 | 205.4 | +0.0% | 1146.8 | +0.0% | 1.26 | `results/real/replay_gsm8k_test_random_s3/vanilla/b200_replay/result.json` |
| 200 | cachescout | 50.8% | +4.1 pp | 87.2 | 78.7 | 201.7 | -2.3% | 1138.8 | -0.7% | 1.26 | `results/real/replay_gsm8k_test_random_s3/cachescout/b200_replay/result.json` |
