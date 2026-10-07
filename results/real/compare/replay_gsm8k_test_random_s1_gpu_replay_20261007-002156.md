| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 13.9% | +0.0 pp | 4517.3 | 4474.0 | 11718.4 | +0.0% | 5820.1 | +0.0% | 1.27 | `results/real/replay_gsm8k_test_random_s1/vanilla/b100_replay/result.json` |
| 100 | cachescout | 16.7% | +2.8 pp | 3942.4 | 4031.3 | 9103.1 | -12.7% | 5229.7 | -10.1% | 1.38 | `results/real/replay_gsm8k_test_random_s1/cachescout/b100_replay/result.json` |
| 150 | vanilla | 22.2% | +0.0 pp | 500.8 | 251.4 | 2425.7 | +0.0% | 1857.1 | +0.0% | 1.85 | `results/real/replay_gsm8k_test_random_s1/vanilla/b150_replay/result.json` |
| 150 | cachescout | 22.1% | -0.1 pp | 669.6 | 347.0 | 3579.6 | +33.7% | 2015.0 | +8.5% | 1.84 | `results/real/replay_gsm8k_test_random_s1/cachescout/b150_replay/result.json` |
| 200 | vanilla | 30.9% | +0.0 pp | 161.9 | 112.0 | 858.1 | +0.0% | 1503.0 | +0.0% | 1.88 | `results/real/replay_gsm8k_test_random_s1/vanilla/b200_replay/result.json` |
| 200 | cachescout | 34.2% | +3.2 pp | 162.1 | 110.7 | 784.9 | +0.2% | 1463.2 | -2.7% | 1.89 | `results/real/replay_gsm8k_test_random_s1/cachescout/b200_replay/result.json` |
