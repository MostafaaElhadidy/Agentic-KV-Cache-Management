| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 20.7% | +0.0 pp | 1545.7 | 990.3 | 6465.5 | +0.0% | 2955.5 | +0.0% | 1.28 | `results/real/replay_gsm8k_test_pipeline_s2/vanilla/b100_replay/result.json` |
| 100 | cachescout | 25.7% | +5.0 pp | 1628.3 | 1003.9 | 6564.6 | +5.3% | 3036.6 | +2.7% | 1.27 | `results/real/replay_gsm8k_test_pipeline_s2/cachescout/b100_replay/result.json` |
| 150 | vanilla | 32.3% | +0.0 pp | 218.3 | 103.9 | 2504.0 | +0.0% | 1661.4 | +0.0% | 1.43 | `results/real/replay_gsm8k_test_pipeline_s2/vanilla/b150_replay/result.json` |
| 150 | cachescout | 42.7% | +10.5 pp | 199.1 | 93.3 | 2358.3 | -8.8% | 1616.4 | -2.7% | 1.44 | `results/real/replay_gsm8k_test_pipeline_s2/cachescout/b150_replay/result.json` |
| 200 | vanilla | 35.7% | +0.0 pp | 107.1 | 97.4 | 330.9 | +0.0% | 1542.0 | +0.0% | 1.46 | `results/real/replay_gsm8k_test_pipeline_s2/vanilla/b200_replay/result.json` |
| 200 | cachescout | 45.9% | +10.2 pp | 101.2 | 86.0 | 397.9 | -5.5% | 1522.9 | -1.2% | 1.47 | `results/real/replay_gsm8k_test_pipeline_s2/cachescout/b200_replay/result.json` |
