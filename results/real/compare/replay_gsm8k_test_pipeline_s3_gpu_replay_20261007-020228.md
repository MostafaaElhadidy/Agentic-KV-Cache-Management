| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 24.6% | +0.0 pp | 1526.4 | 175.4 | 7480.0 | +0.0% | 2953.5 | +0.0% | 0.97 | `results/real/replay_gsm8k_test_pipeline_s3/vanilla/b100_replay/result.json` |
| 100 | cachescout | 32.9% | +8.3 pp | 1273.7 | 182.5 | 7164.8 | -16.6% | 2664.1 | -9.8% | 0.99 | `results/real/replay_gsm8k_test_pipeline_s3/cachescout/b100_replay/result.json` |
| 150 | vanilla | 32.9% | +0.0 pp | 343.3 | 100.2 | 2944.2 | +0.0% | 1739.3 | +0.0% | 1.08 | `results/real/replay_gsm8k_test_pipeline_s3/vanilla/b150_replay/result.json` |
| 150 | cachescout | 44.6% | +11.7 pp | 298.7 | 83.3 | 3317.8 | -13.0% | 1725.2 | -0.8% | 1.08 | `results/real/replay_gsm8k_test_pipeline_s3/cachescout/b150_replay/result.json` |
| 200 | vanilla | 37.1% | +0.0 pp | 113.2 | 92.7 | 508.8 | +0.0% | 1493.7 | +0.0% | 1.09 | `results/real/replay_gsm8k_test_pipeline_s3/vanilla/b200_replay/result.json` |
| 200 | cachescout | 48.6% | +11.5 pp | 109.6 | 83.0 | 595.4 | -3.2% | 1473.4 | -1.4% | 1.09 | `results/real/replay_gsm8k_test_pipeline_s3/cachescout/b200_replay/result.json` |
