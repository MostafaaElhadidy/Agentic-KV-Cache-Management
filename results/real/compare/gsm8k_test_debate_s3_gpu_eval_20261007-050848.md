| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 38.4% | +0.0 pp | 152.7 | 85.3 | 1168.2 | +0.0% | 1521.9 | +0.0% | 0.90 | `results/real/gsm8k_test_debate_s3/vanilla/b100_eval/result.json` |
| 100 | cachescout | 47.5% | +9.1 pp | 180.2 | 80.4 | 1195.8 | +18.0% | 1577.0 | +3.6% | 0.93 | `results/real/gsm8k_test_debate_s3/cachescout/b100_eval/result.json` |
| 150 | vanilla | 47.7% | +0.0 pp | 95.8 | 77.5 | 372.9 | +0.0% | 1428.7 | +0.0% | 0.92 | `results/real/gsm8k_test_debate_s3/vanilla/b150_eval/result.json` |
| 150 | cachescout | 55.1% | +7.4 pp | 87.9 | 77.4 | 203.7 | -8.3% | 1452.3 | +1.6% | 0.91 | `results/real/gsm8k_test_debate_s3/cachescout/b150_eval/result.json` |
| 200 | vanilla | 51.9% | +0.0 pp | 91.4 | 79.2 | 265.3 | +0.0% | 1592.3 | +0.0% | 0.93 | `results/real/gsm8k_test_debate_s3/vanilla/b200_eval/result.json` |
| 200 | cachescout | 53.7% | +1.7 pp | 89.2 | 75.2 | 320.5 | -2.4% | 1383.7 | -13.1% | 0.94 | `results/real/gsm8k_test_debate_s3/cachescout/b200_eval/result.json` |
