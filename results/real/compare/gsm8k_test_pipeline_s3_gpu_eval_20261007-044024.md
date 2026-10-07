| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 24.6% | +0.0 pp | 1550.6 | 177.9 | 7629.5 | +0.0% | 2991.9 | +0.0% | 0.97 | `results/real/gsm8k_test_pipeline_s3/vanilla/b100_eval/result.json` |
| 100 | cachescout | 34.5% | +9.9 pp | 657.4 | 104.8 | 3659.4 | -57.6% | 1975.4 | -34.0% | 1.09 | `results/real/gsm8k_test_pipeline_s3/cachescout/b100_eval/result.json` |
| 150 | vanilla | 31.6% | +0.0 pp | 103.2 | 97.6 | 286.2 | +0.0% | 1269.9 | +0.0% | 1.20 | `results/real/gsm8k_test_pipeline_s3/vanilla/b150_eval/result.json` |
| 150 | cachescout | 43.7% | +12.1 pp | 189.7 | 88.8 | 2366.2 | +83.8% | 1597.1 | +25.8% | 1.17 | `results/real/gsm8k_test_pipeline_s3/cachescout/b150_eval/result.json` |
| 200 | vanilla | 35.2% | +0.0 pp | 100.9 | 91.5 | 229.8 | +0.0% | 1349.3 | +0.0% | 1.12 | `results/real/gsm8k_test_pipeline_s3/vanilla/b200_eval/result.json` |
| 200 | cachescout | 45.4% | +10.2 pp | 102.5 | 92.0 | 225.0 | +1.6% | 1706.2 | +26.4% | 1.22 | `results/real/gsm8k_test_pipeline_s3/cachescout/b200_eval/result.json` |
