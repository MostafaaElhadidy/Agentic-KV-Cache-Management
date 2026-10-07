| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 19.5% | +0.0 pp | 1304.3 | 363.9 | 6517.5 | +0.0% | 2379.9 | +0.0% | 1.11 | `results/real/gsm8k_test_selector_s3/vanilla/b100_eval/result.json` |
| 100 | cachescout | 34.6% | +15.1 pp | 886.7 | 104.1 | 4402.6 | -32.0% | 1962.7 | -17.5% | 1.06 | `results/real/gsm8k_test_selector_s3/cachescout/b100_eval/result.json` |
| 150 | vanilla | 31.1% | +0.0 pp | 135.0 | 92.9 | 948.1 | +0.0% | 1145.4 | +0.0% | 1.17 | `results/real/gsm8k_test_selector_s3/vanilla/b150_eval/result.json` |
| 150 | cachescout | 32.7% | +1.6 pp | 1419.0 | 112.1 | 6799.8 | +951.3% | 2820.9 | +146.3% | 1.02 | `results/real/gsm8k_test_selector_s3/cachescout/b150_eval/result.json` |
| 200 | vanilla | 34.9% | +0.0 pp | 150.3 | 91.9 | 668.4 | +0.0% | 1259.0 | +0.0% | 1.32 | `results/real/gsm8k_test_selector_s3/vanilla/b200_eval/result.json` |
| 200 | cachescout | 46.3% | +11.4 pp | 123.1 | 89.7 | 533.7 | -18.1% | 1245.4 | -1.1% | 1.26 | `results/real/gsm8k_test_selector_s3/cachescout/b200_eval/result.json` |
