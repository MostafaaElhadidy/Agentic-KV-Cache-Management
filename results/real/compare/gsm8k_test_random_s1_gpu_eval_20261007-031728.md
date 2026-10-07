| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 13.2% | +0.0 pp | 4375.7 | 4318.2 | 11262.9 | +0.0% | 5665.2 | +0.0% | 1.30 | `results/real/gsm8k_test_random_s1/vanilla/b100_eval/result.json` |
| 100 | cachescout | 15.7% | +2.5 pp | 4031.1 | 3380.6 | 10129.0 | -7.9% | 5385.7 | -4.9% | 1.29 | `results/real/gsm8k_test_random_s1/cachescout/b100_eval/result.json` |
| 150 | vanilla | 23.1% | +0.0 pp | 309.1 | 131.1 | 2507.9 | +0.0% | 1684.4 | +0.0% | 1.49 | `results/real/gsm8k_test_random_s1/vanilla/b150_eval/result.json` |
| 150 | cachescout | 26.2% | +3.0 pp | 626.7 | 297.5 | 3216.7 | +102.7% | 2058.1 | +22.2% | 1.60 | `results/real/gsm8k_test_random_s1/cachescout/b150_eval/result.json` |
| 200 | vanilla | 23.6% | +0.0 pp | 489.3 | 171.4 | 3528.0 | +0.0% | 1990.8 | +0.0% | 1.77 | `results/real/gsm8k_test_random_s1/vanilla/b200_eval/result.json` |
| 200 | cachescout | 36.2% | +12.5 pp | 122.6 | 103.2 | 472.1 | -74.9% | 1411.3 | -29.1% | 1.89 | `results/real/gsm8k_test_random_s1/cachescout/b200_eval/result.json` |
