| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 18.1% | +0.0 pp | 2067.4 | 1729.6 | 6215.6 | +0.0% | 3157.9 | +0.0% | 1.47 | `results/real/gsm8k_test_selector_s1/vanilla/b100_eval/result.json` |
| 100 | cachescout | 22.6% | +4.5 pp | 2564.6 | 1937.3 | 8534.7 | +24.0% | 3606.5 | +14.2% | 1.76 | `results/real/gsm8k_test_selector_s1/cachescout/b100_eval/result.json` |
| 150 | vanilla | 29.4% | +0.0 pp | 564.8 | 108.3 | 4075.0 | +0.0% | 1617.3 | +0.0% | 1.75 | `results/real/gsm8k_test_selector_s1/vanilla/b150_eval/result.json` |
| 150 | cachescout | 37.8% | +8.4 pp | 325.5 | 99.9 | 2602.1 | -42.4% | 1565.5 | -3.2% | 1.75 | `results/real/gsm8k_test_selector_s1/cachescout/b150_eval/result.json` |
| 200 | vanilla | 27.3% | +0.0 pp | 561.1 | 114.6 | 3561.5 | +0.0% | 1818.8 | +0.0% | 1.83 | `results/real/gsm8k_test_selector_s1/vanilla/b200_eval/result.json` |
| 200 | cachescout | 39.7% | +12.4 pp | 236.2 | 96.8 | 1784.7 | -57.9% | 1468.2 | -19.3% | 1.84 | `results/real/gsm8k_test_selector_s1/cachescout/b200_eval/result.json` |
