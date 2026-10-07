| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 16.2% | +0.0 pp | 1803.5 | 1714.7 | 6578.4 | +0.0% | 2946.4 | +0.0% | 1.65 | `results/real/gsm8k_test_random_s2/vanilla/b100_eval/result.json` |
| 100 | cachescout | 16.3% | +0.1 pp | 2223.7 | 2249.9 | 5027.7 | +23.3% | 3398.6 | +15.3% | 1.68 | `results/real/gsm8k_test_random_s2/cachescout/b100_eval/result.json` |
| 150 | vanilla | 27.3% | +0.0 pp | 302.4 | 129.7 | 2319.8 | +0.0% | 1515.4 | +0.0% | 1.77 | `results/real/gsm8k_test_random_s2/vanilla/b150_eval/result.json` |
| 150 | cachescout | 25.8% | -1.5 pp | 350.7 | 160.6 | 1628.9 | +15.9% | 1456.8 | -3.9% | 2.04 | `results/real/gsm8k_test_random_s2/cachescout/b150_eval/result.json` |
| 200 | vanilla | 33.0% | +0.0 pp | 123.6 | 100.5 | 535.2 | +0.0% | 1328.6 | +0.0% | 1.80 | `results/real/gsm8k_test_random_s2/vanilla/b200_eval/result.json` |
| 200 | cachescout | 37.7% | +4.7 pp | 141.6 | 98.8 | 647.3 | +14.5% | 1311.2 | -1.3% | 1.75 | `results/real/gsm8k_test_random_s2/cachescout/b200_eval/result.json` |
