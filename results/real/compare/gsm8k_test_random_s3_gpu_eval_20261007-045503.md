| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 27.4% | +0.0 pp | 250.2 | 101.4 | 1698.3 | +0.0% | 1304.8 | +0.0% | 1.19 | `results/real/gsm8k_test_random_s3/vanilla/b100_eval/result.json` |
| 100 | cachescout | 25.2% | -2.3 pp | 401.3 | 129.1 | 3375.3 | +60.4% | 1657.8 | +27.1% | 1.20 | `results/real/gsm8k_test_random_s3/cachescout/b100_eval/result.json` |
| 150 | vanilla | 35.0% | +0.0 pp | 144.3 | 90.4 | 1357.2 | +0.0% | 1457.5 | +0.0% | 1.20 | `results/real/gsm8k_test_random_s3/vanilla/b150_eval/result.json` |
| 150 | cachescout | 37.5% | +2.5 pp | 104.6 | 100.5 | 260.2 | -27.5% | 1297.2 | -11.0% | 1.20 | `results/real/gsm8k_test_random_s3/cachescout/b150_eval/result.json` |
| 200 | vanilla | 35.9% | +0.0 pp | 112.8 | 91.3 | 253.6 | +0.0% | 1388.6 | +0.0% | 1.28 | `results/real/gsm8k_test_random_s3/vanilla/b200_eval/result.json` |
| 200 | cachescout | 49.3% | +13.3 pp | 92.2 | 78.2 | 227.3 | -18.2% | 1197.3 | -13.8% | 1.32 | `results/real/gsm8k_test_random_s3/cachescout/b200_eval/result.json` |
