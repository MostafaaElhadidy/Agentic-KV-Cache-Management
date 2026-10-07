| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 20.3% | +0.0 pp | 1882.1 | 1226.0 | 8049.8 | +0.0% | 3321.7 | +0.0% | 1.26 | `results/real/gsm8k_test_pipeline_s2/vanilla/b100_eval/result.json` |
| 100 | cachescout | 20.9% | +0.6 pp | 4599.6 | 4487.8 | 10067.9 | +144.4% | 5925.6 | +78.4% | 1.46 | `results/real/gsm8k_test_pipeline_s2/cachescout/b100_eval/result.json` |
| 150 | vanilla | 31.7% | +0.0 pp | 270.0 | 118.8 | 2051.6 | +0.0% | 1610.7 | +0.0% | 1.68 | `results/real/gsm8k_test_pipeline_s2/vanilla/b150_eval/result.json` |
| 150 | cachescout | 34.5% | +2.8 pp | 688.6 | 369.2 | 3027.6 | +155.0% | 2205.8 | +36.9% | 1.61 | `results/real/gsm8k_test_pipeline_s2/cachescout/b150_eval/result.json` |
| 200 | vanilla | 40.6% | +0.0 pp | 114.1 | 91.9 | 513.9 | +0.0% | 1435.1 | +0.0% | 1.68 | `results/real/gsm8k_test_pipeline_s2/vanilla/b200_eval/result.json` |
| 200 | cachescout | 44.2% | +3.6 pp | 123.0 | 87.4 | 643.0 | +7.8% | 1548.4 | +7.9% | 1.64 | `results/real/gsm8k_test_pipeline_s2/cachescout/b200_eval/result.json` |
