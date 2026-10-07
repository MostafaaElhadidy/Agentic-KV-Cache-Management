| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median | TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 18.8% | +0.0 pp | 2966.1 | 2946.0 | 6892.4 | +0.0% | 4392.8 | +0.0% | 1.48 | `results/real/gsm8k_test_pipeline_s1/vanilla/b100_eval/result.json` |
| 100 | cachescout | 19.6% | +0.8 pp | 3652.0 | 3882.3 | 7027.1 | +23.1% | 5138.8 | +17.0% | 1.33 | `results/real/gsm8k_test_pipeline_s1/cachescout/b100_eval/result.json` |
| 150 | vanilla | 29.4% | +0.0 pp | 296.8 | 124.6 | 1735.6 | +0.0% | 1743.0 | +0.0% | 1.51 | `results/real/gsm8k_test_pipeline_s1/vanilla/b150_eval/result.json` |
| 150 | cachescout | 36.3% | +6.9 pp | 294.1 | 109.0 | 2021.9 | -0.9% | 1760.6 | +1.0% | 1.60 | `results/real/gsm8k_test_pipeline_s1/cachescout/b150_eval/result.json` |
| 200 | vanilla | 28.1% | +0.0 pp | 390.9 | 112.2 | 2246.9 | +0.0% | 1968.3 | +0.0% | 1.61 | `results/real/gsm8k_test_pipeline_s1/vanilla/b200_eval/result.json` |
| 200 | cachescout | 42.0% | +13.9 pp | 107.9 | 93.7 | 446.9 | -72.4% | 1504.4 | -23.6% | 1.62 | `results/real/gsm8k_test_pipeline_s1/cachescout/b200_eval/result.json` |
