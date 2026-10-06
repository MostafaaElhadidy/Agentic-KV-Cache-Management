## Main sweep (selector_eval, seed 1, GPU)

| blocks | system | hit rate | Δ hit (pp) | TTFT mean (ms) | Δ TTFT | TTFT P99 (ms) | latency mean (ms) | Δ latency | throughput (turns/s) | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | vanilla | 54.0% | +0.0 | 96.8 | +0.0% | 357.3 | 408.0 | +0.0% | 7.63 | `results/main/selector_eval/vanilla/b100_eval/result.json` |
| 100 | continuum | 54.0% | +0.0 | 100.0 | +3.3% | 380.5 | 412.9 | +1.2% | 7.62 | `results/main/selector_eval/continuum/b100_eval/result.json` |
| 100 | warmup_only | 54.9% | +0.8 | 95.9 | -0.9% | 352.0 | 408.3 | +0.1% | 7.63 | `results/main/selector_eval/warmup_only/b100_eval/result.json` |
| 100 | eviction_only | 55.9% | +1.9 | 96.1 | -0.7% | 369.2 | 400.0 | -2.0% | 7.65 | `results/main/selector_eval/eviction_only/b100_eval/result.json` |
| 100 | cachescout | 55.8% | +1.8 | 98.1 | +1.4% | 336.5 | 411.9 | +1.0% | 7.63 | `results/main/selector_eval/cachescout/b100_eval/result.json` |
| 100 | cachescout_literal | 54.4% | +0.4 | 94.3 | -2.5% | 338.1 | 406.6 | -0.3% | 7.63 | `results/main/selector_eval/cachescout_literal/b100_eval/result.json` |
| 100 | no_prediction | 56.1% | +2.1 | 109.7 | +13.4% | 396.2 | 433.7 | +6.3% | 7.61 | `results/main/selector_eval/no_prediction/b100_eval/result.json` |
| 150 | vanilla | 64.1% | +0.0 | 74.5 | +0.0% | 154.3 | 380.5 | +0.0% | 7.64 | `results/main/selector_eval/vanilla/b150_eval/result.json` |
| 150 | continuum | 64.2% | +0.1 | 74.3 | -0.3% | 146.0 | 384.8 | +1.1% | 7.63 | `results/main/selector_eval/continuum/b150_eval/result.json` |
| 150 | warmup_only | 65.1% | +1.1 | 74.0 | -0.7% | 160.2 | 378.9 | -0.4% | 7.64 | `results/main/selector_eval/warmup_only/b150_eval/result.json` |
| 150 | eviction_only | 66.5% | +2.5 | 74.2 | -0.4% | 140.1 | 379.6 | -0.2% | 7.64 | `results/main/selector_eval/eviction_only/b150_eval/result.json` |
| 150 | cachescout | 66.8% | +2.8 | 74.1 | -0.5% | 141.4 | 384.6 | +1.1% | 7.63 | `results/main/selector_eval/cachescout/b150_eval/result.json` |
| 150 | cachescout_literal | 64.1% | +0.1 | 75.7 | +1.6% | 172.9 | 383.2 | +0.7% | 7.63 | `results/main/selector_eval/cachescout_literal/b150_eval/result.json` |
| 150 | no_prediction | 66.8% | +2.8 | 73.7 | -1.0% | 140.3 | 377.8 | -0.7% | 7.64 | `results/main/selector_eval/no_prediction/b150_eval/result.json` |
| 200 | vanilla | 68.0% | +0.0 | 72.3 | +0.0% | 140.9 | 376.6 | +0.0% | 7.65 | `results/main/selector_eval/vanilla/b200_eval/result.json` |
| 200 | continuum | 68.0% | -0.1 | 74.7 | +3.4% | 132.7 | 386.2 | +2.6% | 7.64 | `results/main/selector_eval/continuum/b200_eval/result.json` |
| 200 | warmup_only | 68.6% | +0.5 | 71.8 | -0.7% | 135.3 | 373.0 | -0.9% | 7.65 | `results/main/selector_eval/warmup_only/b200_eval/result.json` |
| 200 | eviction_only | 69.6% | +1.6 | 71.6 | -0.9% | 128.8 | 374.8 | -0.5% | 7.64 | `results/main/selector_eval/eviction_only/b200_eval/result.json` |
| 200 | cachescout | 69.7% | +1.6 | 72.1 | -0.3% | 139.3 | 375.9 | -0.2% | 7.65 | `results/main/selector_eval/cachescout/b200_eval/result.json` |
| 200 | cachescout_literal | 67.1% | -1.0 | 74.3 | +2.7% | 140.9 | 382.7 | +1.6% | 7.64 | `results/main/selector_eval/cachescout_literal/b200_eval/result.json` |
| 200 | no_prediction | 69.8% | +1.8 | 71.9 | -0.5% | 133.0 | 377.4 | +0.2% | 7.64 | `results/main/selector_eval/no_prediction/b200_eval/result.json` |

## CacheScout vs vanilla, mean ± std over 3 evaluation seeds (GPU)

| blocks | system | vanilla hit | system hit | Δ hit (pp) | Δ TTFT mean | Δ latency mean | per-seed Δ hit (pp) |
|---|---|---|---|---|---|---|---|
| 100 | eviction_only | 56.1% | 58.7% | +2.7 ± 1.0 | -2.3 ± 2.1% | -1.8 ± 1.6% | +1.9, +2.4, +3.7 |
| 150 | eviction_only | 65.6% | 67.9% | +2.3 ± 0.3 | -0.8 ± 1.0% | -0.8 ± 0.8% | +2.5, +1.9, +2.4 |
| 200 | eviction_only | 70.2% | 71.4% | +1.2 ± 0.5 | +0.4 ± 1.6% | +0.7 ± 1.5% | +1.6, +0.7, +1.3 |
| 100 | cachescout | 56.1% | 58.7% | +2.7 ± 0.9 | -2.1 ± 3.8% | -1.2 ± 2.2% | +1.8, +2.6, +3.5 |
| 150 | cachescout | 65.6% | 67.8% | +2.2 ± 0.5 | -0.5 ± 1.7% | +0.2 ± 1.8% | +2.8, +1.8, +2.1 |
| 200 | cachescout | 70.2% | 71.4% | +1.2 ± 0.4 | -0.3 ± 0.1% | -0.0 ± 0.3% | +1.6, +0.8, +1.2 |

## Coordination topologies (GPU, CacheScout vs vanilla)

| trace | true R (trace) | paper R | blocks | vanilla hit | CacheScout hit | Δ hit (pp) | Δ TTFT | source (CacheScout) |
|---|---|---|---|---|---|---|---|---|
| pipeline_eval | 1.00 | 1.00 | 100 | 52.9% | 66.0% | +13.1 | -8.7% | `results/main/pipeline_eval/cachescout/b100_eval/result.json` |
| pipeline_eval | 1.00 | 1.00 | 150 | 62.4% | 68.0% | +5.7 | -2.1% | `results/main/pipeline_eval/cachescout/b150_eval/result.json` |
| debate_eval | 0.77 | 0.78 | 100 | 64.7% | 66.0% | +1.3 | -0.3% | `results/main/debate_eval/cachescout/b100_eval/result.json` |
| debate_eval | 0.77 | 0.78 | 150 | 73.9% | 74.8% | +0.8 | -3.4% | `results/main/debate_eval/cachescout/b150_eval/result.json` |
| selector_eval | 0.58 | 0.57 | 100 | 54.0% | 55.8% | +1.8 | +1.4% | `results/main/selector_eval/cachescout/b100_eval/result.json` |
| selector_eval | 0.58 | 0.57 | 150 | 64.1% | 66.8% | +2.8 | -0.5% | `results/main/selector_eval/cachescout/b150_eval/result.json` |
| random_eval | 0.12 | 0.12 | 100 | 46.3% | 49.6% | +3.3 | +2.0% | `results/main/random_eval/cachescout/b100_eval/result.json` |
| random_eval | 0.12 | 0.12 | 150 | 60.5% | 65.5% | +5.0 | -2.7% | `results/main/random_eval/cachescout/b150_eval/result.json` |

## Load sweep at 150 blocks (GPU)

| arrival rate (sessions/s) | vanilla hit | CacheScout hit | vanilla TTFT (ms) | CacheScout TTFT (ms) | vanilla thr | CacheScout thr | source (CacheScout) |
|---|---|---|---|---|---|---|---|
| 0.5 | 64.1% | 66.8% | 75 | 74 | 7.64 | 7.63 | `results/main/selector_eval/cachescout/b150_eval/result.json` |
| 1.0 | 54.9% | 56.6% | 195 | 192 | 13.42 | 13.36 | `results/main/selector_eval_r1/cachescout/b150_eval/result.json` |
| 2.0 | 51.2% | 52.8% | 901 | 927 | 15.95 | 15.81 | `results/main/selector_eval_r2/cachescout/b150_eval/result.json` |
| 4.0 | 49.8% | 52.9% | 1592 | 1522 | 15.51 | 15.90 | `results/main/selector_eval_r4/cachescout/b150_eval/result.json` |

## Hook verification: sequential GPU vs simulator (tuning trace, 100 blocks)

| system | requests | exact matches | GPU hit | simulator hit | source |
|---|---|---|---|---|---|
| vanilla | 703 | 703 (100.0%) | 58.78% | 58.78% | `results/tune_gpu/vanilla/b100_seq/crosscheck.json` |
| lru_hook | 703 | 703 (100.0%) | 58.78% | 58.78% | `results/tune_gpu/lru_hook/b100_seq/crosscheck.json` |
| eviction_only | 703 | 688 (97.9%) | 62.48% | 62.14% | `results/tune_gpu/eviction_only/b100_seq/crosscheck.json` |
