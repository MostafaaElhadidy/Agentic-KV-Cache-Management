## Workload statistics (vanilla live recordings, 100 blocks, mean over 3 seeds)

| topology | measured R | LLM calls/run | GSM8K accuracy | FINAL ANSWER rate | selector fallback | strict NEXT | tool calls/run (failures) | trim rate | prompt tokens mean / max | sources |
|---|---|---|---|---|---|---|---|---|---|---|
| pipeline | 1.00 | 163 | 0.53 | 0.97 | n/a | n/a | 55.3 (4.7) | 0.012 | 623 / 1444 | `results/real/gsm8k_test_pipeline_s1/vanilla/b100_eval/result.json` (+2) |
| random | 0.14 | 181 | 0.38 | 0.88 | n/a | n/a | 38.3 (4.0) | 0.002 | 639 / 1451 | `results/real/gsm8k_test_random_s1/vanilla/b100_eval/result.json` (+2) |
| debate | 1.00 | 122 | 0.42 | 1.00 | n/a | n/a | 45.7 (3.3) | 0.000 | 536 / 995 | `results/real/gsm8k_test_debate_s1/vanilla/b100_eval/result.json` (+2) |
| selector | 0.77 | 165 | 0.50 | 0.90 | 0.77 | 0.23 | 32.0 (4.0) | 0.032 | 592 / 1455 | `results/real/gsm8k_test_selector_s1/vanilla/b100_eval/result.json` (+2) |

## REPLAY (controlled; identical recorded prompts) — headline

Mean ± std over 3 seeds of CacheScout − vanilla.

| topology | blocks | vanilla hit | CacheScout hit | Δ hit (pp) | Δ TTFT | Δ per-turn latency | Δ session completion | Δ throughput |
|---|---|---|---|---|---|---|---|---|
| pipeline | 100 | 20.7% | 26.5% | +5.7 ± 2.3 | -5.2 ± 11.0% | -3.7 ± 6.3% | -3.6 ± 6.0% | +1.3 ± 2.0% |
| pipeline | 150 | 31.9% | 41.6% | +9.6 ± 2.6 | -7.9 ± 5.5% | -1.5 ± 1.0% | -1.4 ± 1.0% | +0.1 ± 0.4% |
| pipeline | 200 | 36.5% | 46.4% | +9.9 ± 1.7 | -4.0 ± 1.2% | -0.5 ± 1.3% | -0.5 ± 1.2% | +0.2 ± 0.3% |
| random | 100 | 19.3% | 21.5% | +2.2 ± 1.3 | -7.0 ± 5.1% | -4.1 ± 6.2% | -4.0 ± 6.0% | +3.3 ± 4.2% |
| random | 150 | 31.4% | 33.0% | +1.6 ± 1.9 | +9.9 ± 21.3% | +2.7 ± 5.5% | +2.5 ± 5.0% | -0.4 ± 0.8% |
| random | 200 | 38.0% | 41.9% | +3.9 ± 0.6 | -0.4 ± 1.7% | -1.7 ± 1.0% | -1.5 ± 0.9% | +0.4 ± 0.1% |
| debate | 100 | 34.6% | 40.9% | +6.4 ± 4.4 | +2.5 ± 9.0% | +1.6 ± 4.1% | +1.5 ± 4.0% | -0.0 ± 0.2% |
| debate | 150 | 46.3% | 50.4% | +4.1 ± 0.4 | -2.3 ± 5.7% | -0.2 ± 0.9% | -0.2 ± 0.8% | -0.0 ± 0.0% |
| debate | 200 | 50.9% | 53.3% | +2.4 ± 0.4 | +2.3 ± 3.8% | -0.5 ± 2.0% | -0.4 ± 1.8% | +0.1 ± 0.1% |
| selector | 100 | 18.5% | 27.1% | +8.6 ± 3.0 | -2.0 ± 7.8% | -1.2 ± 3.9% | -1.1 ± 3.7% | +0.2 ± 0.6% |
| selector | 150 | 29.3% | 39.0% | +9.7 ± 1.1 | -3.4 ± 1.2% | -1.0 ± 1.1% | -0.9 ± 1.0% | +0.1 ± 0.3% |
| selector | 200 | 36.1% | 45.9% | +9.8 ± 1.5 | -2.8 ± 4.4% | -1.5 ± 1.1% | -1.3 ± 1.0% | +0.3 ± 0.2% |

replay — mean over all topologies and seeds:

| blocks | Δ hit (pp) | Δ TTFT | Δ latency | Δ completion |
|---|---|---|---|---|
| 100 | +5.7 | -2.9% | -1.9% | -1.8% |
| 150 | +6.3 | -0.9% | -0.0% | -0.0% |
| 200 | +6.5 | -1.2% | -1.0% | -0.9% |

## LIVE closed-loop (outputs diverge between systems; not attributable to CacheScout alone)

Mean ± std over 3 seeds of CacheScout − vanilla.

| topology | blocks | vanilla hit | CacheScout hit | Δ hit (pp) | Δ TTFT | Δ per-turn latency | Δ session completion | Δ throughput | accuracy vanilla / CacheScout |
|---|---|---|---|---|---|---|---|---|---|
| pipeline | 100 | 21.2% | 25.0% | +3.8 ± 5.3 | +36.6 ± 101.7% | +20.5 ± 56.3% | +19.0 ± 61.0% | +6.1 ± 14.2% | 0.53 / 0.58 |
| pipeline | 150 | 30.9% | 38.2% | +7.3 ± 4.7 | +79.3 ± 78.1% | +21.2 ± 18.4% | +20.3 ± 17.4% | -0.1 ± 5.6% | 0.55 / 0.55 |
| pipeline | 200 | 34.6% | 43.9% | +9.2 ± 5.2 | -21.0 ± 44.6% | +3.6 ± 25.3% | +3.1 ± 25.6% | +2.6 ± 5.9% | 0.47 / 0.57 |
| random | 100 | 19.0% | 19.1% | +0.1 ± 2.4 | +25.3 ± 34.2% | +12.5 ± 16.2% | +15.1 ± 21.6% | +0.8 ± 1.3% | 0.38 / 0.47 |
| random | 150 | 28.5% | 29.8% | +1.3 ± 2.5 | +30.4 ± 66.3% | +2.4 ± 17.5% | +9.0 ± 21.4% | +7.1 ± 7.7% | 0.47 / 0.42 |
| random | 200 | 30.8% | 41.0% | +10.2 ± 4.8 | -26.2 ± 45.3% | -14.7 ± 13.9% | -15.0 ± 16.0% | +2.3 ± 5.2% | 0.43 / 0.37 |
| debate | 100 | 34.8% | 39.9% | +5.1 ± 3.5 | +11.2 ± 18.4% | +5.3 ± 8.6% | +9.5 ± 10.3% | -2.8 ± 8.2% | 0.42 / 0.43 |
| debate | 150 | 45.1% | 51.0% | +5.8 ± 2.6 | +15.9 ± 21.8% | +9.8 ± 7.1% | +6.9 ± 7.2% | -3.8 ± 3.3% | 0.45 / 0.45 |
| debate | 200 | 49.3% | 52.4% | +3.2 ± 2.8 | +0.9 ± 3.0% | -6.1 ± 8.5% | -9.3 ± 4.4% | +0.8 ± 2.5% | 0.42 / 0.43 |
| selector | 100 | 19.3% | 31.7% | +12.3 ± 6.9 | -28.5 ± 50.8% | -18.9 ± 33.9% | -16.3 ± 34.4% | +10.3 ± 12.9% | 0.50 / 0.48 |
| selector | 150 | 30.3% | 39.6% | +9.4 ± 8.3 | +292.9 ± 570.2% | +42.6 ± 89.9% | +34.8 ± 81.7% | -2.3 ± 9.3% | 0.45 / 0.47 |
| selector | 200 | 34.3% | 45.9% | +11.6 ± 0.7 | -27.8 ± 26.6% | -7.9 ± 9.9% | -10.8 ± 9.1% | -1.9 ± 2.7% | 0.47 / 0.48 |

live — mean over all topologies and seeds:

| blocks | Δ hit (pp) | Δ TTFT | Δ latency | Δ completion |
|---|---|---|---|---|
| 100 | +5.3 | +11.2% | +4.8% | +6.8% |
| 150 | +6.0 | +104.6% | +19.0% | +17.8% |
| 200 | +8.6 | -18.5% | -6.3% | -8.0% |
