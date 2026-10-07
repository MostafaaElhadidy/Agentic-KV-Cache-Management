# Real multi-agent workload: CacheScout vs vanilla vLLM

Branch `real-agents`. Real agents (Qwen2.5-1.5B-Instruct on vLLM 0.31.0, RTX 4060 Laptop 8 GB) solve real
GSM8K problems:
- prompts are built with the model's chat template;
- every agent's output feeds the later prompts;
- tools (calculator, scratchpad) really execute;
- answers are scored against GSM8K.

**Every number below comes from a result file.** Aggregates are in `results/report_real/summary.json` and
`results/report_real/tables.md` (regenerate with `python scripts/aggregate_real.py`). Per-run files are listed
in the `sources` fields. Plan and decisions: `docs/REAL_AGENTS_PLAN.md`, `docs/decisions.md` (2026-10-06
entries). Chronology: `docs/WORK_LOG.md`.

## 1. Summary

- **Controlled result (replay, the headline).**
  - Each live vanilla session was recorded once, then the identical prompts were replayed under vanilla and
    CacheScout.
  - CacheScout raised the KV-cache hit rate by **+5.7 / +6.3 / +6.5 pp** at 100 / 150 / 200 blocks (mean over 4
    topologies × 3 seeds).
  - The gain was positive in **35 of 36** seed × topology × budget pairs, ranging from −0.1 to +12.0 pp.
  - Latency effects are small and inconsistent: mean TTFT −2.9 / −0.9 / −1.2%, per-turn latency
    −1.9 / −0.0 / −1.0%, with TTFT lower in only 23 of 36 pairs.
- **Live closed-loop.** Hit-rate gains are similar (+5.3 / +6.0 / +8.6 pp), but live runs **diverge between
  systems**: the same problem at temperature 0 produces different conversations under different engine
  configurations. Latency differences in live runs (−18% to +105% TTFT on average, dominated by single outliers)
  therefore **cannot be attributed to CacheScout alone**.
- **Compared with the synthetic traces** (Selector, mean of 3 seeds: +2.7 / +2.2 / +1.2 pp), the real workload
  shows 2–5× larger hit-rate gains. They are still below the paper's +10–18 pp, except for single topology
  cells, which reach +9.6–9.9 pp.
- **The 1.5B model barely follows the selector protocol:** 77% of selector routing decisions are fallbacks. An
  AutoGen-style separate selector call was tried and rejected because it chose DECIDER 64.7% of the time.

## 2. Setup

| | |
|---|---|
| Model / engine | Qwen2.5-1.5B-Instruct (bf16), vLLM 0.31.0, `enforce_eager`, in-process AsyncLLM; CacheScout via `scheduler_cls` (same plugin as the synthetic study) |
| KV budget | `num_gpu_blocks_override` 100 / 150 / 200 (99 / 149 / 199 usable), `max_model_len` 1584 |
| Agents | 6 (PLANNER, ANALYST, CODER, TESTER, REVIEWER, DECIDER). Anchors 175–292 tokens; 6/6 distinct 32-token fingerprints (tested with the real tokenizer) |
| Tools | `calculator` (safe `ast` arithmetic), `scratchpad`. Results are fed back; the same agent continues (≤ 2 tool calls per invocation) |
| Routing | pipeline P→A→C→T→R→D; random (uniform over the other 5); debate P→C→R→D, D answers or `REVISE` → C; selector = agent's own `NEXT:` line, lenient name-only line, else round-robin fallback |
| Session | ends at DECIDER `FINAL ANSWER:`, end of chain, or 14 agent calls (forced DECIDER answer). Temperature 0, `max_tokens` 128, history trimmed to fit 1584 |
| Tasks | GSM8K test split, 3 disjoint slices of 20 problems (seeds 1/2/3, fixed permutation; ids in `summary.json → workload.*.problem_ids`). Tuning used train-split problems only |
| Load | Poisson session arrivals at 0.2 sessions/s (seeded per problem seed), 20 sessions per run |
| Constants | Synthetic constants, with λ re-tuned to 0.001 on train recordings (`results/tuning/real_eviction_sweep.json`) |
| Runs | 12 live recordings + 72 replay runs + 60 more live runs (+ 4 tuning recordings, 3 cross-checks, pilots). All campaign runs record `git_dirty=false` |

**Simulator validity on real sessions.** Sequential replay of a recorded real selector session matched the
simulator on **155/155 requests** for vanilla, the neutral hook and CacheScout eviction
(`results/real/replay_gsm8k_train_selector_s101/*/b100_seq/crosscheck.json`). Tuning in the simulator was
therefore valid.

## 3. Workload statistics (vanilla live recordings, 100 blocks, mean of 3 seeds)
Source: `results/report_real/summary.json → workload` (per-run files e.g.
`results/real/gsm8k_test_selector_s1/vanilla/b100_eval/result.json`).

| Topology | Measured R | LLM calls / run | GSM8K accuracy | FINAL ANSWER rate | Selector fallback | Strict `NEXT:` | Tool calls / run (failures) | Trim rate | Prompt tokens mean / max |
|---|---|---|---|---|---|---|---|---|---|
| pipeline | 1.00 | 163 | 0.53 | 0.97 | n/a | n/a | 55.3 (4.7) | 0.012 | 623 / 1444 |
| random | 0.14 | 181 | 0.38 | 0.88 | n/a | n/a | 38.3 (4.0) | 0.002 | 639 / 1451 |
| debate | 1.00 | 122 | 0.42 | 1.00 | n/a | n/a | 45.7 (3.3) | 0.000 | 536 / 995 |
| selector | 0.77 | 165 | 0.50 | 0.90 | **0.77** | 0.23 | 32.0 (4.0) | 0.032 | 592 / 1455 |

- **Predictability R vs the paper (Fig. 4a):**
  - Pipeline 1.00 matches the paper's 1.00.
  - Random 0.14 is close to the paper's 0.12.
  - Selector 0.77 is higher than the paper's 0.57. This does not mean the model routes predictably: 77% of its
    routing decisions are the round-robin fallback, which is deterministic (P→A→C→T→R→D), so the measured R
    mostly reflects the fallback rule rather than the model's own choices.
  - Debate is 1.00 vs the paper's 0.78, because this debate is rule-based and deterministic.
- **Next-agent prediction accuracy** (CacheScout engine-side learner, replay runs; `summary.json → replay.*.engine_pred_acc`):
  pipeline 0.94–0.95, debate 0.94, selector 0.81, random 0.17–0.18. Paper: 76–86% (Fig. 4b).
- **Selector fallback rate: 77%.** Most selector "decisions" are the round-robin fallback, so the selector
  topology is largely round-robin with occasional model choices. Two prompt revisions on train problems only
  raised strict `NEXT:` lines to 13–31% on the pilots.
  - A separate constrained selector call (vLLM guided choice) made 100% model decisions but was degenerate:
    DECIDER 64.7% (`results/real/gsm8k_train_selector_s102/vanilla/b100_pilot_router/result.json`).
  - By the agreed criterion (decisions > 90% and no agent > 60%) it was not adopted.
- **Trimming** is rare: 0–3.2% of calls.

## 4. Replay (controlled, headline)
Identical recorded prompts and output lengths for both systems, same arrival times and gaps. Mean ± std over 3
seeds of CacheScout − vanilla. Source: `results/report_real/summary.json → replay`. Per-run files:
`results/real/replay_gsm8k_test_<topology>_s<seed>/<system>/b<blocks>_replay/result.json`.
Figure: `results/report_real/fig_real_replay_gain.png`.

| Topology | Blocks | Vanilla hit | CacheScout hit | Δ hit (pp) | Δ TTFT | Δ per-turn latency | Δ session completion |
|---|---|---|---|---|---|---|---|
| pipeline | 100 | 20.7% | 26.5% | +5.7 ± 2.3 | −5.2 ± 11.0% | −3.7 ± 6.3% | −3.6 ± 6.0% |
| pipeline | 150 | 31.9% | 41.6% | +9.6 ± 2.6 | −7.9 ± 5.5% | −1.5 ± 1.0% | −1.4 ± 1.0% |
| pipeline | 200 | 36.5% | 46.4% | +9.9 ± 1.7 | −4.0 ± 1.2% | −0.5 ± 1.3% | −0.5 ± 1.2% |
| random | 100 | 19.3% | 21.5% | +2.2 ± 1.3 | −7.0 ± 5.1% | −4.1 ± 6.2% | −4.0 ± 6.0% |
| random | 150 | 31.4% | 33.0% | +1.6 ± 1.9 | +9.9 ± 21.3% | +2.7 ± 5.5% | +2.5 ± 5.0% |
| random | 200 | 38.0% | 41.9% | +3.9 ± 0.6 | −0.4 ± 1.7% | −1.7 ± 1.0% | −1.5 ± 0.9% |
| debate | 100 | 34.6% | 40.9% | +6.4 ± 4.4 | +2.5 ± 9.0% | +1.6 ± 4.1% | +1.5 ± 4.0% |
| debate | 150 | 46.3% | 50.4% | +4.1 ± 0.4 | −2.3 ± 5.7% | −0.2 ± 0.9% | −0.2 ± 0.8% |
| debate | 200 | 50.9% | 53.3% | +2.4 ± 0.4 | +2.3 ± 3.8% | −0.5 ± 2.0% | −0.4 ± 1.8% |
| selector | 100 | 18.5% | 27.1% | +8.6 ± 3.0 | −2.0 ± 7.8% | −1.2 ± 3.9% | −1.1 ± 3.7% |
| selector | 150 | 29.3% | 39.0% | +9.7 ± 1.1 | −3.4 ± 1.2% | −1.0 ± 1.1% | −0.9 ± 1.0% |
| selector | 200 | 36.1% | 45.9% | +9.8 ± 1.5 | −2.8 ± 4.4% | −1.5 ± 1.1% | −1.3 ± 1.0% |
| **all** | 100 | | | **+5.7** | −2.9% | −1.9% | −1.8% |
| **all** | 150 | | | **+6.3** | −0.9% | −0.0% | −0.0% |
| **all** | 200 | | | **+6.5** | −1.2% | −1.0% | −0.9% |

- **GSM8K accuracy** is identical for both systems in replay by construction (recorded outputs are replayed).
  See the recordings' accuracy in §3.
- Throughput: mean differences within ±3.3%, per-run −0.9% to +8.0% (`tables.md`). The load (0.2 sessions/s) is far below saturation.

## 5. Live closed-loop (reported separately; NOT a controlled comparison)
Each system generates its own outputs. Source: `results/report_real/summary.json → live`. Per-run files:
`results/real/gsm8k_test_<topology>_s<seed>/<system>/b<blocks>_eval/result.json`.
Figure: `results/report_real/fig_real_live_gain.png`.

| Blocks | Δ hit (pp) | Δ TTFT | Δ per-turn latency | Δ session completion |
|---|---|---|---|---|
| 100 | +5.3 | +11.2% | +4.8% | +6.8% |
| 150 | +6.0 | +104.6% | +19.0% | +17.8% |
| 200 | +8.6 | −18.5% | −6.3% | −8.0% |

**Live runs diverge between systems.** On the same problems at temperature 0, the two systems produce
different conversations: different lengths, routing, answers and trimming. This is because engine scheduling
and batching change floating-point results, and greedy decoding amplifies tiny differences.

Example (selector, 150 blocks, seed 3):
- vanilla: 155 calls, largest prompt 1,061 tokens, mean TTFT 135 ms;
- CacheScout: 154 calls, largest prompt 1,456 tokens, 2 capped sessions, mean TTFT 1,419 ms;
- source: `.../gsm8k_test_selector_s3/{vanilla,cachescout}/b150_eval/result.json`.

That single seed drives the +105% mean at 150 blocks. **Live latency differences are not attributable to
CacheScout alone.** Live GSM8K accuracy is 0.461 (vanilla) vs 0.475 (CacheScout) averaged over the 12
topology × budget cells; per-cell differences range from −0.07 to +0.10. That's consistent with divergence
noise, not an effect of the cache policy, which does not change model outputs.

## 6. Comparison with the synthetic traces and the paper

| | Paper (Llama-3.1-8B) | Synthetic traces (this repo) | Real agents, replay (this repo) |
|---|---|---|---|
| Hit-rate gain over vanilla | +10–18 pp | +1.2–2.7 pp (Selector, 3 seeds) | +5.7–6.5 pp mean; up to +9.6–9.9 pp (pipeline/selector) |
| Gain vs cache size | shrinks as memory grows (Fig. 14a) | shrinks (2.7 → 2.2 → 1.2 pp) | does **not** shrink (5.7 → 6.3 → 6.5 pp) |
| Gain under random routing | diminishes | +3.3 / +5.0 pp (no) | +1.6 to +3.9 pp, the smallest of the 4 topologies (**partly yes**) |
| TTFT reduction | 18–45% | none consistent | −0.9 to −2.9% mean, not consistent |
| Next-agent prediction accuracy | 76–86% | 76% (Selector) | 81% selector, 94–95% pipeline/debate, 17–18% random |

Why the real workload shows larger gains than the synthetic one:
- Vanilla hit rates are much lower (19–51% vs 56–70%), because real outputs (49 tokens on average, run means 41–62, cap 128) and
  tool results make histories longer and more varied, so there's more to win.
- Every agent's prompt starts with its anchor, and multi-call tool invocations re-read long shared prefixes.

Random routing now gives the smallest gain, which is directionally what the paper predicts, although it isn't
zero.

## 7. Limitations and threats to validity
- **Small model.**
  - It follows protocols poorly: 77% selector fallbacks, and 3–5 tool-call failures per run out of 32–55 tool
    calls.
  - GSM8K accuracy is 0.38–0.53, low for a team of agents.
  - The selector topology is therefore largely round-robin.
- **Latency.** Prefill is cheap at 1.5B; hit-rate gains barely move TTFT even in controlled replay.
- **Live divergence** makes live comparisons noisy. Replay fixes content but replays recorded timing gaps rather
  than reacting to them.
- **Scale.**
  - 20 sessions per run, 3 seeds, one GPU, 0.2 sessions/s.
  - Higher load and larger models were not tested (see `docs/CLOUD_RUNBOOK.md`).
- **Rule-based protocols.** Debate is deterministic here, so R = 1.00 rather than the paper's 0.78.
- **Prompt and constant choices.**
  - Prompts were iterated on train problems; one test problem (`test:844`, in seed 1) appeared in the very first
    smoke run, before prompt iteration moved to train problems.
  - λ was re-tuned on train recordings with a borderline gain (0.20 pp, exactly the threshold).
- **Same two interpretations as the synthetic study.** Session-scope learning and anchor-only block mapping
  (`docs/decisions.md`).
