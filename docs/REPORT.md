# CacheScout replication: local phase report (RTX 4060 8 GB)

Paper: "Learning Agent Execution for KV-Cache Management in Agentic Serving" (CacheScout, arXiv:2608.14624).
Date: 2026-10-06. Code commit: see `git log` (each result.json also stores its commit).
**Every number below comes from a file in `results/`, cited by path.** The aggregated values are in
`results/report/summary.json` and `results/report/tables.md` (regenerate: `python scripts/aggregate_report.py`).

---

## 1. Summary

**What was built.**
- Engine-independent CacheScout core: Transition Learner (Eqs. 2-5), Survival Scorer (Eqs. 7-9), Alg. 1
  runtime and the Background Prefetch Coordinator (Eqs. 10-11). Code: `src/cachescout/core/`.
- A hook into **real vLLM 0.31.0** through its official `scheduler_cls` extension point
  (`src/cachescout/vllm_plugin/scheduler.py`). No file in site-packages was edited.
- An experiment runner with an on/off switch (`python -m cachescout.run --system ...`, or `cachescout.enabled`
  in the YAML config), plus ablations (eviction-only, warmup-only) and two baselines: vanilla vLLM and a
  Continuum-style TTL baseline.
- A pure-Python simulator of vLLM's prefix cache, a synthetic six-agent trace generator based on paper Fig. 5,
  `scripts/compare.py` (side-by-side tables, paper trend checks, plots), and 86 unit tests.

**Fidelity level achieved: full vLLM hook.** The paper's three hooks (ObserveTouch at prefix matching,
survival-guided eviction, warmup through the serving API) run inside vLLM's engine process. None of the
fallback levels was needed. The hook was verified on the GPU:
- With the hook installed but set to plain LRU, results were identical to vanilla vLLM on 703 of 703 requests.
- The simulator matched vLLM exactly on 703 of 703 requests for vanilla, and on 97.9% of requests for CacheScout
  eviction.
- Source: `results/tune_gpu/*/b100_seq/crosscheck.json`.

**Main result (honest).**
- On real vLLM with Qwen2.5-1.5B and the paper's cache budgets (100/150/200 blocks), CacheScout raises the
  KV-cache hit rate by **+2.7 / +2.2 / +1.2 percentage points** over vanilla (mean of 3 seeds; paper: +10 to
  +18 pp).
- The gain is positive for every seed and budget, and on average it shrinks as the cache grows (same direction
  as paper Fig. 14a).
- The largest gain is on the deterministic Pipeline topology: **+13.1 pp** at 100 blocks, with mean TTFT −8.7%.
- On the main Selector workload, TTFT, per-turn latency and throughput show **no consistent improvement**.
  The differences are ±0-6% and change sign across seeds. The paper reports −18 to −45% TTFT.
- Two of the paper's claims did **not** reproduce here:
  - Gains do not vanish under Random routing (+3.3 / +5.0 pp).
  - Prediction itself adds nearly nothing: a variant with prediction switched off is as good as full
    CacheScout.
- Getting the hit-rate gains at all required two interpretations where the paper is ambiguous (§5). Algorithm 1
  implemented literally gave only +0.4 / +0.1 / −1.0 pp.

---

## 2. Environment, scaled-down setup, and differences from the paper

| | Paper (Sec. 5.1) | This replication (local) |
|---|---|---|
| GPU | 8× RTX PRO 6000 96 GB; 4× H200 for the 235B model | 1× RTX 4060 Laptop 8 GB (WSL2, Windows 11) |
| vLLM | v0.11 (V1), 800-line patch to 5 files | **0.31.0**, `scheduler_cls` plugin, no source edits |
| Model | Llama-3.1-8B-Instruct; Qwen3-235B-A22B-FP8 | Qwen2.5-1.5B-Instruct (bf16), `enforce_eager=True` |
| KV budget | 100-200 blocks (Fig. 14); main budget not stated | `num_gpu_blocks_override` 100/150/200 (99/149/199 usable: vLLM reserves 1 null block) |
| Workloads | GSM8K, MT-Bench, GAIA, SWE-bench on a 6-agent AutoGen SelectorGroupChat | **Synthetic** six-agent traces from the Fig. 5 transition tables (Selector, Pipeline, Debate, Random); no real LLM routing |
| Sessions / load | 0.2-50 sessions/s | 60 sessions per trace, 0.5-4 sessions/s, `max_num_seqs` 8 |
| Baselines | vanilla vLLM, Continuum (authors' fork, TTL 0.3 s) | vanilla vLLM (stock scheduler), Continuum-style soft TTL pinning (our re-implementation) |
| CPU KV tier | used | not used |
| Software | not stated | Python 3.12.15, torch 2.13.0+cu132, transformers 5.17.0 |

Workload calibration was done **with vanilla only, on tuning traces**, so it cannot favour CacheScout
(`scripts/calibrate_workload.py`, `results/calibration/workload_calibration.json`). The synthetic traces match the
paper's motivation statistics (`results/traces/stats.json`):

| Statistic | Paper | Ours |
|---|---|---|
| Anchor share of prompt tokens (Fig. 2) | 53-62% | 54-62% |
| Entropy reduction R, Pipeline / Debate / Selector / Random (Fig. 4a) | 1.00 / 0.78 / 0.57 / 0.12 | 1.00 / 0.77 / 0.58 / 0.12 |
| Anchor ratio φ at turn 12 (Fig. 3b) | 43-60% | 0.42-0.45 (eval traces; turn 11 for Pipeline: 0.36) |
| Max request footprint (Sec. 5.4) | ~93 blocks | 27-44 blocks |
| Reuse per anchor / history block (Fig. 3a) | 49-173 / 12-15 | 84-190 / 0.3-1.7 |

The last two rows are **deviations**. Our requests are smaller, and our session-history blocks are reused much
less, so anchors carry relatively more of the value. Vanilla's hit rate, however, falls in the paper's vanilla
range: 56-70% here vs 64-77% in Fig. 14a.

---

## 3. How to use it

```bash
cd ~/CacheScoutImplementation
source ~/testLLM/.venv/bin/activate      # the existing vLLM environment
pytest -q                                # 86 tests, no GPU
python scripts/make_traces.py            # (re)create the synthetic traces in results/traces/
```

**The on/off switch.** All runs share the same config file (`configs/experiments/main/local.yaml`): the same
trace, model, seed and cache budget. Only the *system* changes:

```bash
CFG=configs/experiments/main/local.yaml
python -m cachescout.run --config $CFG --system vanilla    --blocks 100   # CacheScout OFF (stock vLLM)
python -m cachescout.run --config $CFG --system cachescout --blocks 100   # CacheScout ON (eviction + warmup)
python -m cachescout.run --config $CFG --system eviction_only  --blocks 100   # ablation
python -m cachescout.run --config $CFG --system warmup_only    --blocks 100   # ablation
python -m cachescout.run --config $CFG --system continuum      --blocks 100   # TTL baseline
python -m cachescout.run --config $CFG --variant cachescout_literal --blocks 100  # Alg. 1 exactly as written
python -m cachescout.run --config $CFG --variant no_prediction      --blocks 100  # tau = 0
```

- Without `--system`, the YAML key `cachescout.enabled: true|false` decides (plus `cachescout.eviction` /
  `cachescout.warmup` for the ablations). `--system` always wins.
- When CacheScout is ON, the runner tells vLLM to use `CacheScoutScheduler` and passes the constants through one
  environment variable (`CACHESCOUT_CONFIG`), as in the paper (Sec. 4).
- Each run writes `results/main/<trace>/<system>/b<blocks>_<tag>/result.json`, which contains the config,
  versions, git commit, every request and a summary.
- For safety on the 8 GB laptop, wrap GPU runs:
  `scripts/gpu_run.sh <name> 2400 python -m cachescout.run ...`. It refuses to start if another GPU job is
  running and logs memory to `logs/`.

**Comparison, cache-size sweep, and the whole campaign:**

```bash
python scripts/compare.py --config $CFG --blocks 100,150,200 --run   # runs only missing results, one at a time
python scripts/compare.py --config $CFG --blocks 100,150,200         # load + print table + plot
python scripts/compare.py --config $CFG --blocks 100,150,200 --sim   # same table from the simulator (no GPU)
bash scripts/run_all_local.sh                                        # full campaign (~2 h if nothing exists)
python scripts/aggregate_report.py                                   # report tables + figures
```

`compare.py` prints hit rate, mean/median/P99 TTFT, per-turn latency and throughput for each system and budget.
It also prints PASS/FAIL trend checks against the paper and saves `.json/.md/.png` under `results/main/compare/`.

---

## 4. Results (trends vs the paper)

All GPU numbers are from vLLM 0.31.0 on the RTX 4060 (online mode, Poisson session arrivals). The paper's
numbers are for Llama-3.1-8B on a much larger testbed and much larger workloads, so **only directions and
orderings are comparable**.

### 4.1 Cache-size sweep: CacheScout vs vanilla (paper Fig. 14a, Fig. 8a)
Selector topology, mean ± std over 3 evaluation seeds (traces `selector_eval`, `_s2`, `_s3`).
Source: `results/report/summary.json → seeds` (per-run files listed there, e.g.
`results/main/selector_eval_s2/cachescout/b100_eval/result.json`).

| Budget (blocks) | Vanilla hit | CacheScout hit | Δ hit (pp) | Δ TTFT mean | Δ latency mean | Paper |
|---|---|---|---|---|---|---|
| 100 | 56.1% | 58.7% | **+2.7 ± 0.9** | −2.1 ± 3.8% | −1.2 ± 2.2% | vanilla 64.4%, CacheScout 87% |
| 150 | 65.6% | 67.8% | **+2.2 ± 0.5** | −0.5 ± 1.7% | +0.2 ± 1.8% | (between) |
| 200 | 70.2% | 71.4% | **+1.2 ± 0.4** | −0.3 ± 0.1% | −0.0 ± 0.3% | vanilla 76.6%, CacheScout 86-87% |

Figure: `results/report/fig_cache_sweep.png`. The eviction-only line lies exactly under the CacheScout line
because their 3-seed means coincide.

| Trend (paper) | Ours | Same direction? |
|---|---|---|
| CacheScout hit rate > vanilla at every budget (Fig. 8a, 14a) | yes, all 9 seed×budget pairs positive (+0.8 to +3.5 pp) | **yes** |
| Gain largest at small budgets, shrinking as memory grows (Fig. 14a) | 3-seed mean 2.7 → 2.2 → 1.2 pp (seed 1 alone is non-monotone: 1.8 / 2.8 / 1.6) | **yes, on average** |
| Size of gain: +10-18 pp (Fig. 8a), ~+23 pp at 100 blocks (Fig. 14a) | +1.2 to +2.7 pp | **no** (≈5-10× smaller) |
| Mean TTFT −18-45% (Fig. 8b) | −2.1% to −0.3% (seed std up to 3.8%) | **no** |
| Mean per-turn latency −29-38% (Fig. 10a) | −1.2% to +0.2% (seed std up to 2.2%) | **no** |
| Higher throughput (Fig. 10b, Fig. 11) | no difference (see 4.4) | **no** |

### 4.2 Mechanism ablation and baselines (paper Fig. 12, Fig. 8)
`selector_eval` (seed 1). Source: `results/report/tables.md` "Main sweep" (one result.json per row, e.g.
`results/main/selector_eval/eviction_only/b100_eval/result.json`).

| System | Δ hit @100 | Δ hit @150 | Δ hit @200 | Paper |
|---|---|---|---|---|
| eviction only | +1.9 | +2.5 | +1.6 | +18-22 pp (Fig. 12a) |
| warmup only | +0.8 | +1.1 | +0.5 | "at most one percentage point" (Sec. 5.3) |
| CacheScout (eviction + warmup) | +1.8 | +2.8 | +1.6 | +10-18 pp (Fig. 8a) |
| Continuum-style TTL (0.3 s) | +0.0 | +0.1 | −0.1 | below CacheScout (Fig. 8a) |
| CacheScout-literal (Alg. 1 as written) | +0.4 | +0.1 | −1.0 | n/a |
| no prediction (τ = 0) | +2.1 | +2.8 | +1.8 | n/a |

Trends:
- Eviction is the main source of gain and warmup adds little: **same as the paper**. Warmup-only is marginally
  above 1 pp at 150 blocks (+1.1).
- Full CacheScout is not better than eviction-only on hit rate. The paper reports extra *latency* benefit from
  warmup (−28% on GAIA); we see none.
- CacheScout beats our Continuum-style baseline at every budget, but that baseline is degenerate (§5).
- In the 3-seed means, eviction-only and full CacheScout are equal (+2.7 / +2.3 / +1.2 vs +2.7 / +2.2 / +1.2).

### 4.3 Coordination topologies (paper Fig. 4a, Fig. 5, Tab. 1)
Seed 1. Source: `results/report/summary.json → topologies`, figure `results/report/fig_topologies.png`.

| Topology | R (ours / paper) | Δ hit @100 | Δ hit @150 | Δ TTFT @100 |
|---|---|---|---|---|
| Pipeline | 1.00 / 1.00 | **+13.1** | +5.7 | −8.7% |
| Debate | 0.77 / 0.78 | +1.3 | +0.8 | −0.3% |
| Selector | 0.58 / 0.57 | +1.8 | +2.8 | +1.4% |
| Random | 0.12 / 0.12 | +3.3 | +5.0 | +2.0% |

- The paper expects gains to diminish as routing approaches random (Tab. 1). **Not reproduced:** Random still
  gains +3.3 / +5.0 pp.
- Explanation: together with the no-prediction ablation (§4.2), this shows the gain on our workload comes from
  keeping shared anchor blocks in preference to session history, which works regardless of routing. It does not
  come from predicting the next agent.
- Pipeline is the only case with a large gain and a TTFT reduction. There, the next agent's anchor is
  deterministic, and every anchor has a long, regular reuse distance that LRU handles badly.

### 4.4 Load sweep (paper Fig. 11), 150 blocks, seed 1
Source: `results/report/summary.json → load`.

| Arrival (sessions/s) | Vanilla hit / CacheScout hit | Vanilla TTFT / CacheScout TTFT (ms) | Throughput (turns/s) |
|---|---|---|---|
| 0.5 | 64.1% / 66.8% | 75 / 74 | 7.64 / 7.63 |
| 1.0 | 54.9% / 56.6% | 195 / 192 | 13.42 / 13.36 |
| 2.0 | 51.2% / 52.8% | 901 / 927 | 15.95 / 15.81 |
| 4.0 | 49.8% / 52.9% | 1592 / 1522 | 15.51 / 15.90 |

- The hit-rate gain holds under load (+1.5 to +3.1 pp).
- Both systems saturate at about 15.5-16 turns/s. Unlike the paper, CacheScout does **not** delay saturation:
  the GPU is compute-bound on decode, not prefill-bound, at this model size.

### 4.5 Learner quality and overhead (paper Fig. 4b, Fig. 15)
- **Prediction accuracy.** Engine-side next-agent accuracy during the full CacheScout runs was **75.7-76.2%**,
  and the learned R was 0.58 (true 0.58). Paper: 76-86% within 50 dispatches.
  Source: `results/report/summary.json → engine`.
- With Alg. 1's literal global current agent, the interleaved stream gives only 27-65% accuracy
  (`results/traces/stats.json → online_accuracy_global`). This is why session scope was used.
- **Overhead (pure Python, `results/microbench/overhead.json`):**
  - Per dispatch (observe, including the survival refresh): 37 / 70 / 172 µs for 6 / 12 / 24 agents.
  - Victim selection over 255 candidate blocks: ~225-235 µs.
  - Learner state: 2.7 / 12.2 / 40.5 KB after 10,000 dispatches.
  - Paper: ~1 µs ObserveTouch, ≤ 6 µs PredictSurvival, < 25 KB at 24 agents. **Ours is 1-2 orders of magnitude
    slower** (Python dicts, full re-scoring at every eviction). During the GPU runs, the engine-side mean was
    76-79 µs per observe and 37-94 µs per selection.

### 4.6 Simulator (no GPU) on the same evaluation trace
Source: `results/main/compare/selector_eval_sim_eval_20261006-165243.json`.
- CacheScout gains +3.3 / +2.6 / +1.0 pp, the literal Alg. 1 variant +0.2 / −1.0 / −0.9, and no-prediction
  +3.0 / +3.1 / +1.6. These are the same orderings as on the GPU.
- Belady's optimal (offline) eviction on the tuning trace would gain +12 / +9 / +6 pp (WORK_LOG, Belady
  diagnostic). So there is headroom that CacheScout captures only partly (roughly a third at 100 blocks).

---

## 5. What worked, what didn't, deviations

**Worked**
- Non-invasive vLLM 0.31 hook via `scheduler_cls`, verified request by request against the simulator.
- The on/off switch, ablations, comparison script and plots.
- Synthetic traces that reproduce the paper's R values and anchor share.
- Online transition learning reaching the paper's accuracy range (session scope).
- Small but consistent hit-rate gains, with the paper's direction for the cache-size and ablation trends.

**Didn't work / didn't reproduce**
- The *magnitude* of hit-rate gains (≈5-10× smaller).
- Any TTFT, latency or throughput improvement on the main workload (only Pipeline showed −8.7% TTFT).
- "Gains vanish under random routing".
- A measurable benefit of next-agent *prediction* over simply protecting shared anchors.
- Microsecond-scale overhead.

**Deviations and interpretations** (full list: `docs/decisions.md`, `docs/open_questions.md`)
1. **Session scope (B4, interpretation).**
   - Alg. 1 keeps one global "current agent". With ~8 interleaved sessions it learned R = 0.18 and 43% accuracy,
     against a true 0.57, so survival scores were flat and the policy behaved like LRU.
   - Fix: transitions are counted per session (vLLM's native `Request.session_id`), and survival is averaged
     over the 8 most recent sessions.
2. **Anchor-only block mapping (B3, interpretation).**
   - Alg. 1 line 12 maps every touched block, including session history, to the agent. Sec. 3.3 says blocks
     "inherit the survival score of its corresponding agent anchor".
   - The literal mapping let stale history of busy agents displace anchors (Belady diagnostic in WORK_LOG).
   - Fix: only blocks shared by ≥ 2 sessions inherit survival; other blocks rank by recency.
   - Literal variant result: +0.4 / +0.1 / −1.0 pp.
3. **Constants were never given by the paper.** They were tuned in the simulator on separate tuning traces
   (seeds 101/102), never on evaluation traces: ε 0.01, τ 0.1, E_max 6, λ 0.005/step, δ 0.01, R_min 0.3,
   fingerprint of 2 blocks. Files: `results/tuning/*.json`.
4. **Synthetic traces, no AutoGen or real datasets.** The Selector column placement in Fig. 5 is a
   reconstruction chosen to match R = 0.57.
5. **Continuum baseline is degenerate.**
   - Soft 0.3 s pinning protects exactly the blocks LRU already treats as most recent, so it equals vanilla.
   - It is not a faithful reproduction of the authors' Continuum fork.
6. **Measurement setup.**
   - `enforce_eager=True` and `max_num_batched_tokens=2048` are used for all systems (8 GB GPU, crash
     mitigation).
   - No CPU KV tier, so the CPU-resident preference of Sec. 4 is not implemented.
   - In-process AsyncLLM instead of an HTTP server.

**Paper inconsistencies that affected the work** (`docs/paper_notes.md` §9)
- Eviction-only gains (+18-22 pp) exceed the full system's (+10-18 pp), so it's unclear which configuration
  produced which number.
- Hit rates of 81-85% vs 86-87% in different places.
- Alg. 1 uses a global current agent and maps all blocks, while the text speaks of anchors. This directly
  caused deviations 1 and 2.
- τ is undeclared in Alg. 1, and the units of `age` are inconsistent.

---

## 6. Limitations and threats to validity
- **Small model, short prompts.**
  - Qwen2.5-1.5B with ~340-token prompts: prefill is cheap, so hit-rate gains barely move TTFT.
  - This alone may explain the missing latency gains; the cloud phase with Llama-3.1-8B is needed to test it.
- **Synthetic workloads.**
  - Token content is random, and only the sharing structure is modelled.
  - Session-history reuse is far below the paper's (Fig. 3a), and requests are smaller (≤ 44 vs ~93 blocks).
  - Both shift value toward anchors and may inflate or deflate the gains relative to real AutoGen traces.
- **Guessed constants and two interpretations.** Without them, CacheScout ≈ vanilla on our workload. A
  different reading of the paper could give different results.
- **Statistics.**
  - 3 seeds for the main sweep; single seed for topologies, load and extra ablations.
  - Online runs are timing-dependent. Identical configurations were not repeated, so run-to-run noise is not
    quantified; the seed-to-seed spread (±0.4-0.9 pp hit, ±0.1-3.8% TTFT) is the only variability estimate.
- **WSL2 and laptop GPU.**
  - One unexplained VM crash occurred during start-up (WORK_LOG crash #1). Afterwards nothing ran concurrently
    with GPU jobs.
  - Eager mode slows decoding, so absolute latencies are pessimistic.
- **Version difference.** The hook targets vLLM 0.31.0, not the paper's v0.11, so internals differ (e.g.
  AsyncScheduler is the default here).
- **Python implementation.** Overhead is not representative of an optimized runtime.

---

## 7. Next steps (cloud phase)
See `docs/CLOUD_RUNBOOK.md` for exact commands on an A100/H100. Plan:
1. Repeat the sim-vs-GPU cross-check with Llama-3.1-8B.
2. Repeat the 100-200 block sweep, ablations, topologies and the load sweep up to 50 sessions/s. This tests
   whether a model with costlier prefill turns the hit-rate gains into TTFT gains.
3. Run at a natural, non-overridden cache size.
4. If possible, record real AutoGen SelectorGroupChat traces with an 8B model. That would check history reuse
   (Fig. 3a) and request sizes, the two workload deviations above.
5. Optimize the runtime: precomputed per-agent scores, a heap instead of a full scan. This targets the paper's
   microsecond overhead.
