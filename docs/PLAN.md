# Replication plan

Two phases. **Phase 1 (local):** RTX 4060 Laptop 8 GB, WSL2, Qwen2.5-1.5B-Instruct, scaled-down workloads.
**Phase 2 (cloud):** A100/H100, Llama-3.1-8B-Instruct (paper's main model), paper-scale workloads.
Same code; only `configs/hardware/*.yaml` and the experiment `cloud.yaml` change.

## What "successful replication" means in Phase 1

Absolute numbers will differ (smaller model, smaller workloads, different GPU). Success = **same trends**:
1. CacheScout hit rate > vLLM-LRU hit rate at tight block budgets (paper Fig. 14a: 86–87% vs 64–77% over 100–200
   blocks), and the gap **shrinks** as the budget grows.
2. Eviction-only delivers most of the gain; prefetch alone is ≈ inert (Sec. 5.3, Fig. 12a).
3. Under **Random** routing the advantage mostly disappears (graceful degradation toward LRU, Sec. 3.3).
4. Higher hit rate → lower mean TTFT and per-turn latency (direction must match Fig. 8b and Fig. 10a).
5. Online top-1 next-agent accuracy reaches roughly 70–90% within ~50 dispatches on structured topologies (Fig. 4b),
   and measured R ordering is Pipeline > Debate > Selector > Random (Fig. 4a).
6. Runtime overhead stays in the microsecond range per operation (Fig. 15b).

### How Phase 1 differs from the paper

| Aspect | Paper | Phase 1 local |
|---|---|---|
| Model | Llama-3.1-8B-Instruct; Qwen3-235B-A22B-FP8 | Qwen2.5-1.5B-Instruct (bf16) |
| GPU | 8× RTX PRO 6000 96 GB; 4× H200 | 1× RTX 4060 Laptop 8 GB (≈370 MiB used by Windows) |
| vLLM | v0.11 | 0.31.0 (see open_questions A1) |
| KV budget | 100–200 blocks (Fig. 14); main budget unknown | Explicit `num_gpu_blocks_override` 100–200 (+ larger) |
| Workloads | GSM8K, MT-Bench, GAIA, SWE-bench via 6-agent AutoGen | Synthetic Fig. 5 traces first; small AutoGen runs on GSM8K/MT-Bench; GAIA/SWE-bench prompt-only with simulated tools |
| Sessions / load | 0.2–50 sessions/s | ≤ 200 sessions, ≈ 0.1–2 sessions/s, `max_num_seqs` ≤ 8 |
| Turns, outputs | unspecified | ≤ 13 turns, max_tokens ≤ 256, max_model_len 4096 |
| CPU offload tier | used | not used initially |
| Continuum | authors' vLLM fork | our TTL-pin re-implementation (interpretation) |

---

## M1. Verified environment + vanilla vLLM baseline + metrics (local)
**Goal:** reproducible vanilla vLLM runs with the paper's four metrics.
- [x] Environment inspected; versions recorded in CLAUDE.md; `docs/env_freeze_before.txt` saved
- [x] Smoke test (imports, CUDA): `pytest -q tests/test_smoke_env.py`
- [x] **Deliberate** vLLM generation smoke test passed 2026-10-06 (`smoke_run3.log`): 256 blocks, override OK, ~3.7 GiB used, eager mode
- [x] `num_gpu_blocks_override` honoured at runtime (256 allocated, 255 usable: vLLM reserves one null block)
- [x] Budget-sweep configs use max_model_len 1584 (fits 100 blocks)
- [x] Metrics module (`src/cachescout/metrics/`): hit rate, TTFT, per-turn latency, throughput; unit-tested; vLLM sources in `docs/vllm_internals.md`
- [x] GPU check `scripts/check_prefix_metrics.py` passes 10/10 (results/m1_metrics_check/20261006-153348)
- [x] Online driver with streaming TTFT (`python -m cachescout.run`, AsyncLLM in-process instead of an HTTP server)
- [x] Baseline runner: `python -m cachescout.run --system vanilla` (online + sequential modes)
- [x] Results saved with resolved config + git commit + versions (result.json)
**Verify:** a two-request prefix test where the 2nd request shows cached tokens ≈ shared prefix; a block-budget test
where an LRU-evicted prefix shows 0 cached tokens.
**Working local settings (2026-10-06, `configs/hardware/local.yaml`):**
| Setting | Value | What it fixed |
|---|---|---|
| `enforce_eager` | true | Avoids torch.compile/Inductor autotuning and CUDA-graph capture (memory spikes; 1st OOM was reported there). Applied to all compared systems. |
| `num_gpu_blocks_override` | 256 | The real OOM: `allocate_kv_cache` tried 1.23 GiB (derived from `gpu_memory_utilization`). Now ~112 MiB. 255 usable (1 null block). |
| `max_model_len` | 2048 | 128 blocks per full-length sequence, so it fits in 255 usable blocks (≈2 concurrent full-length sequences). |
Measured: weights 2.98 GiB; total GPU used ~3.7 GiB (incl. ~0.4 GiB Windows desktop).

**Risks:** WSL memory limits; metric APIs differ in 0.31.

## M2. Agentic workload / trace generation
**Goal:** traces of (session, turn, agent, prompt tokens, output length, arrival time).
- [x] Synthetic generator: 6 agents, Fig. 5 matrices, anchor share 0.54-0.62, seeded Poisson arrivals
- [x] Trace stats (`scripts/make_traces.py` → results/traces/stats.json)
- [ ] ~~Real AutoGen runs~~ not done: synthetic traces used (see decisions.md / REPORT)
**Verify:** R measured on the synthetic traces ≈ the paper values for each matrix (1.0 / 0.78 / 0.57 / 0.12) within tolerance; unit tests.
**Risks:** the 1.5B model routes differently from the 8B (open_questions E3); unreleased prompts (C1).

## M3. Cache-management component (simulator first)
**Goal:** Transition Learner + Survival Scorer + LRU in a pure-Python block-cache simulator (no GPU).
- [x] Simulator (exact vs vLLM on 703/703 requests, sequential)
- [x] Transition Learner, graph + BFS, Score (Eq. 9), Alg. 1 (`src/cachescout/core/`)
- [x] Continuum-style TTL baseline (soft pinning, interpretation)
- [x] Fig. 14a trend in simulation (tuning traces): gain shrinks with budget
**Verify:** unit tests per equation against hand-computed values; LRU limit (Score → recency-only) matches the plain LRU simulator.
**Risks:** simulator may not match vLLM's real eviction behaviour, so M4 cross-checks it.

## M4. Learned component in vLLM (runtime plugin)
**Goal:** the same logic hooked into vLLM 0.31's block pool / eviction, plus warmup.
- [x] Hook mechanism: `scheduler_cls` subclass of AsyncScheduler (user granted autonomy)
- [x] ObserveTouch, fingerprint (2 blocks), ScoreBlock victim selection, CACHESCOUT_CONFIG env var
- [x] Warmup via AsyncLLM API, R ≥ R_min gate, rate-limited, `cswarm-` excluded from learning
- [ ] Microbenchmarks for Fig. 15 (state size, ObserveTouch/PredictSurvival latency)
**Verify:** vLLM-with-plugin hit rate matches the simulator on the same trace within a few pp; with the plugin
disabled, results equal vanilla vLLM.
**Risks:** engine core runs in a separate process (the patch must load there); internal API drift.

## M5. Evaluation against baselines (local)
- [ ] vLLM vs Continuum-TTL vs CacheScout on all local workloads (hit rate, mean/median/P99 TTFT, per-turn latency, throughput)
- [ ] Ablation: vLLM / eviction only / prefetch only / full (Fig. 12)
- [ ] Block budget sweep 100–200+ (Fig. 14a); prefetch gate on/off (Fig. 14b)
- [ ] Topology sweep Pipeline/Debate/Selector/Random
- [ ] Hyperparameter sensitivity (open_questions B1); 3 seeds; write-up vs paper
**Verify:** `experiment-reviewer` agent passes; every result has config + versions + seed.

## M6. Cloud scale-up
- [ ] `configs/hardware/cloud.yaml` for A100-80GB / H100; Llama-3.1-8B-Instruct; larger budgets and paper-range loads (0.2–50 sess/s)
- [ ] Re-run M5 at paper scale; optional larger model (Qwen3-235B is likely out of budget, so a mid-size substitute)
- [ ] Optional: CPU-offload tier + CPU-resident preference (open_questions A3)
**Verify:** compare with the paper's absolute numbers; document every remaining gap honestly.
**Risks:** cost; vLLM version on the cloud image must equal the local one (0.31.0) unless decided otherwise.
