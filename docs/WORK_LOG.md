# Work log

Running log of autonomous work (newest at the bottom). Times are local (WSL clock).
Resume with `claude --continue`; the last entry says what was in progress.

## 2026-10-06 15:32 Start of autonomous phase
- User granted autonomy for M1–M5 (+ M6 prep). Safety rules: one GPU job at a time, `timeout`, logs under logs/,
  check `free -h` + `nvidia-smi` before each run, keep enforce_eager=True and num_gpu_blocks_override.
- Added `*.log` and `logs/` to .gitignore.

## 2026-10-06 15:34 M1 check_prefix_metrics.py run 1: PASS 10/10
- Ran: `timeout 900 python scripts/check_prefix_metrics.py --config configs/experiments/m1_metrics_check/local.yaml`
  → `logs/check_prefix_run1.log`, `results/m1_metrics_check/20261006-153348/result.json`.
- (a) B.cached=64 ✓, (b) repeat 96-token prompt cached=80 ✓ (last-token rule), (c) A cached 192 then 0 after
  304 filler blocks ✓ (LRU eviction at 256 blocks works), (d) counters match exactly (5792/5792, 336/336).
- Issue found: log shows "unauthenticated requests to the HF Hub": the script did not apply the hardware config's
  `env: HF_HUB_OFFLINE=1`. No download happened (model cached), but fixing: config loader will apply env vars.

## 2026-10-06 15:50 Design of the vLLM hook (read-only source study)
- vLLM 0.31.0 default scheduler is `AsyncScheduler` (async_scheduling auto-enabled, `config/vllm.py:1576-1635`).
- Official extension point: `SchedulerConfig.scheduler_cls` (`config/scheduler.py:229-254`), resolved by qualname
  inside the engine-core process. Plan: `CacheScoutScheduler(AsyncScheduler)` installs instance-level wrappers on
  its own `kv_cache_manager` / `block_pool` (no site-packages edits):
  - `get_computed_blocks(request)` wrapper → ObserveTouch at prefix-matching time (fingerprint from
    `request.block_hashes[:k]`), Alg. 1 lines 10-21;
  - `allocate_slots` wrapper + `block_pool.get_new_blocks` replacement → map new blocks to the request's agent and
    choose victims by Eq. 9 instead of popping the LRU head (Alg. 1 line 9);
  - `block_pool.free_blocks` wrapper → last-access time = free time (matches vLLM LRU semantics);
  - `schedule()` override → scheduler-step counter (age unit, Eq. 9).
- Config passed via one env var `CACHESCOUT_CONFIG` (JSON), mirroring the paper's single env-var toggle (Sec. 4).
- Vanilla runs use the stock scheduler (no scheduler_cls). A `policy: lru` mode of our scheduler is used to verify
  "hook installed but neutral == vanilla".
- Warmup (Sec. 3.4) is issued by the client-side driver through the normal engine API (`max_tokens=1`, request-id
  prefix `cswarm-`, which the engine-side runtime excludes from transition learning).

## 2026-10-06 15:55 Installed matplotlib 3.11.2
- Freeze saved to docs/env_freeze_before_matplotlib.txt; dry run: 7 new packages (contourpy, cycler, fonttools,
  kiwisolver, matplotlib, pyparsing, python-dateutil), no changes to existing packages. Installed; diff confirms.

## 2026-10-06 16:10 M3 core written (pure Python, shared by simulator and vLLM hook)
- `src/cachescout/core/`: learner.py (Eqs. 2-5), scorer.py (Eqs. 7-9, BFS), runtime.py (Alg. 1 state machine,
  policies lru/cachescout/continuum, global|session scope), warmup.py (Sec. 3.4 coordinator, Eqs. 10-11).
- tests/test_core.py: 21 tests with hand-computed values, all pass.

## 2026-10-06 16:40 M2 trace generator + statistics
- Real AutoGen runs NOT used: a 1.5B selector would not reproduce the paper's routing and adds a server +
  framework dependency; synthetic traces from the Fig. 5 transition tables are used (allowed fallback).
- Generator: 6 agents (P,A,C,T,R,D), anchor + shared group-chat history, multi-call invocations for tool agents
  (A, C, T). Workload lengths calibrated against the paper's *motivation statistics only* (never against policy
  results). `python scripts/make_traces.py` → results/traces/{*.json, stats.json}.
- First version: history reuse 0.7/block and Selector R 0.48. Fixes: (1) multi-call invocations; (2) Selector column
  placement chosen by exhaustive search for analytic R=0.570 (planner→coder dominant).
- Result (results/traces/stats.json): anchor share 0.57-0.63 (paper 0.53-0.62); R selector 0.54-0.56 (0.57),
  debate 0.76-0.77 (0.78), random 0.14 (0.12), pipeline 1.00; phi at turn 12-14 = 0.46-0.43 (paper 43-60%);
  max request footprint ≤ 90 blocks (paper ~93).
- Remaining deviation: session-history block reuse ≈1-2 vs paper 12-15 (Fig. 3a): anchors are relatively more
  valuable in our workload → possible bias in CacheScout's favour. Will be reported as a threat to validity.
- Observation: online next-agent accuracy with interleaved sessions (literal global a_t) is only ~0.40 vs 0.58-0.72
  per session → evidence for open question B4 (scope).
