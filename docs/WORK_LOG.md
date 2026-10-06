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

## 2026-10-06 17:20 Simulator, workload calibration, first tuning: CacheScout ≈ LRU (+0.24 pp). Investigated.
- Simulator made concurrent (max 8 running requests, blocks held until completion, FIFO admission), because the
  sequential version was overloaded (warmups never ran, Continuum TTL always expired).
- Workload calibration (scripts/calibrate_workload.py, vanilla only, tuning seed): base profile gave vanilla
  31-45% at 100-200 blocks vs paper 64-77% (Fig. 14a). Scale 0.4 → vanilla 0.60/0.68/0.72, anchor share 0.58,
  phi@12 0.44. Deviation: max request footprint 42 blocks (paper ~93). Longer sessions were tried and lowered both
  vanilla hit rate and anchor share far below the paper → rejected. Profile written to configs/traces/local.yaml.
- Tuning sweep 1 (384 configs, eviction_only, tuning traces): best 0.6954 vs vanilla 0.6930 → negligible gain.
- Diagnosis (miss breakdown on selector_tune): engine-side learner with the literal GLOBAL current agent sees
  ~8 interleaved sessions → learned R = 0.18, top-1 accuracy 0.43 (true R 0.57) → survival flat → LRU
  (the paper's own graceful-degradation case). Session scope with multi-source BFS also flattens.
- Change (interpretation, B4): session scope with per-session Eq. 8 tables averaged per agent
  (`session_aggregate: mean`). Rerunning the sweep with scope ∈ {global, session×{4,8,16} active}.

## 2026-10-06 (after restart) CRASH #1 during first GPU run of the experiment runner
- Running when WSL/session died: `PYTHONPATH=src timeout 900 python -m cachescout.run
  --config configs/experiments/smoke/local.yaml --system vanilla > logs/smoke_vanilla_1.log`
  (AsyncLLM, vanilla, smoke trace: 4 sessions / 42 turns, 200 blocks, max_model_len 1584).
- ALSO running at the same time: the CPU-only tuning sweep `scripts/tune_constants.py` (background, started
  ~17:20). So two heavy Python processes were alive, not one.
- Counts as crash #1 for the step "first GPU run of the runner".
- Diagnosis (2026-10-06 16:06):
  - Previous boot ended 16:02:06, ~22 s after the runner started (results/smoke/vanilla/b200_20261006-160144
    created 16:01:44; no engine stats file → died during engine start-up: weight loading/profiling phase, cf. the
    successful check run where this phase spans +12 s..+28 s).
  - Logs empty (0 bytes): unflushed page cache lost when the VM died.
  - Kernel log of previous boot: no OOM-killer messages; only "dxg: dxgkio_query_adapter_info: Ioctl failed: -22",
    which appear at a constant ~52/min since 14:55 including during the successful 15:33 GPU run → background noise,
    not the cause. Abrupt VM end without Linux-side errors suggests the Windows host killed/reset the VM (host
    memory pressure or GPU driver reset).
  - After restart: free 10 GiB RAM, swap 0 used, GPU 193 MiB used, no python/vllm leftovers.
  - Differences vs the runs that worked (smoke script, check_prefix_metrics): same gpu_memory_utilization 0.6,
    enforce_eager, max_num_seqs 8; smaller max_model_len (1584) and budget (200); AsyncLLM vs LLM (both spawn one
    engine-core process). Only material difference: the CPU tuning sweep ran concurrently (two heavy Python
    processes) during model loading → likely host memory pressure.
  - NOTE: earlier WORK_LOG timestamps (16:10/16:40/17:20) were estimates, not clock readings; actual clock was
    ~15:40-16:02. From now on timestamps come from `date`.
- Fixes: (1) never run anything else heavy during a GPU job; (2) max_num_batched_tokens 2048 (default 8192) to cut
  the profiling activation peak, applied to all systems; (3) scripts/gpu_run.sh: preflight checks + 2-s memory
  monitor that fsyncs to logs/ so a crash leaves evidence; python -u for unbuffered logs.
- Plan: tiny (1 session) → smoke (4 sessions) → full trace, one GPU process at a time.
- 16:07 Stage 1 starting: tiny trace (1 session, 13 calls), vanilla, via scripts/gpu_run.sh
- 16:08 Stage 1 PASS: tiny vanilla, 13 turns, hit 0.624, peak gpu_used=3878M peak ram_used=3611M min avail=8344M. Stage 2 (smoke, 4 sessions) next.
- 16:09 Stage 2 PASS: smoke vanilla 42 turns hit 0.738, peak gpu_used=3878M peak ram_used=3632M min avail=8323M. Stage 2b: smoke with CacheScout hook.
- 16:10 Stage 2b PASS: smoke with CacheScoutScheduler hook (engine core loaded it; 42 dispatches, 15 warmups excluded, acc 0.74, observe 43 us, select 27 us). Hit = vanilla (no pressure at 200 blocks on 4 sessions).
