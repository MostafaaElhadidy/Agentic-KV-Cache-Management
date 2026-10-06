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
- 16:10 Stage 3: full-size tuning trace (selector_tune, 60 sessions/703 calls), vanilla, 100 blocks.
- 16:13 Stage 3 PASS: full tuning trace vanilla @100 blocks: 703 turns, hit 0.585 (simulator predicted 0.597), peak GPU 3878M, RAM 3672M. Crash #1 not reproduced; most likely cause = concurrent CPU sweep during model load. Rule kept: nothing else runs during GPU jobs. Next: re-run tuning sweep (CPU only, no GPU job).
- 16:24 Tuning sweep v2 (512 configs, session-mean scope added): best still global, 0.6954 vs vanilla
  0.6930 (results/tuning/eviction_sweep_v1_all_mapping.json). Session scope did not help.
- Diagnostic: Belady/MIN upper bound in the simulator (sequential order, tuning trace): vanilla 0.588/0.672/0.712,
  Belady 0.711/0.764/0.771 at 100/150/200 → real headroom (+12/+9/+6 pp) that CacheScout did not capture.
  Region breakdown @100: Belady raises anchor-region hit 0.84→0.95 and history 0.26→0.40; CacheScout 0.84→0.85.
- Root cause: Alg. 1 line 12 maps EVERY touched block (incl. session history) to the agent, so stale history of
  hot agents (P, C) inherits high survival and displaces anchors of less frequent agents.
- Change (interpretation of Sec. 3.3 "inherits the survival score of its corresponding agent anchor", open question
  B3): `block_mapping: anchor_only` — only blocks shared by >= 2 sessions (= anchor blocks, detected without
  labels) inherit survival; history blocks rank by recency only. Probe (sequential, tuning trace):
  anchor_only 0.626/0.690-0.697/0.724-0.730 vs vanilla 0.588/0.672/0.712 (+3.8/+2.5/+1.8 pp).
- Next: sweep v3 with block_mapping ∈ {all, anchor_only}.
- 16:33 Sweep v3 (results/tuning/eviction_sweep.json, tuning traces only): best = scope session(8, mean)
  + block_mapping anchor_only, tau 0.1, E_max 6, lambda 0.005/step, delta 0.01: mean hit 0.7096 vs vanilla 0.6930.
  Literal (global/all) best 0.6954; global/anchor_only 0.7084; session/all 0.6890. tau=0 gives 0.7108 (flat
  survival → gain then comes only from anchor-vs-history protection); kept tau 0.1 so Eq. 7 stays meaningful and
  will report a tau=0 "no-prediction" ablation.
- Warmup sweep (results/tuning/warmup_sweep.json): hit-rate objective flat (0.7089-0.7095); argmax r_min 0.7 simply
  disables warmup. Chose r_min 0.3 (gate on for structured topologies, off for Random R≈0.12), interval 2.0 s.
- 16:34 M4 cross-check: sequential GPU runs on tuning trace @100 blocks (vanilla, lru_hook, eviction_only) vs simulator.
- 16:48 M4 VERIFIED on GPU (sequential mode, tuning trace, 100 blocks; crosscheck.json next to each result):
  - vanilla vs simulator: 703/703 requests exact, hit 0.5878 both (results/tune_gpu/vanilla/b100_seq/).
  - lru_hook (hook installed, neutral) vs vanilla: identical, 703/703 exact (results/tune_gpu/lru_hook/b100_seq/).
  - eviction_only (tuned constants): GPU hit 0.6248 vs vanilla 0.5878 (+3.7 pp); simulator 0.6214, 97.9% of
    requests exact, mean |diff| 2.0 tokens (step-count/age differences) (results/tune_gpu/eviction_only/b100_seq/).
  - Fidelity level achieved: full vLLM hook (scheduler_cls extension point), no fallback needed.
- 16:53 M5: compare.py written (run/load GPU results or --sim; table, trend checks vs paper, plot).
  Simulator on EVAL trace (results/main/compare/selector_eval_sim_eval_20261006-165243.*): CacheScout +3.3/+2.6/+1.0
  pp @100/150/200; literal Alg. 1 +0.2/-1.0/-0.9; no_prediction (tau=0) +3.0/+3.1/+1.6 (≥ full at 150/200);
  Continuum == vanilla exactly (soft TTL pins coincide with LRU's most-recent blocks → degenerate baseline).
  Palette validator (node) unavailable; used the skill's pre-validated reference palette + marker shapes.
- 16:53 Starting GPU main sweep (selector_eval, 5 systems × 100/150/200 blocks) via compare.py --run.
- 16:53 compare.py --run refused by gpu_run.sh preflight (it matched the compare.py orchestrator itself). Fixed: orchestrator excluded. Restarting sweep.
- 17:27 GPU main sweep done (15 runs, all exit 0, ~2:10 each). results/main/compare/selector_eval_gpu_eval_20261006-172634.*
  Hit gain CacheScout vs vanilla: +1.8/+2.8/+1.6 pp @100/150/200 (sim predicted +3.3/+2.6/+1.0; sequential GPU
  cross-check @100 on tuning trace gave +3.7). TTFT/latency: no significant difference (±1-3%, noise) — with
  ~340-token prompts a 2 pp hit gain saves ~7 prefill tokens/request, negligible vs batching/queueing on a 1.5B
  model. Eviction-only ≥ warmup-only ✓; warmup-only +0.5..+1.1 pp; Continuum ≈ vanilla. "Gain shrinks" passes
  only marginally and is non-monotone (largest at 150) → not claimed.
- Next: scripts/run_all_local.sh steps 2-5 (variants, seeds, topologies, load) in background.
- 19:12 Campaign done: 57 GPU runs total, all exit 0 (logs/run_all_local.log, results/main/compare/*).
  Highlights: pipeline +13.1 pp @100 (TTFT -8.7%); debate +1.3/+0.8; random +3.3/+5.0 (NOT ~0 as paper expects:
  gain comes from anchor protection, cf. no_prediction ≈ full); literal Alg. 1 +0.4/+0.1/-1.0; seeds 2/3 selector
  gains +2.6/+1.8/+0.8 and +3.5/+2.1/+1.2; TTFT/latency/throughput differences inconsistent (noise level).
  Load sweep @150: throughput saturates ~15.5-16 turns/s for both systems.
- 19:16 Overhead microbench (results/microbench/overhead.json). Optimised threshold_graph (row totals once,
  graph built once per refresh): identical simulator results, observe 100→37 us @6 agents, 2005→172 us @24.
  (GPU runs used the slower version; affects overhead only, not decisions.)
- 19:16 scripts/aggregate_report.py → results/report/{summary.json,tables.md,fig_*.png};
  docs/REPORT.md written (all numbers cite result files). Removed one unsupported claim (run-to-run variance was
  never measured). PLAN.md / CLAUDE.md updated. Final commit next.

## 2026-10-06 20:56 Public release on GitHub
- Audit: personal email only in commit author fields; secrets none; no files > 5 MB; paper PDF kept (embedded
  license CC BY 4.0, attribution in paper/README.md); setup_prompt.md and smoke_run2.log removed from history.
- Backup: ~/CacheScout_backup.bundle. History rewritten with git-filter-repo (temporary uvx env): all authors and
  committers = Mostafa <110402955+MostafaaElhadidy@users.noreply.github.com>; verified 0 occurrences of the
  personal email in patches and messages. Local (not global) git identity set to the noreply address.
- Added README.md, LICENSE (MIT), paper/README.md; .gitignore updated; small cited results tracked.
- Pushed main to https://github.com/MostafaaElhadidy/Agentic-KV-Cache-Management (remote was empty; no force).
  Visibility: PUBLIC. Description and topics (kv-cache, vllm, llm-serving, agentic-ai, paper-replication,
  llm-inference) set via gh.

## 2026-10-06 22:22 Real-agents phase started (branch real-agents)
- Plan approved (docs/REAL_AGENTS_PLAN.md + amendments). GSM8K download approved (test/train jsonl, openai repo).
- NEXT STEP: fetch GSM8K, build src/cachescout/agents/, driver, tests; then smoke; then PILOT GATE (stop).
- 22:30 GSM8K fetched (scripts/fetch_gsm8k.sh; checksums in data/gsm8k/SHA256SUMS). Built
  src/cachescout/agents/ (definitions, tools, routing, prompting+trimming, session, replay, summary, gsm8k),
  run.py mode: agents + record/replay loading + --set, compare.py --set/agents support, show_session.py.
  Anchors 150-283 tokens, 6/6 distinct 32-token fingerprints with the real tokenizer. Tests: 115 passed.
- NEXT STEP: smoke runs (1 session per topology, vanilla), then one cachescout smoke, then PILOT GATE.
- 22:32 Smoke 1 (pipeline, vanilla, 1 session; used ONE test problem test:844 — prompt iteration from now on uses train problems only): plumbing OK; CODER wrote Python instead of CALL lines, outputs hit 128-token cap. Clarified tool/format instructions in anchors.
- 22:35 Smoke 2 (selector, train seed 101, 2 sessions): fallback rate 0.875 — model writes '[PLANNER]: DECIDER' instead of NEXT lines; old fallback looped P<->C until cap. Added lenient name-only parsing (counted) + round-robin fallback; decisions.md updated.
- 22:39 Smoke 3 (selector, train seed 101): strict NEXT lines 2/15 (13%), fallbacks 13/15, final answer
  3/3, tool failures 0/3. Stopped prompt iteration (1.5B model mostly ignores the NEXT protocol -> report as
  finding). git_dirty now ignores untracked files (results/), so committed code => git_dirty=false.
- NEXT STEP: PILOT GATE: selector @100 blocks, 5 sessions, train seed 102 (tuning problems), vanilla + cachescout
  live; then STOP and report to the user. Do NOT start the campaign.
- 22:42 PILOT GATE done (selector, 100 blocks, 5 sessions, train seed 102 = tuning problems
  train:4868,4549,1141,6014,483; commit 038f48e, git_dirty=false):
  - vanilla (results/real/gsm8k_train_selector_s102/vanilla/b100_pilot/result.json): fallback 31/38 (0.816),
    strict NEXT 0.184, tool failures 0/4, FINAL ANSWER 4/5, accuracy 2/5, trim 0/46, hit 0.233, TTFT 603 ms.
  - cachescout (.../cachescout/b100_pilot/result.json): fallback 20/29 (0.690), tool failures 1/10, FINAL
    ANSWER 5/5, accuracy 3/5, trim 0/42, hit 0.317, TTFT 251 ms, 20 warmups, engine prediction acc 0.66.
  - Same problems + temperature 0, but trajectories DIFFER between systems (46 vs 42 calls, different answers):
    live closed-loop outputs are not deterministic across engine configurations (batching/numerics), so live
    differences (incl. accuracy) are not attributable to CacheScout -> replay is the controlled comparison.
- STOPPED at the pilot gate, waiting for the user's "go". NEXT STEP after go: sim-vs-GPU sequential cross-check on
  recorded real sessions (amendment 2), then tuning on train problems, then campaign (replay first, then live).
- 22:47 User said GO with amendments (router pilot first; then sim cross-check, tuning, campaign replay->live). Implemented AutoGen-style separate SELECTOR call constrained via vLLM 0.31 SamplingParams(structured_outputs=StructuredOutputsParams(choice=[...])) (vllm/sampling_params.py:88-92,341; backends xgrammar/llguidance installed). Agent cap counts agent calls only. 119 tests pass. NEXT: router pilot (selector, 100 blocks, train seed 102, 5 sessions, vanilla).
- 22:50 Router pilot (results/real/gsm8k_train_selector_s102/vanilla/b100_pilot_router/result.json):
  100% model routing, but DECIDER chosen 64.7% (>60%) -> degenerate (P<->D alternation), R 0.74, +370 tok/turn.
  NOT adopted (user criterion). Keeping NEXT-line selector, no more prompt tuning; fallback rate to be reported.
- NEXT STEP: record tuning sessions (vanilla live @100, train seed 101, 20 sessions x 4 topologies), then
  sim-vs-GPU sequential cross-check on a recording, then tuning, then campaign.
