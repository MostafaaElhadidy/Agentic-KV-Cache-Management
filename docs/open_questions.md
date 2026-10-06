# Open questions

Each entry: what's unclear → options → **recommended default** (marked as *interpretation* in code).
Status: `OPEN` (needs your decision or more info) / `DEFAULT` (using the default until told otherwise) / `RESOLVED`.
Paper references use docs/paper_notes.md conventions.

## A. vLLM version and internals

### A1. vLLM v0.11 (paper) vs 0.31.0 (installed). OPEN, top priority
The paper patches vLLM v0.11 V1 internals (Sec. 4). We have 0.31.0 and must not upgrade or downgrade it.
Read-only check (2026-10-06): the hook points still exist in 0.31.0: `vllm/v1/core/block_pool.py`
(`BlockPool.touch`, `get_new_blocks`, `_maybe_evict_cached_block`, `evict_blocks`),
`vllm/v1/core/kv_cache_utils.py::FreeKVCacheBlockQueue`, `CacheConfig.num_gpu_blocks_override`,
`distributed/kv_transfer/kv_connector/v1/offloading_connector.py` and `simple_cpu_offload_connector.py`.
Signatures and behaviour may still differ from v0.11.
- Options: (a) monkey-patch/subclass `BlockPool` and the free queue at runtime from our package (no edits to
  site-packages); (b) maintain a patch file against 0.31.0 site-packages; (c) separate env with v0.11 (needs your approval).
- **Recommended: (a)**, a runtime plugin that swaps the eviction ordering. Must work in the engine-core process
  (V1 runs the scheduler in a separate process), so the patch must be applied via a vLLM plugin entry point or env-var
  hook that loads in that process. **Will ask before any workaround.**

### A2. Eviction granularity in vLLM V1. OPEN
V1 evicts by popping the *head* of an LRU free queue (O(1)). Eq. 9 needs `argmin` over candidates (Alg. 1 line 9).
- Options: (a) full scan of free cached blocks per eviction; (b) score only the first N candidates at the queue head;
  (c) maintain a priority structure re-keyed once per scheduler step (paper: "survival probabilities are computed once
  per scheduler step and cached").
- **Recommended: (c) with (a) as a correctness reference in tests.**

### A3. CPU-tier preference in eviction. DEFAULT
Sec. 4 says eviction additionally prefers blocks already resident in CPU. Not in Eq. 9.
- **Recommended:** omit in phase 1 (no CPU offload locally); add as a tie-breaker flag `prefer_cpu_resident` for cloud.

### A4. Prefix caching configuration. DEFAULT
**Recommended:** `enable_prefix_caching: true`, block_size 16 (Sec. 2.1), identical across all systems.

## B. Algorithm details

### B1. Hyperparameters ε, τ, E_max, λ, δ, R_min. OPEN (all unspecified)
| Param | Role | Recommended default | Sweep |
|---|---|---|---|
| ε | Eq. 3 smoothing | 0.01 | {0.001, 0.01, 0.1} |
| τ | Eq. 7 edge threshold | 0.5 (the Fig. 7 example) | {0.1, 0.2, 0.3, 0.5} |
| E_max | Eq. 8 horizon | 3 | {2, 3, 4, 6} |
| λ | Eq. 9 recency decay | 0.05 per scheduler step | {0.01, 0.05, 0.2} |
| δ | Eq. 9 floor | 0.05 | {0.01, 0.05, 0.1} |
| R_min | Eq. 11 gate | 0.3 | {0.0, 0.3, 0.5} |
Note: with 6 agents and τ = 0.5, many Selector rows (Fig. 5) have no edge ≥ 0.5, so the graph could be very sparse.
A sensitivity sweep is part of M5.

### B2. What is the "agent" / fingerprint? OPEN
Sec. 4: "each request is fingerprinted from its own prefix block hashes". Number of blocks unspecified.
- Options: (a) hash of the first k full blocks (k = 1..4); (b) longest prefix shared with a known anchor; (c) explicit
  agent ID passed by our workload driver (only for validation, since the paper needs no framework cooperation).
- **Recommended: (a) with k configurable (default 2 = 32 tokens)**, validated against ground-truth agent labels from
  our trace generator. Risk: agents with a common system-prompt preamble would collide, so k must cover the
  agent-specific part.

### B3. Which blocks map to an agent? OPEN
Alg. 1 line 12 sets `agentOf[b]` for every touched block, including session-history blocks, not just anchor blocks.
- **Recommended:** follow Alg. 1 literally (all blocks of the request → the request's agent); also log an
  "anchor-only" variant as an experiment.

### B4. Global vs per-session current agent. OPEN
Alg. 1 has a single global `a_t`. With concurrent sessions, interleaved requests from different sessions would create
spurious transitions.
- Options: (a) global, literal; (b) per-session `a_t` keyed by a session fingerprint; (c) per-request-stream.
- **Recommended: (a) literal** (it's what the paper specifies); measure how prediction accuracy degrades with
  concurrency, and test (b) as an ablation, labelled *interpretation*.

### B5. age(b) unit. DEFAULT
Eq. 9 text: scheduler steps; Alg. 1: `now() − lastAccess`. **Recommended:** scheduler steps (the prose definition).

### B6. |b| when block size is fixed. OPEN
In vLLM every full block holds 16 tokens, so |b| is constant except for a partial last block (which isn't cached anyway).
- Options: (a) literal |b| (effectively a constant); (b) prefix depth (tokens up to and including b, so deeper blocks
  cost more because the whole chain must be recomputed); (c) number of dependent descendant blocks.
- **Recommended: (a) literal**, flagged; (b) as an ablation.

### B7. Online estimate of R (Eq. 2) for the gate. OPEN
- **Recommended:** plug-in entropies from transition counts: `H(A_{t+1})` from successor marginals, `H(A_{t+1}|A_t)` as
  row entropies weighted by row counts; recompute when the current agent changes.

### B8. Sliding window vs cumulative counts. DEFAULT
Fig. 6 shows a "Sliding window"; Sec. 3.2 uses cumulative counts. **Recommended:** cumulative (matches the equations),
with an optional `window: null|int` config.

### B9. Warmup details. OPEN
Rate limit unspecified; "minimal user prompt" unspecified; how the engine excludes warmups from learning is unspecified.
- **Recommended:** ≤ 1 warmup per agent transition, skip if that anchor is already fully cached, minimal user prompt =
  a single fixed short token string, warmups tagged via a request-ID prefix (`cs-warmup-`) that `ObserveTouch` ignores.

### B10. Survival horizon K (Eq. 6). DEFAULT
Never set; E_max plays this role. **Recommended:** don't implement Eq. 6 except as an offline evaluation metric.

## C. Workloads and data

### C1. The six-agent supervisor framework. OPEN, major
Agents, prompts, tools and anchor lengths are not released. Fig. 5 uses labels P, A, C, T, R, D.
- Options: (a) write our own 6-agent AutoGen SelectorGroupChat (Planner, Analyst, Coder, Tester, Reviewer, Doc-writer
  [our guess for the labels]); (b) synthetic traces only (sample from Fig. 5 matrices); (c) wait for the release.
- **Recommended: (b) first** (synthetic traces from the Fig. 5 matrices, anchor sizes chosen so the anchor share is
  ~53–62% as in Fig. 2), **then (a)** at small scale in M2. Watch for the code release.

### C2. How each dataset becomes multi-agent sessions. OPEN
Number of tasks/sessions, turns per session, and output lengths are unspecified.
- **Recommended local defaults:** 50–200 sessions per run, ≤ 13 turns (Fig. 3b x-axis), max_tokens 128–256.

### C3. Arrival process. OPEN
"Session arrival rate" is given but not the distribution. **Recommended:** Poisson session arrivals, seeded.

### C4. GAIA and SWE-bench access. OPEN
GAIA is gated on Hugging Face; full SWE-bench tool execution (Docker) is heavy.
- **Recommended:** locally, use prompts/issue text only with *simulated* tool outputs of realistic length; don't
  execute repos.

## D. Baselines and evaluation

### D1. Continuum baseline. OPEN
Continuum is a separate vLLM fork (arXiv 2511.02230) and will not match vLLM 0.31.
- **Recommended:** re-implement "TTL pinning (0.3 s) of the session's blocks after a tool-call boundary" as a policy
  inside our harness, labelled *interpretation*, and verify behaviour on synthetic traces.

### D2. Which configuration produced each result? OPEN
See paper inconsistencies #2 and #3. **Recommended:** report both the main-results setup and the Fig. 14 sweep.

### D3. Seeds, repetitions, decoding. DEFAULT
Unspecified. **Recommended:** temperature 0 for the LLM-driven routing traces once recorded; replay recorded traces for
fair comparison across policies; 3 seeds for each simulated run.

### D4. What the GPU memory budget was in the main results. OPEN
Unknown. **Recommended:** use explicit `num_gpu_blocks_override` budgets (e.g. 100/150/200 blocks, plus 1.5× and 2× the
workload footprint) so local and cloud runs are comparable in *blocks*, not GB.

## E. Local hardware (8 GB) adaptations

### E1. Model substitution. RESOLVED (user, 2026-10-06)
Qwen2.5-1.5B-Instruct (cached) replaces Llama-3.1-8B-Instruct locally. KV per token ≈ 28 KiB (28 layers × 2 KV heads
× 128 dim × K,V × bf16), ≈ 448 KiB per 16-token block, so 200 blocks ≈ 88 MiB. The paper's block budgets fit easily.

### E2. `--enforce-eager`. RESOLVED (2026-10-06)
Enabled locally after a torch.compile autotune OOM; applied to all systems. See decisions.md.

### E3. Small model routing quality. OPEN
A 1.5B model may choose agents poorly in SelectorGroupChat, so its transition structure may differ from Fig. 5.
**Recommended:** measure R (Eq. 2) on our recorded traces and report it next to the paper's.

---

## Resolutions after the autonomous phase (2026-10-06)

| Id | Status | Resolution (details in docs/decisions.md, evidence in docs/WORK_LOG.md) |
|---|---|---|
| A1 | RESOLVED | No downgrade. Hook via vLLM 0.31 `scheduler_cls` (CacheScoutScheduler ⊂ AsyncScheduler), instance-level wrappers, no site-packages edits. Verified: neutral hook == vanilla on 703/703 requests. |
| A2 | RESOLVED | Victim selection scans the free queue per allocation (≤ 255 blocks locally), sorts by Eq. 9 with LRU position as tie-breaker; `lru` policy uses the original `popleft_n`. |
| A3 | DEFAULT | CPU-tier preference not implemented (no CPU offload locally). |
| B1 | RESOLVED | Tuned in the simulator on tuning traces only: eps 0.01, tau 0.1, E_max 6, lambda 0.005/step, delta 0.01, R_min 0.3, warmup interval 2 s. |
| B2 | RESOLVED | Fingerprint = first 2 block hashes (32 tokens), engine side; same token prefix hash on the client. |
| B3 | RESOLVED (interpretation) | `anchor_only`: only blocks shared by ≥ 2 sessions inherit survival. Literal "all blocks" kept as variant `cachescout_literal`. Reason: literal mapping let stale history displace anchors (Belady diagnostic). |
| B4 | RESOLVED (interpretation) | Session scope (per-session transitions, survival averaged over the 8 most recent sessions). Literal global a_t learns noise under ~8 interleaved sessions (R 0.18 vs 0.57). |
| B5 | DEFAULT | age in scheduler steps. |
| B6 | DEFAULT | |b| literal (constant 16 for full blocks). |
| B7 | DEFAULT | Plug-in entropies from counts. |
| B8 | DEFAULT | Cumulative counts. |
| B9 | RESOLVED | Warmup = learned anchor (block-aligned LCP across ≥ 2 sessions) + 4 fixed tokens, max_tokens=1, ≤ 1 in flight, ≥ 2 s per agent, only when a sequence slot is free; ids `cswarm-*`. |
| C1 | RESOLVED (fallback) | Synthetic traces from Fig. 5 (Selector column placement searched for R = 0.57). No real AutoGen runs. |
| C2 | RESOLVED | 60 sessions locally, 6-12 invocations, 1-3 calls per tool-agent invocation; workload scaled ×0.4 so vanilla matches the paper's vanilla hit-rate range (calibration with vanilla only). |
| C3 | DEFAULT | Poisson arrivals. |
| C4 | n/a | No dataset content used (synthetic tokens). |
| D1 | RESOLVED (weak) | Soft TTL pinning (0.3 s). Under our conditions it coincides with LRU (pinned blocks are LRU's most recent), so it is not a faithful Continuum. |
| D3 | RESOLVED | temperature 0, ignore_eos, fixed output lengths from the trace; tuning seeds 101/102, eval seeds 1/2/3. |
| D4 | RESOLVED | Explicit block budgets 100/150/200 (`num_gpu_blocks_override`). |
| E2 | RESOLVED | enforce_eager on locally. |
| E3 | n/a | No real LLM routing (synthetic). |

New open questions found during implementation:
- **Q-new-1:** the paper's history-block reuse (12-15 per block, Fig. 3a) could not be matched together with its
  vanilla hit-rate range at 100-200 blocks; our history reuse is ≈1-2. Unknown how the paper's sessions are
  structured (how many LLM calls per agent invocation, and session length).
- **Q-new-2:** how does Alg. 1 avoid corrupting transitions with concurrent sessions? (Not stated.)
- **Q-new-3:** does Eq. 9 apply survival to history blocks? (Literal reading hurts in our workload.)
