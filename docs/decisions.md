# Decisions log

Format: date, decision, why, alternatives. Label: **paper** / **interpretation** / **engineering choice**.

## 2026-10-06: Reuse the existing vLLM environment
- **Engineering choice.** Use `~/testLLM/.venv` (uv venv, Python 3.12.15, vLLM 0.31.0, torch 2.13.0+cu132).
  No second environment. Pre-change freeze saved to `docs/env_freeze_before.txt`.
- Install safeguard (user rule): before any install, save the freeze, run `uv pip install --dry-run`, and stop if
  vllm/torch/transformers/xformers/nvidia-*/cuda-* would change.

## 2026-10-06: Installed pytest 9.1.1 and ruff 0.16.10
- **Engineering choice.** The dry run showed only 4 additions (iniconfig, pluggy, pytest, ruff), no changes to existing
  packages. pyyaml 6.0.3 was already present (vLLM dependency).

## 2026-10-06: Local model = Qwen2.5-1.5B-Instruct
- **Engineering choice** (user decision). Already cached; ~3.1 GB bf16 weights. Paper uses Llama-3.1-8B-Instruct (Sec. 5.1).

## 2026-10-06: Control KV budget in blocks, not GB
- **Interpretation.** The paper's budget study is in blocks (Fig. 14: 100–200). Use `num_gpu_blocks_override` so local
  and cloud runs are comparable. `gpu_memory_utilization` only has to leave enough room for weights + activations.

## 2026-10-06: Simulator before vLLM hook
- **Engineering choice.** Validate Eqs. 3–9 / Alg. 1 in a GPU-free block-cache simulator (fast tests), then port to a
  vLLM runtime plugin. Prefer plugin / monkey-patch over editing site-packages (see open_questions A1).

## 2026-10-06: Git tracks results/log only
- **Engineering choice.** `results/` is ignored except `results/log/` (small Markdown/JSON experiment records written
  by `/log-experiment`), so experiment metadata is versioned but bulky outputs are not.

## 2026-10-06: Edit hook runs ruff on the edited file only
- **Engineering choice.** `.claude/settings.json` PostToolUse hook → `scripts/hooks/ruff_on_edit.sh`. Fast (<1 s),
  never touches the GPU. Full test suite is run manually (`pytest -q`).

## 2026-10-06: enforce_eager=True locally (applies to ALL compared systems)
- **Engineering choice.** First run of `scripts/smoke_vllm_generate.py` (local config, enforce_eager=false): the
  model loaded fine (2.98 GiB weights), then failed with CUDA OOM during torch.compile autotuning
  (`InductorError: Failed to run autotuning code block`). RAM was fine; WSL did not crash.
- Fix: `enforce_eager: true` in `configs/hardware/local.yaml`. In vLLM 0.31.0 this sets
  `compilation_config.mode = NONE` (no torch.compile/Inductor autotuning) and `cudagraph_mode = NONE`
  (no CUDA graph capture); see `vllm/config/vllm.py`.
- **Fairness:** eager mode is a hardware-profile setting, so it applies identically to vanilla vLLM, the
  Continuum-style baseline, and CacheScout (all ablations). Never compare an eager run against a compiled run.
  Eager mode slows decoding, so absolute local latency/throughput is pessimistic, but relative comparisons hold.
- `configs/hardware/cloud.yaml` keeps `enforce_eager: false` (paper uses a normal vLLM setup, Sec. 5.1).
  Every result records the flag via its saved config.
- (Correction, see next entry: the actual OOM site was the KV-cache allocation, not autotuning. enforce_eager
  stays on; it is harmless for fairness and avoids compile memory spikes.)
- Smoke script reads GPU memory via NVML (pynvml, already installed with vLLM) rather than
  `torch.cuda.mem_get_info`, which reports the same device-wide numbers but would create a CUDA context
  (~0.3–0.5 GB) in the parent process.

## 2026-10-06: Fixed KV budget locally: num_gpu_blocks_override=256, max_model_len=2048
- **Engineering choice.** Root cause of the smoke-test OOM (user-diagnosed): `allocate_kv_cache`
  (`vllm/v1/worker/utils.py`) tried to allocate 1.23 GiB with 3.86 GiB reported free. The KV-cache size vLLM derives
  from `gpu_memory_utilization` is too large to allocate reliably on this WSL setup; it is not the profiling run.
- Fix in `configs/hardware/local.yaml`: `num_gpu_blocks_override: 256` (≈ 256 × 448 KiB = 112 MiB KV for
  Qwen2.5-1.5B: 28 layers × 2 KV heads × 128 dim × K,V × 2 B × 16 tokens) and `max_model_len: 2048`.
- Verified in installed vLLM 0.31.0 source (read-only):
  - `v1/core/kv_cache_utils.py::may_override_num_blocks` replaces the profiled block count with the override;
    `get_kv_cache_config_from_groups` then sets the allocation `size = bytes_per_block * num_blocks`, and
    `v1/worker/utils.py::allocate_kv_cache` allocates exactly that buffer. So the override shrinks the real allocation.
  - `get_kv_cache_configs` logs "Overriding num_gpu_blocks=X with num_gpu_blocks_override=Y" and plans the
    max_model_len admission check against `override × bytes_per_block` **minus one block** (the null block that
    `BlockPool` permanently reserves, `v1/core/block_pool.py`). Usable blocks = override − 1.
  - The final count is sent back to the frontend (`v1/engine/core_client.py`), and the smoke script prints it
    from `llm.llm_engine.vllm_config.cache_config.num_gpu_blocks`.
- Capacity: 255 usable blocks; one 2048-token sequence needs 128 blocks, so ≈ 2 full-length sequences fit at once.
- **Implication for the paper's cache study (Fig. 14, 100–200 blocks):** max_model_len must satisfy
  `ceil(max_model_len / 16) ≤ budget − 1`. For 100 blocks: ≤ 99 × 16 = **1584 tokens** (not 1600, because of the
  null block); for 150: ≤ 2384; for 200: ≤ 3184. The paper notes a ~93-block max request footprint (Sec. 5.4), which
  fits within 99. Experiment configs for the budget sweep must set max_model_len accordingly.
- `gpu_memory_utilization` (0.60) still bounds vLLM's startup memory check and profiling, but no longer sizes the KV cache.

## 2026-10-06: Local settings validated (smoke_run3.log)
- **Engineering choice, verified.** With `enforce_eager: true`, `num_gpu_blocks_override: 256`, `max_model_len: 2048`
  (plus `gpu_memory_utilization: 0.60`, `max_num_seqs: 8`), the smoke test passed:
  - GPU used before load 424 MiB (Windows desktop); weights 2.98 GiB; after load 3771 MiB used, 4417 MiB free.
  - vLLM log: "Available KV cache memory: 1.23 GiB", "Overriding num_gpu_blocks=2867 with num_gpu_blocks_override=256",
    "GPU KV cache size: 4,096 tokens, Maximum concurrency for 2,048 tokens per request: 2.00x".
  - Frontend reports `num_gpu_blocks=256`; generation worked.
- **What each setting fixed:** `num_gpu_blocks_override` fixed the actual OOM (the 1.23 GiB KV allocation);
  `max_model_len 2048` keeps one full sequence (128 blocks) within the budget; `enforce_eager` avoids compile/autotune
  and CUDA-graph memory spikes and stays on for all compared systems.
- **Finding: one block is reserved.** vLLM's `BlockPool` permanently holds one null block, so 256 allocated = **255
  usable**. vLLM's own log line counts all 256 (4,096 tokens, 2.00x); the real usable capacity is 4,080 tokens. All
  budget-sweep configs and hit-rate analysis must use usable = budget − 1.
- Note: the `LLM` class runs with `disable_log_stats: True` by default (seen in the log's non-default args), which
  matters for the metrics design (see M1 metrics).
- Note: vLLM printed "We must use the `spawn` multiprocessing start method". Harmless (the script has a `__main__`
  guard); not investigated further.

## 2026-10-06: Metrics layer design (M1)
- **Engineering choice.** `src/cachescout/metrics/`: `TurnRecord` (one per agent turn) → `summarize()`; collectors
  turn vLLM outputs into records. Same summary code for the offline engine (now) and the OpenAI server + streaming
  client (next step). Sources of every vLLM field: `docs/vllm_internals.md`.
- **Paper (Sec. 5.1):** hit rate = Σ cached prompt tokens / Σ prompt tokens; TTFT = arrival → first token;
  per-turn latency = end-to-end per agent invocation; throughput = completed agent turns per second.
- **Interpretation, warmup requests:** CacheScout warmups (Sec. 3.4) are **excluded from hit rate, TTFT and per-turn
  latency**, but **included in throughput** in the sense that the throughput window (first send → last completion)
  spans them, while only non-warmup turns are counted as completed turns. The paper only states that warmups are
  excluded from transition learning (Sec. 4); it does not say how they enter the metrics. (User-approved.)
- **Engineering choice, percentiles:** `numpy.percentile(..., method="linear")` (numpy's default; interpolates
  between closest ranks) for median/P90/P99. Constant `PERCENTILE_METHOD` in `src/cachescout/metrics/summary.py`.
  The paper does not state its percentile method. (User-approved.)
- **Engineering choice, clocks:** all latencies come from our own `time.perf_counter()`; never subtract vLLM's
  monotonic engine-core timestamps from its wall-clock `arrival_time`. For offline runs, TTFT uses vLLM's
  `first_token_latency` duration (requires `disable_log_stats=False`); with sequential requests this ≈ prefill time.
- Also reported: `max_possible_hit_rate` using vLLM's ceiling floor((N−1)/16)·16 per prompt, because even a perfect
  cache can't reach 100% in vLLM.
- **M1 check script** (`scripts/check_prefix_metrics.py`): requests strictly sequential, `max_tokens=8` with
  `ignore_eos=True` (fixed output length so block arithmetic is exact), synthetic token-ID prompts (seeded),
  `reset_prefix_cache()` between checks; counter comparison uses before/after snapshots.

## 2026-10-06: CacheScout constants (tuned on tuning traces only; simulator)
- **Engineering choice / interpretation.** Paper gives no values (open_questions B1). Tuned in the simulator on
  `selector_tune` + `debate_tune` (seeds 101/102), objective = mean hit rate over 100/150/200 blocks. Evaluation
  traces (seeds 1-3) were never used for tuning. Files: results/tuning/eviction_sweep.json (512 configs),
  results/tuning/eviction_sweep_v1_all_mapping.json, results/tuning/warmup_sweep.json.
- Final: epsilon 0.01, tau 0.1, E_max 6, lambda 0.005 per scheduler step, delta 0.01, fingerprint 2 blocks,
  scope session (8 most recent sessions, per-session Eq. 8 tables averaged), block_mapping anchor_only,
  R_min 0.3, warmup min interval 2.0 s, at most one warmup in flight.
- tau=0.0 scored 0.7108 vs 0.7096 (noise level) but disables the transition graph; tau 0.1 kept for fidelity.
- R_min: hit-rate objective is flat; 0.7 (argmax) disables warmup entirely, so 0.3 was chosen to keep the
  paper's gate semantics (Sec. 3.4: warm for structured execution, not for random).

## 2026-10-06: Session scope and anchor-only block mapping (interpretations of B4 and B3)
- **Interpretation, B4.** Alg. 1's single global current agent learns noise when ~8 sessions interleave
  (learned R 0.18, top-1 0.43 vs true R 0.57). Session scope: transitions counted per session (session id is
  framework metadata, Sec. 3.1 "exchanges lightweight runtime metadata"; vLLM 0.31 `Request.session_id`).
  Survival = mean over the 8 most recent sessions of Eq. 8 (from each session's current agent).
- **Interpretation, B3.** Sec. 3.3 says each block "inherits the survival score of its corresponding agent
  anchor". Literal Alg. 1 line 12 maps all touched blocks to the agent; that lets stale history of hot agents
  displace anchors. `anchor_only`: a block counts as an anchor block once requests from >= 2 different sessions
  used it (no labels needed); other blocks get p_surv = 0 (recency only). The literal variant remains available
  (`scope: global`, `block_mapping: all`) and is reported as "CacheScout-literal".
- Evidence (simulator, tuning trace, Belady diagnostic): see WORK_LOG 2026-10-06.

## 2026-10-06: Real multi-agent workload (branch real-agents)
- **Interpretation, selector routing.** AutoGen's SelectorGroupChat picks the next speaker with a *separate*
  selector LLM call. Here every agent ends its own message with `NEXT: <AGENT>`, which the driver parses
  (src/cachescout/agents/routing.py). If there is no strict `NEXT:` line, a final line that is only an agent
  name (`DECIDER`, `[DECIDER]`, `[PLANNER]: DECIDER`) is accepted and counted as a *lenient* route. Missing,
  unknown or self-referencing names trigger a **round-robin fallback** (next agent in team order
  P,A,C,T,R,D, like AutoGen's RoundRobinGroupChat), counted and reported. (First version used "fallback to
  PLANNER / CODER"; on train problems it looped PLANNER↔CODER until the call cap, so it was replaced before any
  evaluation run. Smoke evidence: results/real/gsm8k_train_selector_s101/vanilla/b100_smoke.) Reason: one fewer LLM call
  per turn, and no extra "selector" anchor that would change the cache workload. Consequence: routing quality
  depends on the 1.5B model following the format; the fallback rate measures this.
- **Engineering choice, prompt structure.** System = agent anchor (always explicit, so Qwen's default system
  prompt never appears) + `Task: <question>` + shared group-chat history as user messages `[NAME]: text` and
  `[tool:name] result`. Chat template applied by the tokenizer; token IDs sent to vLLM (exact prefixes).
- **Engineering choice, trimming.** If prompt + max_tokens > max_model_len (1584), the oldest history messages
  after anchor + task are dropped. The trim rate is reported per run.
- **Engineering choice, termination.** Session ends on `FINAL ANSWER:` from the DECIDER, at the end of the
  pipeline chain, or at 14 calls (the last call is a forced DECIDER answer).
- **Engineering choice, tasks.** GSM8K test split for evaluation, train split for tuning; one fixed permutation
  per split, seed k takes the k-th disjoint slice; arrivals seeded per problem seed.
- **Tried and rejected (user-defined criterion): AutoGen-style separate selector call.** A SELECTOR router call
  after each agent turn, its reply constrained with vLLM 0.31 guided choice
  (`SamplingParams(structured_outputs=StructuredOutputsParams(choice=[5 teammate names]))`). Router pilot (train
  problems seed 102, 5 sessions, results/real/gsm8k_train_selector_s102/vanilla/b100_pilot_router/result.json):
  model-made routing 17/17 = 100% (criterion > 90% met), but choices degenerate: DECIDER 64.7% (> 60% limit),
  PLANNER 23.5%, ANALYST 11.8%, CODER/TESTER/REVIEWER 0%; every session alternated PLANNER<->DECIDER;
  measured R 0.74; ~370 extra tokens per agent turn. Per the agreed rule it is NOT adopted. The code path stays
  available (`agents.router: true`, default false). The selector topology keeps the agent's own `NEXT:` line,
  with no further prompt tuning, and its fallback rate is reported prominently.
- **Simulator validity on real sessions.** Sequential replay of the selector tuning recording on vLLM vs the
  simulator: 155/155 requests exact for vanilla, lru_hook AND eviction_only
  (results/real/replay_gsm8k_train_selector_s101/*/b100_seq/crosscheck.json). Constants were therefore re-tuned
  in the simulator.
- **Re-tuned constant for the real workload (tuning problems only).** Grid of 512 configs on the four vanilla
  recordings of GSM8K TRAIN problems (seed 101), objective = mean hit rate over 100/150/200 blocks
  (results/tuning/real_eviction_sweep.json): vanilla 0.3265, previous constants 0.3783, best 0.3803 (only
  lambda 0.005 -> 0.001). Gain 0.20 pp = exactly the pre-set adoption threshold (0.2 pp), so lambda = 0.001 is
  adopted for configs/experiments/real/local.yaml; a borderline change. Synthetic-trace configs unchanged.
- **Replay is open-loop (engineering choice).** `mode: online` with a recorded live run as `trace` sends the
  recorded prompt token IDs with `max_tokens` = the recorded output length and `ignore_eos`. The model's
  replay-time generations are discarded and do NOT feed later prompts; arrival times and inter-call gaps are
  also the recorded ones. This keeps the request stream identical across systems, so only the cache policy
  differs, but it does not react to the system's own speed or outputs (unlike the live closed-loop runs).
- **Debate topology is deterministic (interpretation).** Our debate routes P→C→R→D and, on `REVISE` (or an
  unclear DECIDER reply), back to C. Measured R = 1.00. The paper's debate (Fig. 4a/5) is stochastic: after the
  reviewer the next speaker varies (R = 0.78). Our results for "debate" therefore correspond to a fully
  predictable coder/reviewer/judge loop, not to the paper's debate statistics.

## 2026-10-07: Remote-box preparation (branch box-prep)
- **Engineering choice.** GPU-size profiles `configs/hardware/box_{24,48,80}gb.yaml` for Llama-3.1-8B-Instruct
  (bf16): 24 GB uses `enforce_eager: true`, `max_num_seqs: 32`, `max_num_batched_tokens: 4096`; 48 GB uses
  eager off, 128 seqs; 80 GB = the original `cloud.yaml` settings (eager off, 256 seqs). All use
  `gpu_memory_utilization: 0.90`, `num_gpu_blocks_override: null` (experiments set 100/150/200 blocks) and
  `HF_HUB_OFFLINE=1` (download first, never mid-run). Memory arithmetic in docs/NEW_BOX_SETUP.md §9
  (KV 128 KiB/token, 2 MiB/block for Llama-3.1-8B; computed by `cachescout.kv_blocks`, tested). KV-memory
  estimates are estimates; vLLM's "Available KV cache memory" log line is authoritative.
- **Engineering choice.** `--hardware <profile>` on run.py / compare.py / check_prefix_metrics.py and
  `HARDWARE=` on run_real_campaign.sh select the profile without editing experiment configs (defaults unchanged).
- **Bug fix (latent in the original cloud prep).** `m1_metrics_check/cloud.yaml` combined 256 blocks
  (255 usable) with the profile's `max_model_len: 4096` (needs 256 blocks), so vLLM would refuse to start.
  It now sets `max_model_len: 2048`, and check_prefix_metrics.py honours an experiment-level `max_model_len`
  like run.py.
- **Fingerprints under Llama 3.1.** Its chat template prepends a shared header ("Cutting Knowledge Date / Today
  Date") before each system prompt; `scripts/check_fingerprints.py` / the tokenizer test verify that the six
  agents still differ within the plugin's 2-block window. Not yet run locally (tokenizer not cached; the unsloth mirror is ungated, see 2026-10-08);
  if it fails on the box, raise `fingerprint_blocks` in configs/experiments/real/cloud.yaml.
- **Ungated fallback model:** Qwen/Qwen2.5-7B-Instruct (Apache-2.0; 56 KiB KV/token). Results with it must be
  labelled as Qwen2.5-7B, not as the paper's model.

## 2026-10-08: Model mirror for Llama-3.1-8B-Instruct (branch box-model-mirror)
- **Engineering choice.** The remote-box profiles (`configs/hardware/{cloud,box_24gb,box_48gb,box_80gb}.yaml`)
  load `unsloth/Llama-3.1-8B-Instruct`, an ungated Hugging Face re-upload of the paper's model (Sec. 5.1),
  because access to the gated `meta-llama/Llama-3.1-8B-Instruct` was requested and is still pending.
- **What was checked (2026-10-08, Hugging Face API):** the four `model-0000x-of-00004.safetensors` files of
  `unsloth/Llama-3.1-8B-Instruct` have the same SHA-256 (LFS oid) as those of
  `NousResearch/Meta-Llama-3.1-8B-Instruct`, a second independent mirror. They could **not** be compared with
  Meta's own files: the gated repo hides its checksums from accounts without access. Treat "same weights as
  Meta's" as very likely, not verified.
- **Why unsloth and not NousResearch:** unsloth's `tokenizer_config.json` carries Meta's official Llama 3.1 chat
  template (system header with "Cutting Knowledge Date / Today Date"); NousResearch's carries an older, simpler
  template. CacheScout works on prompt prefixes, so the exact prompt tokens matter. Its `config.json` differs
  only in metadata (`eos_token_id` 128009 alone, `pad_token_id`, `unsloth_fixed`); `generation_config.json`
  differs only in `max_length`, `pad_token_id` and the transformers version string. `check_fingerprints.py`
  still checks the agents' prefixes on the box before any run.
- **One-line switch:** the name lives only in `configs/models/llama31_8b.yaml` (hardware profiles use
  `model_file:`; `load_yaml` resolves it). When Meta access is approved, change that line to
  `meta-llama/Llama-3.1-8B-Instruct` and download it; the preflight default follows the same file.
- **Claims:** results from the box are "Llama-3.1-8B-Instruct (unsloth mirror)", i.e. the same model as the paper,
  not the paper's numbers: hardware, vLLM version, workload driver and load differ (see README Limitations).
