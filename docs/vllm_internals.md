# vLLM 0.31.0 internals (verified, read-only)

All paths are relative to `~/testLLM/.venv/lib/python3.12/site-packages/vllm/`. Line numbers are for **vLLM 0.31.0**
as installed on 2026-10-06; re-check them if vLLM is ever changed. Paper targets v0.11 (Sec. 4), so these may differ
from the authors' code.

## 1. Per-request prefix-cache hits

| Claim | Evidence |
|---|---|
| `RequestOutput.num_cached_tokens` = "the number of tokens with prefix cache hit" | `outputs.py:132` (doc), `:152` (arg), `:177` (assigned) |
| Every request gets a `PrefillStats` object, **not gated by `log_stats`**, so `num_cached_tokens` is always populated | `v1/request.py:218` (`self.prefill_stats = PrefillStats()`) |
| The scheduler records hits on the request's **first** scheduling only (`num_preemptions <= 0`): `num_cached_tokens = num_local_cached_tokens + num_external_cached_tokens` | `v1/core/sched/scheduler.py:1034-1040` → `v1/metrics/stats.py:282-295` (`PrefillStats.set`) |
| `PrefillStats.finalize` only computes `num_cache_creation_tokens`; it does not change `num_cached_tokens` | `v1/metrics/stats.py:297-300`; called at `v1/core/sched/scheduler.py:2149-2152` |
| Frontend copies it into the request state on the first (prefill) output; remote-prefill (P/D disaggregation) overrides it | `v1/engine/output_processor.py:697-706` |
| It is then passed into `RequestOutput(num_cached_tokens=...)` | `v1/engine/output_processor.py:385, 411` |

## 2. Maximum possible hit (`max_cache_hit_length`)

| Claim | Evidence |
|---|---|
| At most `num_tokens − 1` prompt tokens can hit, because the last token must be recomputed to get logits | `v1/core/kv_cache_manager.py:289-295` |
| Hits are whole blocks only ("computed blocks must be full"); together with the rule above, a repeated N-token prompt hits at most `floor((N−1)/16)·16` tokens. Example: N = 96 → 80 (the whole last block is recomputed), N = 97 → 96 | `v1/core/kv_cache_manager.py:290-294` (comment: "can trigger recomputation of an entire block … allocate_slots() requires num_computed_tokens to be block-size aligned") |

## 3. Engine-wide counters (Prometheus)

| Claim | Evidence |
|---|---|
| Counters `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` (token units) | `v1/metrics/loggers.py:600, 611`; incremented from `scheduler_stats.prefix_cache_stats.queries/hits` at `:1052-1056` |
| Recorded at admission, only if a cache lookup happened: `queries += request.num_tokens`, `hits += num_new_local_computed_tokens` | `v1/core/sched/scheduler.py:1252-1255` → `v1/core/kv_cache_manager.py:253-262` |
| **Preempted** requests go to separate `preempted_*` fields, which are **not** in those two counters | `v1/metrics/stats.py:131-144` (`PrefixCacheStats.record`) |
| Prefix-cache stats exist only when `log_stats` is on | `v1/core/kv_cache_manager.py:165` |
| `LLM` defaults `disable_log_stats=True`; pass `False` to enable | `entrypoints/llm.py:228-229` |
| `LLM.get_metrics()` → `LLMEngine.get_metrics()` asserts `log_stats`, returns an in-memory snapshot (`Counter` has `name`, `labels`, `value`) | `entrypoints/llm.py:857`, `v1/engine/llm_engine.py:398-400`, `v1/metrics/reader.py:12-30, 70` |
| Offline engine records stats synchronously each `step()` | `v1/engine/llm_engine.py:334-336` |
| `LLM.reset_prefix_cache()` exists | `entrypoints/llm.py:802-807` |

## 4. Clocks

| Claim | Evidence |
|---|---|
| `RequestStateStats.arrival_time` is a frontend **wall-clock** timestamp | `v1/metrics/stats.py:226-227` |
| `queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts` are engine-core **monotonic** timestamps, so don't subtract them from `arrival_time` | `v1/metrics/stats.py:229-233` |
| `first_token_latency` is computed in the frontend as time since `arrival_time` (a consistent duration) | `v1/metrics/stats.py:453, 473-475` |
| `RequestOutput.metrics` is only set when `log_stats` is on | `v1/engine/output_processor.py:187, 413` |
| Our choice: measure latencies with our own `time.perf_counter()`; use vLLM's `first_token_latency` only as a cross-check or for offline TTFT | `docs/decisions.md` |

## 5. KV-cache sizing

| Claim | Evidence |
|---|---|
| `num_gpu_blocks_override` replaces the profiled block count | `v1/core/kv_cache_utils.py:1125-1131` |
| Override logged; admission check planned against `override × bytes_per_block` minus one (null) block | `v1/core/kv_cache_utils.py:2764-2786` |
| Allocation size `bytes_per_block × num_blocks` → one buffer | `v1/core/kv_cache_utils.py:1772-1773`; `v1/worker/utils.py:395, 427` |
| Final block count sent to the frontend | `v1/engine/core_client.py:844-847` |
| `enforce_eager` → `compilation_config.mode = NONE`, `cudagraph_mode = NONE` | `config/vllm.py:1694-1696` |

## 6. Block pool and eviction (for the M4 hook)

| Claim | Evidence |
|---|---|
| One block is permanently reserved as the null block, so usable = num_gpu_blocks − 1 | `v1/core/block_pool.py:187` |
| `get_new_blocks(n)` pops `n` blocks from the **head** of the free queue and evicts their cached hash (`_maybe_evict_cached_block`) | `v1/core/block_pool.py:669-702, 732-753` |
| `touch(blocks)` (a prefix hit) removes a free block from the free queue and increments `ref_cnt` | `v1/core/block_pool.py:755-770` |
| `free_blocks(ordered)`: hashed (cacheable) blocks are **appended** to the tail (FIFO → LRU); unhashed blocks are **prepended** (reused first). The input order is the eviction priority | `v1/core/block_pool.py:777-808` |
| Free queue: doubly linked list with `popleft`, `popleft_n`, `remove`, `append`, `append_n` (and `prepend_n`) | `v1/core/kv_cache_utils.py:246, 299, 337, 371, 392, 437` |
| Optional sampled eviction-lifetime metrics: `ObservabilityConfig.kv_cache_metrics` (default off), `kv_cache_metrics_sample` (default 0.01) | `config/observability.py:60, 65`; `v1/core/kv_cache_metrics.py` |

**M4 implication:** vLLM's eviction is "pop from the head of an LRU free list". The paper's Eq. 9 `argmin` (Alg. 1
line 9) must either reorder this queue (e.g. re-insert by score once per scheduler step) or replace `popleft_n`. See
`docs/open_questions.md` A1/A2.
