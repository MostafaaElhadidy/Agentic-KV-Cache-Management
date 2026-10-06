"""CacheScout hook for vLLM 0.31.0 via the official `scheduler_cls` extension point (M4).

Paper Sec. 4 patches the block pool, engine core and scheduler of vLLM v0.11. Here (engineering
choice, docs/decisions.md) nothing in site-packages is edited: vLLM resolves
`scheduler_cls="cachescout.vllm_plugin.scheduler.CacheScoutScheduler"` inside the engine-core
process, and this subclass of the default `AsyncScheduler` installs instance-level wrappers on its
own KV-cache manager and block pool:

1. `kv_cache_manager.get_computed_blocks(request)`  -> ObserveTouch at prefix-matching time,
   fingerprint = first k block hashes of the request (Sec. 4 hook 1, Alg. 1 lines 10-21).
2. `block_pool.get_new_blocks(n)` -> victims chosen by the runtime policy (Eq. 9 for CacheScout)
   instead of popping the LRU head (Sec. 4 hook 2, Alg. 1 line 9). `policy: lru` uses the
   original `popleft_n`, i.e. byte-identical to vanilla.
3. `kv_cache_manager.allocate_slots` -> maps newly allocated blocks to the request's agent.
4. `block_pool.free_blocks` -> last-access time / Continuum TTL pins.
5. `schedule()` -> scheduler-step counter (age unit of Eq. 9).
Warmup (Sec. 4 hook 3) is issued by the client through the normal API; requests whose id starts
with the warmup prefix are excluded from transition learning.

Configuration: env var CACHESCOUT_CONFIG = JSON {"params": {...}, "stats_path": "..."}.
"""

import json
import os
import time
from collections.abc import Iterable
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from cachescout.core.runtime import CacheScoutParams, CacheScoutRuntime, Candidate

ENV_VAR = "CACHESCOUT_CONFIG"
STATS_EVERY_STEPS = 500


def session_of(request: Any) -> str | None:
    """Session metadata from the framework: vLLM 0.31's native `Request.session_id` (only used for
    KV events by vLLM itself, so passing it does not change vanilla behaviour), else the driver's
    request-id convention '<session>|<turn>' (+ '-<suffix>' added by vLLM)."""
    sid = getattr(request, "session_id", None)
    if sid:
        return str(sid)
    rid = request.request_id
    return rid.split("|", 1)[0] if "|" in rid else None


class CacheScoutScheduler(AsyncScheduler):
    """AsyncScheduler + CacheScout runtime hooks (enabled iff CACHESCOUT_CONFIG is set)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = json.loads(os.environ.get(ENV_VAR, "{}") or "{}")
        self._cs_stats_path: str | None = cfg.get("stats_path")
        self._cs = CacheScoutRuntime(CacheScoutParams.from_dict(cfg.get("params")))
        self._cs_block_size = int(self.cache_config.block_size)
        self._cs_current_request: Any = None
        self._cs_started = time.time()
        self._install_hooks()
        self._write_stats()

    # ----- hook installation ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        kvm = self.kv_cache_manager
        pool = kvm.block_pool
        rt = self._cs
        k = rt.p.fingerprint_blocks
        orig_get_computed = kvm.get_computed_blocks
        orig_allocate = kvm.allocate_slots
        orig_free_blocks = pool.free_blocks

        def get_computed_blocks(request: Any) -> Any:
            result = orig_get_computed(request)
            hashes = list(request.block_hashes[:k])
            fp: Any = tuple(hashes) if hashes else ("tok", tuple(request.prompt_token_ids[:16]))
            session = session_of(request)
            agent = rt.observe_dispatch(request.request_id, fp, session)
            blocks = result[0].blocks[0] if result[0].blocks else ()
            rt.on_blocks_used([b.block_id for b in blocks if not b.is_null], agent, session)
            return result

        def allocate_slots(request: Any, *a: Any, **kw: Any) -> Any:
            self._cs_current_request = request
            try:
                return orig_allocate(request, *a, **kw)
            finally:
                self._cs_current_request = None

        def free_blocks(ordered_blocks: Iterable[Any]) -> None:
            blocks = list(ordered_blocks)
            orig_free_blocks(blocks)
            rt.on_blocks_freed([b.block_id for b in blocks
                                if b.ref_cnt == 0 and not b.is_null])

        kvm.get_computed_blocks = get_computed_blocks
        kvm.allocate_slots = allocate_slots
        pool.free_blocks = free_blocks
        pool.get_new_blocks = self._make_get_new_blocks(pool)

    def _make_get_new_blocks(self, pool: Any) -> Any:
        rt = self._cs
        queue = pool.free_block_queue
        bs = self._cs_block_size

        def select(n: int) -> list[Any]:
            if rt.p.policy == "lru":
                return queue.popleft_n(n)                   # identical to vanilla
            order = []
            node = queue.fake_free_list_head.next_free_block
            while node is not None and node is not queue.fake_free_list_tail:
                order.append(node)
                node = node.next_free_block
            cands = [Candidate(b.block_id, b.block_hash is not None, bs) for b in order]
            chosen = [order[i] for i in rt.select_victims(cands, n)]
            for b in chosen:
                queue.remove(b)
            return chosen

        def get_new_blocks(num_blocks: int) -> list[Any]:
            # Mirrors BlockPool.get_new_blocks (vllm/v1/core/block_pool.py:669-702) except for
            # the victim choice.
            if num_blocks > pool.get_num_free_blocks():
                raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            ret = select(num_blocks)
            if pool._reuse_watchers:
                pool._notify_reuse(ret)
            for block in ret:
                if pool.enable_caching:
                    pool._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if pool.metrics_collector:
                    pool.metrics_collector.on_block_allocated(block)
            req = self._cs_current_request
            if req is not None:
                rt.on_blocks_used([b.block_id for b in ret], rt.agent_of_request(req.request_id),
                                  session_of(req), new=True)
            return ret

        return get_new_blocks

    # ----- scheduler overrides ----------------------------------------------------------------

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        self._cs.on_step()
        if self._cs.step % STATS_EVERY_STEPS == 0:
            self._write_stats()
        return super().schedule(*args, **kwargs)

    def shutdown(self) -> None:
        self._write_stats()
        super().shutdown()

    def _write_stats(self) -> None:
        if not self._cs_stats_path:
            return
        data = self._cs.summary()
        data["steps"] = self._cs.step
        data["wall_s"] = time.time() - self._cs_started
        tmp = self._cs_stats_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self._cs_stats_path)
