"""Pure-Python KV-cache simulator mirroring vLLM 0.31.0 prefix caching (M3).

Mirrors (docs/vllm_internals.md): chained block hashes over full 16-token blocks; hits limited to
`num_tokens - 1` and whole blocks (§2); one reserved null block (§6); free queue where freed
blocks with a hash are appended to the tail (LRU) and blocks without a hash are prepended (§6);
allocation evicts the hash of each reused block. Decode-region blocks get unique hashes (the
generated tokens differ from the trace tokens in real runs, so they never hit).

Execution model (engineering choice): event-driven, up to `max_concurrency` requests run at once
(vLLM max_num_seqs); each holds its blocks from admission to completion; requests that do not fit
wait FIFO. Background warmups only run when no foreground request is waiting (paper Sec. 3.4:
"during idle periods"). Service time is a simple linear model, so simulated latencies are
indicative only; the simulator's primary output is the hit rate.
"""

import heapq
import itertools
import math
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from cachescout.core.runtime import CacheScoutParams, CacheScoutRuntime, Candidate
from cachescout.core.warmup import WarmupCoordinator
from cachescout.metrics import TurnRecord, summarize
from cachescout.workload.trace import Trace


def chained_hashes(tokens: list[int], block_size: int) -> list[int]:
    """Hash of each full block, chained on the parent hash (like vLLM block hashing)."""
    out, parent = [], 0
    for i in range(len(tokens) // block_size):
        parent = hash((parent, tuple(tokens[i * block_size:(i + 1) * block_size])))
        out.append(parent)
    return out


class SimBlockPool:
    """vLLM-like block pool with pluggable victim selection via CacheScoutRuntime."""

    def __init__(self, num_gpu_blocks: int, block_size: int, runtime: CacheScoutRuntime | None):
        if num_gpu_blocks < 2:
            raise ValueError("need at least 2 blocks (one is the null block)")
        self.bs = block_size
        self.rt = runtime
        self.free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(1, num_gpu_blocks))
        self.ref = [0] * num_gpu_blocks
        self.hash_of: dict[int, Hashable] = {}
        self.cached: dict[Hashable, int] = {}
        self._unique = itertools.count()
        self.evictions = 0

    @property
    def usable(self) -> int:
        return len(self.ref) - 1

    def _allocate(self, n: int, now: float) -> list[int]:
        if n > len(self.free):
            raise RuntimeError(f"cannot allocate {n} blocks, {len(self.free)} free")
        if self.rt is None:
            chosen = list(itertools.islice(self.free, n))
        else:
            order = list(self.free)
            cands = [Candidate(b, b in self.hash_of, self.bs) for b in order]
            chosen = [order[i] for i in self.rt.select_victims(cands, n, now)]
        for b in chosen:
            del self.free[b]
            h = self.hash_of.pop(b, None)
            if h is not None:
                self.evictions += 1
                if self.cached.get(h) == b:
                    del self.cached[h]
            self.ref[b] = 1
        return chosen

    def start(self, request_id: str, tokens: list[int], output_tokens: int, session: str,
              fingerprint_blocks: int, now: float) -> tuple[int, list[int]] | None:
        """Admit a request: prefix match (ObserveTouch), touch hits, allocate all its blocks and
        cache its prompt blocks. Returns (cached_tokens, blocks), or None if blocks are short
        (the request then waits, like vLLM's scheduler)."""
        hashes = chained_hashes(tokens, self.bs)
        max_hit_blocks = (len(tokens) - 1) // self.bs
        hit: list[int] = []
        for h in hashes[:max_hit_blocks]:
            b = self.cached.get(h)
            if b is None:
                break
            hit.append(b)
        kv_tokens = len(tokens) + output_tokens - 1       # last sampled token is never written
        total_blocks = math.ceil(kv_tokens / self.bs)
        reclaim = sum(1 for b in hit if self.ref[b] == 0)
        if total_blocks - len(hit) > len(self.free) - reclaim:
            return None
        agent = None
        if self.rt is not None:
            fp = tuple(hashes[:fingerprint_blocks]) or (tuple(tokens[:1]),)
            agent = self.rt.observe_dispatch(request_id, fp, session)
        for b in hit:                                     # touch
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1
        if self.rt is not None:
            self.rt.on_blocks_used(hit, agent)
        new = self._allocate(total_blocks - len(hit), now)
        if self.rt is not None:
            self.rt.on_blocks_used(new, agent)
        blocks = hit + new
        full_prompt_blocks = len(tokens) // self.bs
        for i in range(len(hit), kv_tokens // self.bs):   # cache newly computed full blocks
            b = blocks[i]
            h = hashes[i] if i < full_prompt_blocks else ("gen", next(self._unique))
            self.hash_of[b] = h
            self.cached.setdefault(h, b)
        return len(hit) * self.bs, blocks

    def finish(self, request_id: str, blocks: list[int], now: float) -> None:
        """Free a finished request's blocks in reverse order (tail evicted first)."""
        freed = []
        for b in reversed(blocks):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                freed.append(b)
                self.free[b] = None
                if b not in self.hash_of:
                    self.free.move_to_end(b, last=False)  # prepend unhashed
        if self.rt is not None:
            self.rt.on_blocks_freed(freed, now)
            self.rt.forget_request(request_id)

    def run(self, request_id: str, tokens: list[int], output_tokens: int, session: str,
            fingerprint_blocks: int, now: float) -> int:
        """Serve one request start-to-finish (sequential); return cached prompt tokens."""
        res = self.start(request_id, tokens, output_tokens, session, fingerprint_blocks, now)
        if res is None:
            raise RuntimeError(f"cannot allocate blocks for {request_id}")
        cached, blocks = res
        if self.rt is not None:
            for _ in range(output_tokens):                # prefill + decode steps (Eq. 9 age)
                self.rt.on_step()
        self.finish(request_id, blocks, now)
        return cached


@dataclass
class SimConfig:
    num_gpu_blocks: int = 200
    block_size: int = 16
    system: str = "vanilla"   # vanilla | cachescout | eviction_only | warmup_only | continuum
    params: dict[str, Any] = field(default_factory=dict)
    warmup_min_interval_s: float = 1.0
    minimal_prompt: tuple[int, ...] = (1001, 1002, 1003, 1004)
    max_concurrency: int = 8          # like vLLM max_num_seqs (local config)
    max_pending_warmups: int = 1      # rate limit: at most one queued warmup
    step_time_s: float = 0.02         # one scheduler step (decode iteration), for Eq. 9 age
    # linear service-time model (seconds): a + b * uncached_prompt_tokens + c * output_tokens
    svc_a: float = 0.02
    svc_b: float = 0.0001
    svc_c: float = 0.02


SYSTEMS = ("vanilla", "cachescout", "eviction_only", "warmup_only", "continuum", "lru_hook")


def system_flags(system: str) -> tuple[str | None, bool]:
    """(engine policy or None for stock vLLM, warmup enabled) for each comparable system."""
    table = {
        "vanilla": (None, False),
        "lru_hook": ("lru", False),          # hook installed but neutral (validation)
        "eviction_only": ("cachescout", False),
        "warmup_only": ("lru", True),
        "cachescout": ("cachescout", True),
        "continuum": ("continuum", False),
    }
    if system not in table:
        raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
    return table[system]


def simulate(trace: Trace, cfg: SimConfig) -> dict[str, Any]:
    """Event-driven replay with up to `cfg.max_concurrency` running requests (like vLLM's
    max_num_seqs). Blocks are held from admission to completion; requests that do not fit wait
    in FIFO order. Warmups run only when no foreground request is waiting (Sec. 3.4)."""
    policy, warm = system_flags(cfg.system)
    params = CacheScoutParams.from_dict({**cfg.params, "policy": policy or "lru"})
    rt = CacheScoutRuntime(params) if policy is not None else None
    pool = SimBlockPool(cfg.num_gpu_blocks, cfg.block_size, rt)
    coord = (WarmupCoordinator(params, cfg.block_size, cfg.minimal_prompt,
                               cfg.warmup_min_interval_s) if warm else None)
    sessions = {s.session_id: s for s in trace.sessions}
    seq = itertools.count()
    events: list[tuple[float, int, str, Any]] = []         # (time, seq, kind, payload)
    for s in trace.sessions:
        if s.turns:
            heapq.heappush(events, (s.arrival_s, next(seq), "ready", (s.session_id, 0)))
    waiting: list[tuple[float, str, int]] = []              # FIFO of (ready_time, sid, turn)
    warm_queue: list[tuple[float, list[int], str]] = []
    running = 0
    records: list[TurnRecord] = []
    order: list[dict[str, Any]] = []
    wid = itertools.count()
    now = 0.0

    def svc_times(prompt_len: int, cached: int, out: int) -> tuple[float, float]:
        prefill = cfg.svc_a + cfg.svc_b * (prompt_len - cached)
        return prefill, prefill + cfg.svc_c * out

    def try_start() -> None:
        nonlocal running
        while running < cfg.max_concurrency and waiting:
            t_ready, sid, ti = waiting[0]
            s = sessions[sid]
            tokens = trace.prompt(s, ti)
            out = s.turns[ti].output_tokens
            rid = f"{sid}-t{ti}"
            res = pool.start(rid, tokens, out, sid, params.fingerprint_blocks, now)
            if res is None:
                return                                      # head-of-line waits for blocks
            waiting.pop(0)
            cached, blocks = res
            prefill, total = svc_times(len(tokens), cached, out)
            running += 1
            order.append({"request_id": rid, "session_id": sid, "turn_idx": ti})
            heapq.heappush(events, (now + total, next(seq), "done",
                                    (rid, sid, ti, blocks, t_ready, now + prefill, cached)))
            if coord is not None:
                coord.observe(tokens, sid)
        while running < cfg.max_concurrency and not waiting and warm_queue:
            t_issue, tokens, sid = warm_queue.pop(0)
            rid = f"{params.warmup_prefix}{next(wid)}"
            res = pool.start(rid, tokens, 1, sid, params.fingerprint_blocks, now)
            if res is None:
                warm_queue.clear()                          # no room: drop warmups
                return
            cached, blocks = res
            _, total = svc_times(len(tokens), cached, 1)
            running += 1
            order.append({"request_id": rid, "session_id": sid, "warmup_tokens": len(tokens)})
            heapq.heappush(events, (now + total, next(seq), "wdone",
                                    (rid, blocks, t_issue, len(tokens), cached)))

    while events:
        now, _, kind, payload = heapq.heappop(events)
        if rt is not None:
            rt.step = int(now / cfg.step_time_s)            # scheduler steps elapsed (Eq. 9 age)
        if kind == "ready":
            sid, ti = payload
            waiting.append((now, sid, ti))
        elif kind == "done":
            rid, sid, ti, blocks, t_ready, t_first, cached = payload
            s = sessions[sid]
            pool.finish(rid, blocks, now)
            running -= 1
            records.append(TurnRecord(rid, len(trace.prompt(s, ti)), cached,
                                      s.turns[ti].output_tokens, t_ready, t_first, now,
                                      session_id=sid, agent_id=s.turns[ti].agent, turn_idx=ti))
            if ti + 1 < len(s.turns):
                heapq.heappush(events, (now + s.turns[ti].think_s, next(seq), "ready",
                                        (sid, ti + 1)))
            if coord is not None and len(warm_queue) < cfg.max_pending_warmups:
                d = coord.maybe_warmup(sid, now)
                if d is not None:
                    warm_queue.append((now, d.tokens, sid))
        elif kind == "wdone":
            rid, blocks, t_issue, n_tok, cached = payload
            pool.finish(rid, blocks, now)
            running -= 1
            records.append(TurnRecord(rid, n_tok, cached, 1, t_issue, None, now,
                                      is_warmup=True))
        try_start()
    if waiting:
        raise RuntimeError(f"{len(waiting)} requests could never be admitted (cache too small)")
    summary = summarize(records, block_size=cfg.block_size)
    return {
        "summary": summary.to_dict(),
        "records": [r.to_dict() for r in records],
        "order": order,
        "evictions": pool.evictions,
        "runtime": rt.summary() if rt is not None else None,
        "warmups_issued": coord.issued if coord is not None else 0,
        "warmups_gated": coord.gated if coord is not None else 0,
    }
