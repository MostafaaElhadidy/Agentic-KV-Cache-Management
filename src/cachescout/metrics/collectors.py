"""Build TurnRecords from vLLM outputs and read vLLM's engine-wide prefix-cache counters.

Duck-typed on vLLM objects so this module imports without vLLM (fast, GPU-free tests).
Field sources are documented in docs/vllm_internals.md.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from cachescout.metrics.records import TurnRecord

QUERIES_COUNTER = "vllm:prefix_cache_queries"
HITS_COUNTER = "vllm:prefix_cache_hits"


def record_from_request_output(
    output: Any,
    t_send: float,
    t_done: float,
    *,
    t_first_token: float | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    turn_idx: int | None = None,
    is_warmup: bool = False,
) -> TurnRecord:
    """Convert a finished offline `vllm.RequestOutput` into a TurnRecord.

    - prompt_tokens = len(output.prompt_token_ids); cached_tokens = output.num_cached_tokens
      (always populated, docs/vllm_internals.md §1).
    - If `t_first_token` is not given and vLLM stats are enabled, TTFT is taken from vLLM's
      `metrics.first_token_latency` (a duration, so no clock mixing): offline approximation.
    """
    first_token_latency = None
    stats = getattr(output, "metrics", None)
    if stats is not None:
        ftl = getattr(stats, "first_token_latency", 0.0)
        first_token_latency = float(ftl) if ftl and ftl > 0 else None
    if t_first_token is None and first_token_latency is not None:
        t_first_token = min(t_send + first_token_latency, t_done)

    completion = output.outputs[0] if output.outputs else None
    return TurnRecord(
        request_id=str(output.request_id),
        prompt_tokens=len(output.prompt_token_ids or []),
        cached_tokens=int(output.num_cached_tokens or 0),
        output_tokens=len(completion.token_ids) if completion is not None else 0,
        t_send=t_send,
        t_first_token=t_first_token,
        t_done=t_done,
        session_id=session_id,
        agent_id=agent_id,
        turn_idx=turn_idx,
        is_warmup=is_warmup,
        vllm_first_token_latency=first_token_latency,
    )


@dataclass(frozen=True)
class PrefixCounters:
    """vLLM engine-wide prefix-cache counters, in tokens (docs/vllm_internals.md §3)."""

    queries: int
    hits: int

    def __sub__(self, other: "PrefixCounters") -> "PrefixCounters":
        return PrefixCounters(self.queries - other.queries, self.hits - other.hits)


def prefix_counters(metrics: Iterable[Any]) -> PrefixCounters:
    """Sum `vllm:prefix_cache_queries/hits` over engines from `LLM.get_metrics()`.

    Requires `disable_log_stats=False`. Preempted-request recomputation is not included in
    these counters (vLLM tracks it separately).
    """
    queries = hits = 0
    for m in metrics:
        name = getattr(m, "name", None)
        if name == QUERIES_COUNTER:
            queries += int(m.value)
        elif name == HITS_COUNTER:
            hits += int(m.value)
    return PrefixCounters(queries=queries, hits=hits)
