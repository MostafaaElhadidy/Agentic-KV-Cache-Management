"""Aggregate TurnRecords into the paper's serving metrics (Sec. 5.1)."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cachescout.metrics.records import TurnRecord

# Engineering choice (docs/decisions.md): numpy's default "linear" interpolation.
PERCENTILE_METHOD = "linear"


@dataclass(frozen=True)
class LatencyStats:
    """Distribution summary in seconds. The paper reports mean, median, P90 and P99 (Figs. 8-11)."""

    n: int
    mean: float
    median: float
    p90: float
    p99: float


@dataclass(frozen=True)
class MetricsSummary:
    """Paper Sec. 5.1 metrics over non-warmup turns (throughput window includes warmups)."""

    num_turns: int
    num_warmup: int
    total_prompt_tokens: int
    total_cached_tokens: int
    hit_rate: float | None
    max_possible_hit_rate: float | None
    ttft: LatencyStats | None
    per_turn_latency: LatencyStats | None
    throughput_turns_per_s: float | None
    per_agent: dict[str, "MetricsSummary"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON results."""
        return asdict(self)


def latency_stats(values: Sequence[float]) -> LatencyStats | None:
    """Mean/median/P90/P99 with numpy linear percentiles; None for an empty input."""
    if len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99], method=PERCENTILE_METHOD)
    return LatencyStats(
        n=len(arr), mean=float(arr.mean()), median=float(p50), p90=float(p90), p99=float(p99)
    )


def max_cacheable_tokens(prompt_tokens: int, block_size: int) -> int:
    """Upper bound on prefix-cache hits for one prompt in vLLM.

    vLLM caps hits at `prompt_tokens - 1` and only counts full blocks
    (docs/vllm_internals.md §2), i.e. floor((N - 1) / block_size) * block_size.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if prompt_tokens <= 0:
        return 0
    return ((prompt_tokens - 1) // block_size) * block_size


def summarize(
    records: Iterable[TurnRecord], block_size: int = 16, by_agent: bool = True
) -> MetricsSummary:
    """Compute paper Sec. 5.1 metrics.

    - Hit rate = sum(cached_tokens) / sum(prompt_tokens) (paper Sec. 5.1), non-warmup turns only.
    - max_possible_hit_rate: same ratio with `max_cacheable_tokens` (vLLM ceiling, context only).
    - TTFT and per-turn latency: non-warmup turns only (TTFT where t_first_token is known).
    - Throughput = non-warmup turns / (last t_done - first t_send), window over all records
      including warmups (interpretation, docs/decisions.md).
    """
    recs = list(records)
    turns = [r for r in recs if not r.is_warmup]
    prompt = sum(r.prompt_tokens for r in turns)
    cached = sum(r.cached_tokens for r in turns)
    cacheable = sum(max_cacheable_tokens(r.prompt_tokens, block_size) for r in turns)

    throughput = None
    if turns:
        window = max(r.t_done for r in recs) - min(r.t_send for r in recs)
        throughput = len(turns) / window if window > 0 else None

    per_agent: dict[str, MetricsSummary] = {}
    if by_agent:
        groups: dict[str, list[TurnRecord]] = defaultdict(list)
        for r in turns:
            if r.agent_id is not None:
                groups[r.agent_id].append(r)
        per_agent = {
            agent: summarize(group, block_size=block_size, by_agent=False)
            for agent, group in sorted(groups.items())
        }

    return MetricsSummary(
        num_turns=len(turns),
        num_warmup=len(recs) - len(turns),
        total_prompt_tokens=prompt,
        total_cached_tokens=cached,
        hit_rate=cached / prompt if prompt else None,
        max_possible_hit_rate=cacheable / prompt if prompt else None,
        ttft=latency_stats([r.ttft for r in turns if r.ttft is not None]),
        per_turn_latency=latency_stats([r.latency for r in turns]),
        throughput_turns_per_s=throughput,
        per_agent=per_agent,
    )
