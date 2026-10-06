"""Serving metrics from paper Sec. 5.1: KV-cache hit rate, TTFT, per-turn latency, throughput."""

from cachescout.metrics.records import TurnRecord
from cachescout.metrics.summary import (
    LatencyStats,
    MetricsSummary,
    latency_stats,
    max_cacheable_tokens,
    summarize,
)

__all__ = [
    "LatencyStats",
    "MetricsSummary",
    "TurnRecord",
    "latency_stats",
    "max_cacheable_tokens",
    "summarize",
]
