"""Unit tests for the metrics layer (no GPU, no vLLM engine)."""

from types import SimpleNamespace

import numpy as np
import pytest

from cachescout.metrics import TurnRecord, latency_stats, max_cacheable_tokens, summarize
from cachescout.metrics.collectors import (
    HITS_COUNTER,
    QUERIES_COUNTER,
    PrefixCounters,
    prefix_counters,
    record_from_request_output,
)


def rec(rid: str, prompt: int, cached: int, t0: float, t1: float, t_end: float, **kw) -> TurnRecord:
    return TurnRecord(rid, prompt, cached, 8, t0, t1, t_end, **kw)


# --- max_cacheable_tokens (docs/vllm_internals.md §2) ---

@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, 0), (1, 0), (16, 0), (17, 16), (96, 80), (97, 96), (100, 96), (2048, 2032)],
)
def test_max_cacheable_tokens(n: int, expected: int) -> None:
    assert max_cacheable_tokens(n, 16) == expected


def test_max_cacheable_tokens_rejects_bad_block_size() -> None:
    with pytest.raises(ValueError):
        max_cacheable_tokens(10, 0)


# --- TurnRecord validation ---

def test_record_properties() -> None:
    r = rec("a", 100, 64, 1.0, 1.25, 2.0)
    assert r.ttft == pytest.approx(0.25)
    assert r.latency == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(prompt=10, cached=11, t0=0.0, t1=None, t_end=1.0),   # cached > prompt
        dict(prompt=10, cached=-1, t0=0.0, t1=None, t_end=1.0),   # negative cached
        dict(prompt=10, cached=0, t0=2.0, t1=None, t_end=1.0),    # done before send
        dict(prompt=10, cached=0, t0=0.0, t1=3.0, t_end=1.0),     # first token after done
    ],
)
def test_record_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        rec("x", **kwargs)


# --- latency_stats ---

def test_latency_stats_linear_percentiles() -> None:
    vals = [0.1, 0.2, 0.3, 0.4, 1.0]
    s = latency_stats(vals)
    assert s is not None
    assert s.n == 5
    assert s.mean == pytest.approx(0.4)
    assert s.median == pytest.approx(0.3)
    # linear: position 0.9*(5-1)=3.6 -> 0.4 + 0.6*(1.0-0.4) = 0.76
    assert s.p90 == pytest.approx(0.76)
    assert s.p99 == pytest.approx(np.percentile(vals, 99, method="linear"))


def test_latency_stats_empty() -> None:
    assert latency_stats([]) is None


# --- summarize ---

def test_summarize_hand_computed() -> None:
    records = [
        rec("a", 100, 0, 0.0, 0.1, 0.5, agent_id="P"),
        rec("b", 100, 64, 0.5, 0.55, 1.0, agent_id="C"),
        rec("c", 200, 192, 1.0, 1.02, 2.0, agent_id="P"),
    ]
    s = summarize(records)
    assert s.num_turns == 3 and s.num_warmup == 0
    assert s.total_prompt_tokens == 400 and s.total_cached_tokens == 256
    assert s.hit_rate == pytest.approx(256 / 400)
    assert s.max_possible_hit_rate == pytest.approx((96 + 96 + 192) / 400)
    assert s.ttft is not None and s.ttft.mean == pytest.approx((0.1 + 0.05 + 0.02) / 3)
    assert s.per_turn_latency is not None and s.per_turn_latency.median == pytest.approx(0.5)
    assert s.throughput_turns_per_s == pytest.approx(3 / 2.0)
    assert set(s.per_agent) == {"C", "P"}
    assert s.per_agent["P"].hit_rate == pytest.approx(192 / 300)
    assert s.per_agent["P"].per_agent == {}


def test_summarize_excludes_warmup_from_hit_rate_and_latency_but_not_window() -> None:
    records = [
        rec("w", 50, 0, 0.0, 0.2, 1.0, is_warmup=True),
        rec("a", 100, 96, 1.0, 1.1, 2.0),
    ]
    s = summarize(records)
    assert s.num_turns == 1 and s.num_warmup == 1
    assert s.hit_rate == pytest.approx(0.96)
    assert s.per_turn_latency is not None and s.per_turn_latency.n == 1
    # window spans the warmup too: 1 turn / (2.0 - 0.0)
    assert s.throughput_turns_per_s == pytest.approx(0.5)


def test_summarize_ttft_unknown() -> None:
    s = summarize([rec("a", 10, 0, 0.0, None, 1.0)])
    assert s.ttft is None
    assert s.per_turn_latency is not None


def test_summarize_empty_and_warmup_only() -> None:
    for records in ([], [rec("w", 10, 0, 0.0, None, 1.0, is_warmup=True)]):
        s = summarize(records)
        assert s.num_turns == 0
        assert s.hit_rate is None and s.ttft is None and s.per_turn_latency is None
        assert s.throughput_turns_per_s is None


def test_summary_to_dict_is_json_friendly() -> None:
    import json

    s = summarize([rec("a", 100, 64, 0.0, 0.1, 1.0, agent_id="P")])
    json.dumps(s.to_dict())


# --- collectors (duck-typed vLLM objects) ---

def fake_output(prompt_len: int, cached: int | None, n_out: int, ftl: float | None):
    metrics = None if ftl is None else SimpleNamespace(first_token_latency=ftl)
    return SimpleNamespace(
        request_id="r1",
        prompt_token_ids=list(range(prompt_len)),
        num_cached_tokens=cached,
        outputs=[SimpleNamespace(token_ids=list(range(n_out)))],
        metrics=metrics,
    )


def test_record_from_request_output_with_stats() -> None:
    r = record_from_request_output(fake_output(100, 64, 8, 0.05), t_send=10.0, t_done=10.5,
                                   agent_id="P")
    assert (r.prompt_tokens, r.cached_tokens, r.output_tokens) == (100, 64, 8)
    assert r.ttft == pytest.approx(0.05)
    assert r.vllm_first_token_latency == pytest.approx(0.05)
    assert r.agent_id == "P"


def test_record_from_request_output_without_stats() -> None:
    r = record_from_request_output(fake_output(100, None, 8, None), t_send=0.0, t_done=1.0)
    assert r.cached_tokens == 0
    assert r.ttft is None


def test_record_explicit_first_token_wins() -> None:
    r = record_from_request_output(fake_output(10, 0, 1, 0.9), t_send=0.0, t_done=1.0,
                                   t_first_token=0.2)
    assert r.ttft == pytest.approx(0.2)


def test_prefix_counters_sums_engines_and_ignores_others() -> None:
    metrics = [
        SimpleNamespace(name=QUERIES_COUNTER, value=100),
        SimpleNamespace(name=QUERIES_COUNTER, value=50),
        SimpleNamespace(name=HITS_COUNTER, value=64),
        SimpleNamespace(name="vllm:external_prefix_cache_hits", value=999),
        SimpleNamespace(name="vllm:num_requests_running", value=3),
    ]
    c = prefix_counters(metrics)
    assert c == PrefixCounters(queries=150, hits=64)
    assert c - PrefixCounters(100, 60) == PrefixCounters(50, 4)
