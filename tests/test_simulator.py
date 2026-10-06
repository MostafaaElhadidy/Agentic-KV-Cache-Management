"""Simulator tests: must reproduce the vLLM GPU check (results/m1_metrics_check) exactly."""

import random

import pytest

from cachescout.core.runtime import CacheScoutParams, CacheScoutRuntime
from cachescout.sim.simulator import SimBlockPool, SimConfig, simulate
from cachescout.workload.trace import generate_trace

R = random.Random(0)


def toks(n: int) -> list[int]:
    return [R.randrange(1000, 150000) for _ in range(n)]


def test_shared_prefix_and_last_token_rule() -> None:
    pool = SimBlockPool(256, 16, None)
    prefix = toks(64)
    assert pool.run("a", prefix + toks(36), 8, "s", 2, 0.0) == 0
    assert pool.run("b", prefix + toks(36), 8, "s", 2, 0.0) == 64
    p96 = toks(96)
    assert pool.run("c", p96, 8, "s", 2, 0.0) == 0
    assert pool.run("d", p96, 8, "s", 2, 0.0) == 80


def test_lru_eviction_like_gpu_check() -> None:
    pool = SimBlockPool(256, 16, None)
    target = toks(200)
    assert pool.run("A", target, 8, "s", 2, 0.0) == 0
    assert pool.run("A2", target, 8, "s", 2, 0.0) == 192
    for i in range(4):
        assert pool.run(f"F{i}", toks(1200), 8, "s", 2, 0.0) == 0
    assert pool.run("A3", target, 8, "s", 2, 0.0) == 0


def test_block_conservation() -> None:
    pool = SimBlockPool(64, 16, CacheScoutRuntime(CacheScoutParams()))
    for i in range(30):
        pool.run(f"r{i}", toks(R.randint(20, 600)), R.randint(1, 40), "s", 2, float(i))
        assert len(pool.free) == pool.usable          # sequential: everything freed
        assert all(r == 0 for r in pool.ref[1:])


def test_request_larger_than_pool_raises() -> None:
    pool = SimBlockPool(10, 16, None)
    with pytest.raises(RuntimeError):
        pool.run("x", toks(400), 8, "s", 2, 0.0)


def test_neutral_hook_equals_vanilla_on_trace() -> None:
    trace = generate_trace(topology="selector", num_sessions=25, arrival_rate=1.0, seed=7)
    van = simulate(trace, SimConfig(num_gpu_blocks=120, system="vanilla"))
    hook = simulate(trace, SimConfig(num_gpu_blocks=120, system="lru_hook"))
    assert [r["cached_tokens"] for r in van["records"]] == \
           [r["cached_tokens"] for r in hook["records"]]


def test_all_systems_run_and_report() -> None:
    trace = generate_trace(topology="debate", num_sessions=15, arrival_rate=1.0, seed=3)
    for system in ("vanilla", "cachescout", "eviction_only", "warmup_only", "continuum"):
        out = simulate(trace, SimConfig(num_gpu_blocks=120, system=system))
        s = out["summary"]
        assert 0.0 <= s["hit_rate"] <= s["max_possible_hit_rate"] <= 1.0
        assert s["num_turns"] == sum(len(x.turns) for x in trace.sessions)
