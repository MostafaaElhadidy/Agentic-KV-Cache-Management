"""Tests for the synthetic trace generator and statistics (M2). No GPU."""

from cachescout.workload.stats import trace_stats
from cachescout.workload.topologies import TOPOLOGIES, normalized
from cachescout.workload.trace import Trace, WorkloadProfile, generate_trace


def small(topology: str, seed: int = 0, n: int = 20) -> Trace:
    return generate_trace(topology=topology, num_sessions=n, arrival_rate=1.0, seed=seed)


def test_rows_normalised_and_no_self_loops() -> None:
    for topo in TOPOLOGIES:
        for a, row in normalized(topo).items():
            assert a not in row
            if row:
                assert abs(sum(row.values()) - 1.0) < 1e-9


def test_deterministic_and_seed_dependent() -> None:
    a, b, c = small("selector", 1), small("selector", 1), small("selector", 2)
    assert a.prompt(a.sessions[0], 0) == b.prompt(b.sessions[0], 0)
    assert [s.task for s in a.sessions] != [s.task for s in c.sessions]
    assert a.anchors == c.anchors           # same agents across seeds (anchor_seed fixed)


def test_prompt_structure_anchor_then_history() -> None:
    t = small("selector")
    s = t.sessions[0]
    for i, turn in enumerate(s.turns):
        p = t.prompt(s, i)
        anchor = t.anchors[turn.agent]
        assert p[: len(anchor)] == anchor
        assert p[len(anchor): len(anchor) + len(s.task)] == s.task
        assert len(p) <= WorkloadProfile().max_prompt_tokens + 200


def test_history_prefix_reused_by_same_agent() -> None:
    t = small("selector", n=40)
    found = False
    for s in t.sessions:
        for i in range(1, len(s.turns)):
            if s.turns[i].agent == s.turns[i - 1].agent:   # multi-call invocation
                prev, cur = t.prompt(s, i - 1), t.prompt(s, i)
                assert cur[: len(prev)] == prev
                found = True
    assert found


def test_pipeline_is_deterministic_chain() -> None:
    st = trace_stats(small("pipeline", n=30))
    assert abs(st["entropy_reduction_R_true"] - 1.0) < 1e-9


def test_save_load_roundtrip(tmp_path) -> None:
    t = small("debate", n=5)
    path = tmp_path / "t.json"
    t.save(path)
    u = Trace.load(path)
    assert u.prompt(u.sessions[2], 1) == t.prompt(t.sessions[2], 1)
    assert u.meta["topology"] == "debate"
