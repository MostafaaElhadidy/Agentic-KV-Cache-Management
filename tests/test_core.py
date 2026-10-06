"""Unit tests for the CacheScout core (paper Eqs. 2-11, Alg. 1). No GPU."""

import math

import pytest

from cachescout.core.learner import TransitionLearner
from cachescout.core.runtime import CacheScoutParams, CacheScoutRuntime, Candidate
from cachescout.core.scorer import (
    bfs_hops,
    block_score,
    survival_score,
    survival_table,
    threshold_graph,
)
from cachescout.core.warmup import WarmupCoordinator, token_fingerprint


def learner_from(transitions, eps=0.01):
    lr = TransitionLearner(eps)
    for a, b in transitions:
        lr.observe(a, b)
    return lr


# ---- Eq. 3-5 ----

def test_eq3_smoothed_probability() -> None:
    lr = learner_from([("A", "B"), ("A", "B"), ("A", "C")], eps=0.5)
    # known agents A, B, C (n=3); row A counts B=2, C=1, total 3
    assert lr.prob("A", "B") == pytest.approx((2 + 0.5) / (3 + 1.5))
    assert lr.prob("A", "A") == pytest.approx(0.5 / 4.5)
    assert sum(lr.row("A").values()) == pytest.approx(1.0)
    # unseen row -> uniform
    assert lr.prob("C", "A") == pytest.approx(1 / 3)


def test_eq5_predict_and_ties() -> None:
    lr = learner_from([("A", "B"), ("A", "C"), ("A", "C")])
    assert lr.predict("A") == "C"
    lr2 = learner_from([("A", "B"), ("A", "C")])
    assert lr2.predict("A") == "B"          # tie -> discovery order
    assert TransitionLearner(0.1).predict("X") is None


def test_epsilon_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TransitionLearner(0.0)


# ---- Eq. 2 ----

def test_eq2_deterministic_chain_is_one() -> None:
    lr = learner_from([("A", "B"), ("B", "C"), ("C", "A")] * 10)
    assert lr.entropy_reduction() == pytest.approx(1.0)


def test_eq2_independent_is_zero() -> None:
    # every row has the same successor distribution as the marginal -> R = 0
    trans = [(a, b) for a in "AB" for b in "AB"] * 5
    lr = learner_from(trans)
    assert lr.entropy_reduction() == pytest.approx(0.0, abs=1e-12)


def test_eq2_hand_computed() -> None:
    # A->B x3, A->C x1, B->A x4.  marginal successors: B 3, C 1, A 4 (N=8)
    lr = learner_from([("A", "B")] * 3 + [("A", "C")] + [("B", "A")] * 4)
    h_next = -sum(p * math.log2(p) for p in (3 / 8, 1 / 8, 4 / 8))
    h_row_a = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    cond = 0.5 * h_row_a + 0.5 * 0.0
    assert lr.entropy_reduction() == pytest.approx(1 - cond / h_next)


# ---- Eq. 7-9, Fig. 7 ----

def test_eq7_threshold_graph_and_bfs() -> None:
    lr = learner_from([("A", "B")] * 9 + [("A", "C")] + [("B", "C")] * 5, eps=0.001)
    g = threshold_graph(lr, tau=0.5)
    assert g["A"] == ["B"] and g["B"] == ["C"] and g["C"] == []
    hops = bfs_hops(g, ["A"])
    assert hops == {"A": 0, "B": 1, "C": 2}


def test_eq8_survival_score() -> None:
    assert survival_score(0, 3) == 1.0
    assert survival_score(1, 3) == pytest.approx(2 / 3)
    assert survival_score(3, 3) == 0.0
    assert survival_score(math.inf, 3) == 0.0
    with pytest.raises(ValueError):
        survival_score(0, 0)


def test_survival_table_unreachable_absent() -> None:
    lr = learner_from([("A", "B")] * 10 + [("C", "A")], eps=0.001)
    table = survival_table(lr, ["A"], tau=0.5, e_max=4)
    assert table == {"A": 1.0, "B": 0.75}


def test_eq9_block_score() -> None:
    s = block_score(p_surv=0.5, age=10, num_tokens=16, lam=0.1, delta=0.05)
    assert s == pytest.approx((0.5 + 0.05) * (math.exp(-1.0) + 0.05) * 16)
    # floor delta keeps cold blocks positive
    assert block_score(0.0, 1e9, 16, 1.0, 0.05) == pytest.approx(0.05 * 0.05 * 16)


# ---- runtime / Alg. 1 ----

def make_rt(**kw) -> CacheScoutRuntime:
    params = dict(epsilon=0.001, tau=0.5, e_max=2, lam=0.0, delta=0.01)
    params.update(kw)
    return CacheScoutRuntime(CacheScoutParams(**params))


def test_alg1_transition_counted_only_on_agent_change() -> None:
    rt = make_rt()
    rt.observe_dispatch("r1", "fpA")
    rt.observe_dispatch("r2", "fpA")       # same agent: no transition (line 13)
    rt.observe_dispatch("r3", "fpB")
    assert rt.stats.transitions == 1
    assert rt.learner.counts[0][1] == 1
    assert rt.stats.dispatches == 3


def test_observe_idempotent_and_warmup_excluded() -> None:
    rt = make_rt()
    rt.observe_dispatch("r1", "fpA")
    rt.observe_dispatch("r1", "fpA")
    rt.observe_dispatch("cswarm-1", "fpB")
    assert rt.stats.dispatches == 1 and rt.stats.warmups_seen == 1
    assert rt.stats.transitions == 0
    assert rt.agent_of_request("cswarm-1") == 1


def test_session_scope_separates_interleaved_sessions() -> None:
    rt = make_rt(scope="session")
    for i in range(5):
        rt.observe_dispatch(f"s1-{i}a", "fpA", session="s1")
        rt.observe_dispatch(f"s2-{i}c", "fpC", session="s2")
        rt.observe_dispatch(f"s1-{i}b", "fpB", session="s1")
        rt.observe_dispatch(f"s2-{i}d", "fpD", session="s2")
    a, b, c, d = (rt.fingerprint_to_agent[f] for f in ("fpA", "fpB", "fpC", "fpD"))
    assert rt.learner.counts[a][b] == 5 and rt.learner.counts[c][d] == 5
    assert rt.learner.counts[a].get(c, 0) == 0


def test_select_victims_lru_matches_queue_order_uncached_first() -> None:
    rt = make_rt(policy="lru")
    cands = [Candidate(10, True, 16), Candidate(11, False, 16), Candidate(12, True, 16)]
    assert rt.select_victims(cands, 2) == [1, 0]


def test_select_victims_cachescout_protects_likely_next_agent() -> None:
    rt = make_rt()
    for i in range(10):
        rt.observe_dispatch(f"x{i}", "fpA")
        rt.observe_dispatch(f"y{i}", "fpB")
    rt.observe_dispatch("z", "fpC")        # current = C; graph A<->B; C unseen successors
    a, b = rt.fingerprint_to_agent["fpA"], rt.fingerprint_to_agent["fpB"]
    rt.on_blocks_used([1], a)
    rt.on_blocks_used([2], b)
    rt.observe_dispatch("w", "fpA")        # current = A: survival A=1, B=1-1/2
    cands = [Candidate(1, True, 16), Candidate(2, True, 16), Candidate(3, True, 16)]
    # block 3 has no agent -> lowest score; then B (hop 1) before A (hop 0)
    assert rt.select_victims(cands, 3) == [2, 1, 0]


def test_cachescout_degrades_to_lru_when_scores_flat() -> None:
    rt = make_rt()
    cands = [Candidate(i, True, 16) for i in range(5)]
    assert rt.select_victims(cands, 3) == [0, 1, 2]


def test_continuum_pins_recently_freed_blocks() -> None:
    rt = make_rt(policy="continuum", continuum_ttl_s=0.3)
    rt.on_blocks_freed([1], now=100.0)
    cands = [Candidate(1, True, 16), Candidate(2, True, 16)]
    assert rt.select_victims(cands, 1, now=100.1) == [1]      # block 1 pinned
    assert rt.select_victims(cands, 1, now=100.5) == [0]      # TTL expired
    assert rt.select_victims(cands, 2, now=100.1) == [1, 0]   # soft: pinned used last


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        CacheScoutParams(policy="nope")
    with pytest.raises(ValueError):
        CacheScoutParams(scope="nope")
    p = CacheScoutParams.from_dict({"tau": 0.2, "unknown": 1})
    assert p.tau == 0.2


# ---- warmup coordinator (Sec. 3.4) ----

def test_token_fingerprint_prefix_only() -> None:
    a = list(range(100))
    b = list(range(32)) + [999] * 68
    assert token_fingerprint(a, 2, 16) == token_fingerprint(b, 2, 16)
    assert token_fingerprint(a, 3, 16) != token_fingerprint(b, 3, 16)


def test_warmup_learns_anchor_and_gates() -> None:
    p = CacheScoutParams(epsilon=0.001, r_min=0.5, fingerprint_blocks=1)
    wc = WarmupCoordinator(p, block_size=4, minimal_prompt=[7], min_interval_s=1.0)
    anchor_a, anchor_b = [1] * 8, [2] * 8
    for s in range(3):
        sid = f"s{s}"
        wc.observe(anchor_a + [100 + s, 101], sid)
        wc.observe(anchor_b + [200 + s, 201, 202, 203, 204], sid)
    # A->B deterministic, B->A deterministic across sessions -> R high
    d = wc.maybe_warmup("s2", now=0.0)
    assert d is not None
    assert d.tokens == anchor_a + [7]     # current B predicts A; anchor = LCP, block-aligned
    assert wc.maybe_warmup("s2", now=0.5) is None   # rate limited
    assert wc.maybe_warmup("s2", now=2.0) is not None


def test_warmup_gate_blocks_unpredictable() -> None:
    p = CacheScoutParams(epsilon=0.001, r_min=0.99, fingerprint_blocks=1)
    wc = WarmupCoordinator(p, block_size=4, minimal_prompt=[], min_interval_s=0.0)
    seq = ["A", "B", "A", "C", "A", "B", "A", "C"]
    for i, ag in enumerate(seq):
        wc.observe([ord(ag)] * 8 + [i], "s0")
    assert wc.maybe_warmup("s0", now=0.0) is None
    assert wc.gated == 1


def test_session_mean_aggregation() -> None:
    rt = make_rt(scope="session", session_aggregate="mean", e_max=2, tau=0.5)
    for i in range(6):                     # learn A->B and C->D, per session
        rt.observe_dispatch(f"a{i}", "fpA", session=f"x{i}")
        rt.observe_dispatch(f"b{i}", "fpB", session=f"x{i}")
        rt.observe_dispatch(f"c{i}", "fpC", session=f"y{i}")
        rt.observe_dispatch(f"d{i}", "fpD", session=f"y{i}")
    rt.session_current.clear()
    rt.observe_dispatch("s1", "fpA", session="S1")   # S1 at A
    rt.observe_dispatch("s2", "fpC", session="S2")   # S2 at C
    a, b, c, d = (rt.fingerprint_to_agent[f] for f in ("fpA", "fpB", "fpC", "fpD"))
    assert rt.survival[a] == pytest.approx(0.5)      # 1 from S1, 0 from S2
    assert rt.survival[b] == pytest.approx(0.25)     # hop 1 -> 0.5 from S1
    assert rt.survival[c] == pytest.approx(0.5) and rt.survival[d] == pytest.approx(0.25)


def test_anchor_only_mapping_requires_two_sessions() -> None:
    rt = make_rt(block_mapping="anchor_only")
    rt.observe_dispatch("r1", "fpA", session="s1")
    a = rt.fingerprint_to_agent["fpA"]
    rt.on_blocks_used([1, 2], a, "s1", new=True)     # created by session s1
    rt.on_blocks_used([1], a, "s2")                  # block 1 reused by another session
    rt.survival = {a: 1.0}
    c1, c2 = Candidate(1, True, 16), Candidate(2, True, 16)
    assert rt.score(c1) > rt.score(c2)               # 2 is history-like -> survival 0
    rt.on_blocks_used([1], a, "s3", new=True)        # block reallocated -> no longer shared
    assert rt.score(c1) == pytest.approx(rt.score(c2))
