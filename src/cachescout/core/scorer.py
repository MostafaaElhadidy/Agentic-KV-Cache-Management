"""Survival Scorer (paper Sec. 3.3): thresholded execution graph, BFS hops, Eqs. 7-9."""

import math
from collections import deque
from collections.abc import Hashable, Iterable

from cachescout.core.learner import TransitionLearner

Agent = Hashable
INF = math.inf


def threshold_graph(learner: TransitionLearner, tau: float) -> dict[Agent, list[Agent]]:
    """Eq. 7: (a, b) in E iff P(b | a) >= tau. Self-loops are excluded (never observed, Alg. 1).

    Computes each row total once (O(A^2) instead of calling Eq. 3 per pair, which is O(A^3)).
    """
    agents = learner.agents
    n, eps = len(agents), learner.epsilon
    graph: dict[Agent, list[Agent]] = {}
    for a in agents:
        row = learner.counts.get(a, {})
        denom = sum(row.values()) + eps * n
        graph[a] = [b for b in agents if b != a and (row.get(b, 0) + eps) / denom >= tau]
    return graph


def bfs_hops(graph: dict[Agent, list[Agent]], sources: Iterable[Agent]) -> dict[Agent, float]:
    """Minimum hop distance E[a] from the current agent(s) (Fig. 7); unreachable agents absent.

    Multiple sources = multi-source BFS (used with per-session scope, open_questions B4).
    """
    hops: dict[Agent, float] = {}
    queue: deque[Agent] = deque()
    for s in sources:
        if s not in hops:
            hops[s] = 0
            queue.append(s)
    while queue:
        u = queue.popleft()
        for v in graph.get(u, ()):
            if v not in hops:
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops


def survival_score(hop: float, e_max: int) -> float:
    """Eq. 8: p~_surv(a) = 1 - min(E[a], E_max) / E_max; unreachable (inf) -> 0."""
    if e_max <= 0:
        raise ValueError("E_max must be positive")
    return 1.0 - min(hop, e_max) / e_max


def survival_table(
    learner: TransitionLearner, current: Iterable[Agent], tau: float, e_max: int,
    graph: dict[Agent, list[Agent]] | None = None,
) -> dict[Agent, float]:
    """p~_surv for every reachable agent; agents missing from the result have score 0.
    A prebuilt `graph` (Eq. 7) may be passed to avoid rebuilding it for several sources."""
    g = threshold_graph(learner, tau) if graph is None else graph
    hops = bfs_hops(g, current)
    return {a: survival_score(h, e_max) for a, h in hops.items()}


def block_score(p_surv: float, age: float, num_tokens: int, lam: float, delta: float) -> float:
    """Eq. 9: Score(b) = (p~_surv(a_b) + delta) * (exp(-lambda * age(b)) + delta) * |b|.

    Lower scores are evicted first.
    """
    return (p_surv + delta) * (math.exp(-lam * max(age, 0.0)) + delta) * num_tokens
