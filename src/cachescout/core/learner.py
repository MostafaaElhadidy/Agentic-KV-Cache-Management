"""Transition Learner: online first-order Markov chain over agents (paper Sec. 3.2).

Pure Python, O(1) update per dispatch. Agents are arbitrary hashable IDs discovered online.
"""

import math
from collections import defaultdict
from collections.abc import Hashable

Agent = Hashable


class TransitionLearner:
    """Counts C_ij of observed agent transitions i -> j and derived quantities.

    Paper: Eq. 3 (smoothed probabilities), Eq. 4 (next-agent distribution), Eq. 5 (argmax),
    Eq. 2 (entropy reduction R, used by the prefetch gate Eq. 11).
    """

    def __init__(self, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0 (paper Eq. 3 smoothing)")
        self.epsilon = epsilon
        self.counts: dict[Agent, dict[Agent, int]] = defaultdict(lambda: defaultdict(int))
        self.agents: list[Agent] = []  # discovery order (deterministic iteration)
        self._known: set[Agent] = set()
        self.total = 0

    def add_agent(self, agent: Agent) -> None:
        """Register an agent seen for the first time (it may not have transitions yet)."""
        if agent not in self._known:
            self._known.add(agent)
            self.agents.append(agent)

    def observe(self, prev: Agent, nxt: Agent) -> None:
        """C_ij <- C_ij + 1 for one observed transition (paper Sec. 3.2)."""
        self.add_agent(prev)
        self.add_agent(nxt)
        self.counts[prev][nxt] += 1
        self.total += 1

    def prob(self, i: Agent, j: Agent) -> float:
        """Eq. 3: P_ij = (C_ij + eps) / sum_k (C_ik + eps), over the known agents k.

        Interpretation: the sum runs over all agents discovered so far (the paper does not define
        the state space for an online learner).
        """
        n = len(self.agents)
        if n == 0:
            return 0.0
        row = self.counts.get(i, {})
        row_total = sum(row.values())
        return (row.get(j, 0) + self.epsilon) / (row_total + self.epsilon * n)

    def row(self, i: Agent) -> dict[Agent, float]:
        """Eq. 4: predicted next-agent distribution p_{t+1} = P_{i,:}."""
        return {j: self.prob(i, j) for j in self.agents}

    def predict(self, i: Agent) -> Agent | None:
        """Eq. 5: argmax_j P_ij; ties broken by discovery order. None if nothing is known."""
        if not self.agents:
            return None
        row = self.counts.get(i, {})
        best, best_count = None, -1
        for j in self.agents:  # argmax of counts == argmax of Eq. 3 (same smoothing per row)
            c = row.get(j, 0)
            if c > best_count:
                best, best_count = j, c
        return best

    def entropy_reduction(self) -> float:
        """Eq. 2: R = 1 - H(A_{t+1} | A_t) / H(A_{t+1}), plug-in estimate from raw counts.

        Interpretation (open_questions B7): H(A_{t+1}) from the empirical marginal of successors;
        H(A_{t+1}|A_t) = sum_i (n_i / N) * H(row_i). Returns 0.0 when undefined (no data or a
        single possible successor overall).
        """
        if self.total == 0:
            return 0.0
        marginal: dict[Agent, int] = defaultdict(int)
        cond = 0.0
        for row in self.counts.values():
            n_i = sum(row.values())
            if n_i == 0:
                continue
            h_row = -sum((c / n_i) * math.log2(c / n_i) for c in row.values() if c)
            cond += (n_i / self.total) * h_row
            for j, c in row.items():
                marginal[j] += c
        h_next = -sum((c / self.total) * math.log2(c / self.total) for c in marginal.values() if c)
        if h_next <= 0:
            return 0.0
        return 1.0 - cond / h_next

    def state_size_bytes(self) -> int:
        """Rough size of the count table (for the Fig. 15a-style overhead report)."""
        import sys

        size = sys.getsizeof(self.counts) + sys.getsizeof(self.agents)
        for row in self.counts.values():
            size += sys.getsizeof(row) + sum(sys.getsizeof(v) for v in row.values())
        return size
