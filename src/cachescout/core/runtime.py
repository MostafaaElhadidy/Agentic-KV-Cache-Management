"""CacheScout runtime state machine (paper Alg. 1), independent of any serving engine.

Used by both the pure-Python simulator and the vLLM scheduler hook, so the policy code is shared.
Host responsibilities: call `on_step()` once per scheduler step, `observe_dispatch()` at
prefix-matching time, `on_blocks_used()` / `on_blocks_freed()` for block bookkeeping, and
`select_victims()` whenever free blocks must be taken for new allocations.
"""

import time
from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from cachescout.core.learner import TransitionLearner
from cachescout.core.scorer import block_score, survival_table

Agent = Hashable
POLICIES = ("lru", "cachescout", "continuum")
SCOPES = ("global", "session")


@dataclass
class CacheScoutParams:
    """Constants. None of these values are given by the paper; see docs/decisions.md."""

    policy: str = "cachescout"        # lru = vanilla order (validation), continuum = TTL baseline
    epsilon: float = 0.01             # Eq. 3
    tau: float = 0.3                  # Eq. 7
    e_max: int = 3                    # Eq. 8
    lam: float = 0.01                 # Eq. 9, per scheduler step
    delta: float = 0.05               # Eq. 9
    r_min: float = 0.3                # Eq. 11 (used by the warmup coordinator)
    fingerprint_blocks: int = 2       # Sec. 4: prefix blocks hashed into the agent fingerprint
    scope: str = "global"             # Alg. 1 literal = global current agent (open_questions B4)
    max_active_sessions: int = 8      # session scope: most recent sessions considered active
    session_aggregate: str = "mean"   # session scope: combine per-session survival (mean | max)
    block_mapping: str = "all"        # all = Alg. 1 line 12 literal; anchor_only (B3): only blocks
                                      # shared by >= 2 sessions inherit survival, others get 0
    continuum_ttl_s: float = 0.3      # Sec. 5.1 Continuum TTL
    warmup_prefix: str = "cswarm-"    # request-id prefix of warmup requests (Sec. 4)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        if self.fingerprint_blocks < 1:
            raise ValueError("fingerprint_blocks must be >= 1")
        if self.session_aggregate not in ("mean", "max"):
            raise ValueError("session_aggregate must be mean or max")
        if self.block_mapping not in ("all", "anchor_only"):
            raise ValueError("block_mapping must be all or anchor_only")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CacheScoutParams":
        d = dict(d or {})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class RuntimeStats:
    """Counters for reports (prediction accuracy, evictions, hot-path overhead)."""

    dispatches: int = 0
    warmups_seen: int = 0
    transitions: int = 0
    predictions: int = 0
    correct_predictions: int = 0
    victim_calls: int = 0
    victims_selected: int = 0
    pinned_evicted: int = 0
    observe_ns: int = 0
    select_ns: int = 0
    refresh_ns: int = 0
    refreshes: int = 0
    accuracy_curve: list[int] = field(default_factory=list)  # 1/0 per prediction, first 200

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """A free block offered for eviction, in free-queue (LRU) order."""

    block_id: int
    cached: bool          # has a prefix-cache hash
    num_tokens: int       # |b| in Eq. 9


class CacheScoutRuntime:
    """Alg. 1 state: transition counts, current agent(s), hop table, agentOf, lastAccess."""

    def __init__(self, params: CacheScoutParams) -> None:
        self.p = params
        self.learner = TransitionLearner(params.epsilon)
        self.step = 0
        self.fingerprint_to_agent: dict[Hashable, int] = {}
        self.block_agent: dict[int, Agent] = {}
        self.block_last: dict[int, int] = {}
        self.block_pin_until: dict[int, float] = {}
        self.block_first_session: dict[int, str | None] = {}
        self.block_shared: set[int] = set()
        self.request_agent: OrderedDict[str, Agent] = OrderedDict()  # bounded
        self.current: Agent | None = None                       # global scope a_t
        self.session_current: OrderedDict[str, Agent] = OrderedDict()  # session scope
        self.survival: dict[Agent, float] = {}
        self.stats = RuntimeStats()

    # ----- host hooks -------------------------------------------------------------------------

    def on_step(self) -> None:
        """One scheduler step elapsed (age unit of Eq. 9)."""
        self.step += 1

    def agent_for_fingerprint(self, fingerprint: Hashable) -> int:
        """Map a prompt-prefix fingerprint to a small integer agent ID (discovery order)."""
        agent = self.fingerprint_to_agent.get(fingerprint)
        if agent is None:
            agent = len(self.fingerprint_to_agent)
            self.fingerprint_to_agent[fingerprint] = agent
        return agent

    def observe_dispatch(
        self, request_id: str, fingerprint: Hashable, session: str | None = None
    ) -> Agent:
        """ObserveTouch at prefix-matching time (Alg. 1 lines 10-21, Sec. 4 hook 1).

        Idempotent per request (vLLM may look up the same waiting request several times).
        Warmup requests (id prefix) are mapped to their agent but excluded from learning (Sec. 4).
        """
        known = self.request_agent.get(request_id)
        if known is not None:
            return known
        t0 = time.perf_counter_ns()
        a = self.agent_for_fingerprint(fingerprint)
        self.request_agent[request_id] = a
        if len(self.request_agent) > 8192:       # bounded state (Sec. 4); keeps preempted ids
            self.request_agent.popitem(last=False)
        self.learner.add_agent(a)
        if request_id.startswith(self.p.warmup_prefix):
            self.stats.warmups_seen += 1
            self.stats.observe_ns += time.perf_counter_ns() - t0
            return a
        self.stats.dispatches += 1
        if self.p.scope == "global":
            prev = self.current
        else:
            key = session if session is not None else request_id
            prev = self.session_current.get(key)
        if a != prev:                                        # line 13
            if prev is not None:                             # line 14
                predicted = self.learner.predict(prev)
                self.stats.predictions += 1
                hit = int(predicted == a)
                self.stats.correct_predictions += hit
                if len(self.stats.accuracy_curve) < 200:
                    self.stats.accuracy_curve.append(hit)
                self.learner.observe(prev, a)                # lines 15-16
                self.stats.transitions += 1
            if self.p.scope == "global":
                self.current = a                             # line 20
            else:
                self.session_current[key] = a
                self.session_current.move_to_end(key)
                while len(self.session_current) > self.p.max_active_sessions:
                    self.session_current.popitem(last=False)
            self._refresh_survival()                         # lines 18-19
        elif self.p.scope == "session":
            key = session if session is not None else request_id
            self.session_current.move_to_end(key)
        self.stats.observe_ns += time.perf_counter_ns() - t0
        return a

    def on_blocks_used(self, block_ids: Sequence[int], agent: Agent | None,
                       session: str | None = None, new: bool = False) -> None:
        """Blocks touched (prefix hit) or allocated for a request: agentOf[b], lastAccess[b].

        `new=True` for freshly allocated blocks (their previous content was evicted). For
        `block_mapping: anchor_only`, a block becomes an anchor block once requests from two
        different sessions have used it (anchors are shared across sessions; history is not).
        """
        for b in block_ids:
            if agent is not None:
                self.block_agent[b] = agent
            self.block_last[b] = self.step
            self.block_pin_until.pop(b, None)
            if new or b not in self.block_first_session:
                self.block_first_session[b] = session
                self.block_shared.discard(b)
            elif session is not None and self.block_first_session[b] != session:
                self.block_shared.add(b)

    def on_blocks_freed(self, block_ids: Sequence[int], now: float | None = None) -> None:
        """Blocks returned to the free queue. lastAccess = free time (interpretation: the request
        used them until now; matches vLLM's LRU order). Continuum: pin for the TTL."""
        for b in block_ids:
            self.block_last[b] = self.step
        if self.p.policy == "continuum":
            until = (time.monotonic() if now is None else now) + self.p.continuum_ttl_s
            for b in block_ids:
                self.block_pin_until[b] = until

    def forget_request(self, request_id: str) -> None:
        self.request_agent.pop(request_id, None)

    def agent_of_request(self, request_id: str) -> Agent | None:
        return self.request_agent.get(request_id)

    # ----- eviction ---------------------------------------------------------------------------

    def score(self, c: Candidate) -> float:
        """Eq. 9 for one free cached block (ScoreBlock, Alg. 1 lines 23-27)."""
        agent = self.block_agent.get(c.block_id)
        p_surv = self.survival.get(agent, 0.0) if agent is not None else 0.0
        if self.p.block_mapping == "anchor_only" and c.block_id not in self.block_shared:
            p_surv = 0.0
        age = self.step - self.block_last.get(c.block_id, self.step)
        return block_score(p_surv, age, c.num_tokens, self.p.lam, self.p.delta)

    def select_victims(
        self, candidates: Sequence[Candidate], n: int, now: float | None = None
    ) -> list[int]:
        """Return indices into `candidates` (free-queue order) of the n blocks to reuse.

        Uncached blocks are always taken first, in queue order (they hold no reusable content;
        vLLM prepends them to the queue head). Then by policy:
        - lru: queue order (identical to vanilla vLLM `popleft_n`);
        - cachescout: ascending Eq. 9 score, ties by queue order (=> LRU when scores are flat);
        - continuum: unpinned blocks in queue order, then pinned ones (soft TTL pinning).
        """
        if n <= 0:
            return []
        t0 = time.perf_counter_ns()
        uncached = [i for i, c in enumerate(candidates) if not c.cached]
        chosen = uncached[:n]
        need = n - len(chosen)
        if need > 0:
            cached = [i for i, c in enumerate(candidates) if c.cached]
            if self.p.policy == "cachescout":
                cached.sort(key=lambda i: (self.score(candidates[i]), i))
            elif self.p.policy == "continuum":
                t = time.monotonic() if now is None else now
                pinned = {i for i in cached
                          if self.block_pin_until.get(candidates[i].block_id, -1.0) > t}
                order = [i for i in cached if i not in pinned] + [i for i in cached if i in pinned]
                self.stats.pinned_evicted += sum(1 for i in order[:need] if i in pinned)
                cached = order
            chosen += cached[:need]
        self.stats.victim_calls += 1
        self.stats.victims_selected += len(chosen)
        self.stats.select_ns += time.perf_counter_ns() - t0
        return chosen

    # ----- prediction for warmup / reporting --------------------------------------------------

    def current_agents(self) -> list[Agent]:
        if self.p.scope == "global":
            return [] if self.current is None else [self.current]
        return list(self.session_current.values())

    def _refresh_survival(self) -> None:
        """Alg. 1 lines 18-19: rebuild the thresholded graph and hop table when a_t changes.

        Session scope (interpretation, open_questions B4): one Eq. 8 table per active session's
        current agent, combined per agent by `session_aggregate` (mean approximates the chance
        that the agent is reused soon by any active session; max = multi-source BFS).
        """
        t0 = time.perf_counter_ns()
        if self.p.scope == "global" or self.p.session_aggregate == "max":
            self.survival = survival_table(self.learner, self.current_agents(), self.p.tau,
                                           self.p.e_max)
        else:
            sources = self.current_agents()
            per_agent: dict[Agent, float] = {}
            tables: dict[Agent, dict[Agent, float]] = {}
            for src in sources:
                if src not in tables:
                    tables[src] = survival_table(self.learner, [src], self.p.tau, self.p.e_max)
                for a, v in tables[src].items():
                    per_agent[a] = per_agent.get(a, 0.0) + v
            n = max(len(sources), 1)
            self.survival = {a: v / n for a, v in per_agent.items()}
        self.stats.refreshes += 1
        self.stats.refresh_ns += time.perf_counter_ns() - t0

    def summary(self) -> dict[str, Any]:
        s = self.stats.to_dict()
        s["prediction_accuracy"] = (self.stats.correct_predictions / self.stats.predictions
                                    if self.stats.predictions else None)
        s["entropy_reduction_R"] = self.learner.entropy_reduction()
        s["num_agents"] = len(self.learner.agents)
        s["learner_state_bytes"] = self.learner.state_size_bytes()
        s["observe_us_mean"] = (self.stats.observe_ns / 1e3 / max(self.stats.dispatches
                                + self.stats.warmups_seen, 1))
        s["select_us_mean"] = self.stats.select_ns / 1e3 / max(self.stats.victim_calls, 1)
        s["refresh_us_mean"] = self.stats.refresh_ns / 1e3 / max(self.stats.refreshes, 1)
        s["params"] = asdict(self.p)
        return s
