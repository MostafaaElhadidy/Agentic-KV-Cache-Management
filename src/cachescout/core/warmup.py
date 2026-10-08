"""Background Prefetch Coordinator (paper Sec. 3.4, Alg. 1 lines 28-33, Eqs. 10-11).

Runs on the client/driver side and issues warmup requests through the normal serving API.
Engineering choice, a deviation from Fig. 6's single shared matrix (docs/decisions.md, 2026-10-08
"Prefetch coordinator keeps its own transition learner"): the coordinator keeps its own
TransitionLearner fed with the same prompt-prefix fingerprints the engine-side runtime uses, so no
state has to cross the engine process boundary. Anchors are learned online as the block-aligned
longest common prefix of an agent's prompts seen in at least two different sessions (no framework
annotations).
"""

import hashlib
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from cachescout.core.learner import TransitionLearner
from cachescout.core.runtime import CacheScoutParams

Agent = Hashable


def token_fingerprint(tokens: Sequence[int], num_blocks: int, block_size: int) -> str:
    """Fingerprint of the first `num_blocks` full blocks of a prompt (Sec. 4: prefix-hash
    fingerprint). Uses fewer blocks if the prompt is shorter (at least its first token)."""
    n = min(len(tokens), num_blocks * block_size)
    full = (n // block_size) * block_size or n
    data = ",".join(map(str, tokens[:full])).encode()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


@dataclass
class WarmupDecision:
    agent: Agent
    tokens: list[int]
    r_value: float


class WarmupCoordinator:
    """Decides, between agent turns, whether to warm the predicted next agent's anchor."""

    def __init__(self, params: CacheScoutParams, block_size: int, minimal_prompt: Sequence[int],
                 min_interval_s: float) -> None:
        self.p = params
        self.block_size = block_size
        self.minimal_prompt = list(minimal_prompt)
        self.min_interval_s = min_interval_s
        self.learner = TransitionLearner(params.epsilon)
        self.fp_to_agent: dict[str, int] = {}
        self.anchor: dict[Agent, list[int]] = {}
        self.anchor_sessions: dict[Agent, set[str]] = {}
        self.current: Agent | None = None
        self.session_current: dict[str, Agent] = {}
        self.last_warm: dict[Agent, float] = {}
        self.issued = 0
        self.gated = 0

    def observe(self, prompt: Sequence[int], session: str) -> Agent:
        """Record one real (non-warmup) dispatch: agent identity, anchor estimate, transition."""
        fp = token_fingerprint(prompt, self.p.fingerprint_blocks, self.block_size)
        a = self.fp_to_agent.setdefault(fp, len(self.fp_to_agent))
        self.learner.add_agent(a)
        self._update_anchor(a, prompt, session)
        prev = self.current if self.p.scope == "global" else self.session_current.get(session)
        if prev is not None and prev != a:
            self.learner.observe(prev, a)
        if prev != a:
            if self.p.scope == "global":
                self.current = a
            else:
                self.session_current[session] = a
        return a

    def _update_anchor(self, a: Agent, prompt: Sequence[int], session: str) -> None:
        sessions = self.anchor_sessions.setdefault(a, set())
        if a not in self.anchor:
            self.anchor[a] = list(prompt)
        elif session not in sessions:
            old = self.anchor[a]
            n = 0
            for x, y in zip(old, prompt, strict=False):
                if x != y:
                    break
                n += 1
            self.anchor[a] = old[: (n // self.block_size) * self.block_size]
        sessions.add(session)

    def maybe_warmup(self, session: str, now: float) -> WarmupDecision | None:
        """BetweenStep: if R >= R_min (Eq. 11), warm a* = argmax P(a | a_t) (Eq. 10)."""
        r = self.learner.entropy_reduction()
        if r < self.p.r_min:
            self.gated += 1
            return None
        cur = self.current if self.p.scope == "global" else self.session_current.get(session)
        if cur is None:
            return None
        target = self.learner.predict(cur)
        if target is None or len(self.anchor_sessions.get(target, ())) < 2:
            return None                       # anchor not yet separated from session history
        anchor = self.anchor.get(target) or []
        if not anchor:
            return None
        if now - self.last_warm.get(target, -1e18) < self.min_interval_s:
            return None                       # rate limit (Sec. 4)
        self.last_warm[target] = now
        self.issued += 1
        return WarmupDecision(target, anchor + self.minimal_prompt, r)
