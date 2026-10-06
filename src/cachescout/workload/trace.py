"""Synthetic multi-agent session traces (M2).

Structure follows the paper's description of agentic prompts (Sec. 1, Sec. 2.1, Fig. 1): every
invocation of agent `a` starts with that agent's fixed **anchor** (system prompt + tool
definitions), followed by the shared session history (task + all previous messages, as in an
AutoGen group chat). Next agents are sampled from a Fig. 5 transition table. One agent invocation
may make several LLM calls (tool call -> tool result -> next call); consecutive calls of the same
agent are not transitions (Alg. 1 line 13; Fig. 5 has an empty diagonal).

Engineering choice (docs/decisions.md): content is synthetic token IDs; only lengths and sharing
structure matter for KV-cache reuse. Workload parameters (anchor/output/tool lengths, turns) are
chosen to approximate the paper's workload statistics (anchor share 53-62% of prompt tokens,
Fig. 2; <= 13 turns, Fig. 3b; max request footprint ~93 blocks, Sec. 5.4), not taken from it.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cachescout.workload.topologies import AGENTS, normalized


@dataclass
class WorkloadProfile:
    """Length distributions (tokens) for one synthetic workload."""

    name: str = "selector-default"
    anchor_tokens: dict[str, int] = field(default_factory=lambda: {
        "P": 560, "A": 448, "C": 528, "T": 400, "R": 432, "D": 352})
    task_tokens: tuple[int, int] = (48, 96)
    header_tokens: int = 4                          # "name: " style speaker header
    output_tokens: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "P": (16, 40), "A": (16, 32), "C": (24, 56), "T": (16, 32), "R": (16, 40),
        "D": (16, 40)})
    tool_tokens: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "P": (0, 0), "A": (24, 72), "C": (0, 32), "T": (16, 56), "R": (0, 0), "D": (0, 0)})
    calls_per_invocation: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "P": (1, 1), "A": (1, 3), "C": (1, 3), "T": (1, 2), "R": (1, 1), "D": (1, 1)})
    turns: tuple[int, int] = (6, 12)                # agent invocations; Fig. 3b spans 13 turns
    max_prompt_tokens: int = 1400                   # keeps footprint <= ~93 blocks (Sec. 5.4)
    think_s: tuple[float, float] = (0.2, 0.5)       # between invocations; paper TTL 0.3 s
    tool_s: tuple[float, float] = (0.05, 0.25)      # between calls of one invocation (tool run)
    start_agent: str = "P"


@dataclass
class Turn:
    agent: str
    num_history_msgs: int      # prompt = anchor[agent] + task + messages[:num_history_msgs]
    output_tokens: int
    think_s: float             # delay after this turn completes before the next is sent


@dataclass
class Session:
    session_id: str
    arrival_s: float
    task: list[int]
    messages: list[list[int]]  # messages[j] = header + output_j + tool_result_j
    turns: list[Turn]


@dataclass
class Trace:
    meta: dict[str, Any]
    anchors: dict[str, list[int]]
    sessions: list[Session]

    def prompt(self, session: Session, turn_idx: int) -> list[int]:
        t = session.turns[turn_idx]
        out = list(self.anchors[t.agent]) + list(session.task)
        for msg in session.messages[: t.num_history_msgs]:
            out.extend(msg)
        return out

    def requests(self) -> list[dict[str, Any]]:
        """Flat list of all turns (for statistics and sequential replay)."""
        reqs = []
        for s in self.sessions:
            for i, t in enumerate(s.turns):
                reqs.append({"session_id": s.session_id, "turn_idx": i, "agent": t.agent,
                             "arrival_s": s.arrival_s, "output_tokens": t.output_tokens})
        return reqs

    def save(self, path: str | Path) -> None:
        data = {"meta": self.meta, "anchors": self.anchors,
                "sessions": [asdict(s) for s in self.sessions]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        data = json.loads(Path(path).read_text())
        sessions = [Session(s["session_id"], s["arrival_s"], s["task"], s["messages"],
                            [Turn(**t) for t in s["turns"]]) for s in data["sessions"]]
        return cls(data["meta"], data["anchors"], sessions)


def _tokens(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    return [rng.randrange(lo, hi) for _ in range(n)]


def generate_trace(
    *,
    topology: str,
    num_sessions: int,
    arrival_rate: float,
    seed: int,
    profile: WorkloadProfile | None = None,
    token_id_range: tuple[int, int] = (1000, 150000),
    anchor_seed: int = 12345,
) -> Trace:
    """Generate sessions with Poisson arrivals (rate in sessions/s; open_questions C3).

    `anchor_seed` fixes the agents' anchors independently of `seed`, so tuning and evaluation
    traces share the same agents (as a deployed system would) but differ in sessions.
    """
    prof = profile or WorkloadProfile()
    lo, hi = token_id_range
    arng = random.Random(anchor_seed)
    anchors = {a: _tokens(arng, prof.anchor_tokens[a], lo, hi) for a in AGENTS}
    rng = random.Random(seed)
    table = normalized(topology)
    sessions: list[Session] = []
    t_arrival = 0.0
    for sidx in range(num_sessions):
        t_arrival += rng.expovariate(arrival_rate) if arrival_rate > 0 else 0.0
        task = _tokens(rng, rng.randint(*prof.task_tokens), lo, hi)
        target_turns = 6 if topology == "pipeline" else rng.randint(*prof.turns)
        agent = prof.start_agent
        messages: list[list[int]] = []
        turns: list[Turn] = []
        hist_len = len(task)
        full = False
        for _ in range(target_turns):
            calls = rng.randint(*prof.calls_per_invocation[agent])
            for c in range(calls):
                if prof.anchor_tokens[agent] + hist_len > prof.max_prompt_tokens:
                    full = True
                    break
                out_len = rng.randint(*prof.output_tokens[agent])
                tool_len = rng.randint(*prof.tool_tokens[agent]) if c < calls - 1 else 0
                gap = rng.uniform(*prof.tool_s) if c < calls - 1 else rng.uniform(*prof.think_s)
                turns.append(Turn(agent, len(messages), out_len, gap))
                msg = _tokens(rng, prof.header_tokens + out_len + tool_len, lo, hi)
                messages.append(msg)
                hist_len += len(msg)
            if full or not turns:
                break
            row = table.get(agent, {})
            if not row:
                break                                   # terminal agent (pipeline end)
            agent = rng.choices(list(row), weights=list(row.values()))[0]
        sessions.append(Session(f"s{sidx:04d}", round(t_arrival, 4), task, messages, turns))
    meta = {"topology": topology, "num_sessions": num_sessions, "arrival_rate": arrival_rate,
            "seed": seed, "anchor_seed": anchor_seed, "token_id_range": list(token_id_range),
            "profile": asdict(prof), "generator": "cachescout.workload.trace.generate_trace"}
    return Trace(meta, anchors, sessions)
