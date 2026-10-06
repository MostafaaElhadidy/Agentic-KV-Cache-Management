"""Next-agent rules per topology (paper Sec. 2.2.3 / Fig. 5) and output parsers.

- pipeline: fixed chain P -> A -> C -> T -> R -> D, then stop.
- random:   uniform over the other five agents (seeded per session).
- debate:   P -> C -> R -> D; D either answers or (REVISE / unclear) sends work back to C.
- selector: the agent's own last line `NEXT: <AGENT>`; missing/invalid/self -> counted fallback.
  Deviation from AutoGen SelectorGroupChat (which makes a separate selector LLM call): see
  docs/decisions.md.
"""

import random
import re
from dataclasses import dataclass

from cachescout.agents.definitions import AGENTS, NAME_TO_LETTER
from cachescout.agents.gsm8k import parse_number

TOPOLOGIES = ("pipeline", "random", "debate", "selector")
PIPELINE = ("P", "A", "C", "T", "R", "D")
_NEXT_RE = re.compile(r"NEXT\s*[:\-]\s*\**\s*\[?([A-Za-z]+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"FINAL\s+ANSWER\s*[:\-]?\s*(.+)", re.IGNORECASE)


@dataclass(frozen=True)
class Route:
    next_agent: str | None      # None = session ends
    reason: str
    fallback: bool = False


def parse_final_answer(text: str) -> str | None:
    """Number after 'FINAL ANSWER:' (last occurrence), or None."""
    matches = _FINAL_RE.findall(text)
    if not matches:
        return None
    return parse_number(matches[-1])


def parse_next(text: str) -> str | None:
    """Agent letter from the last 'NEXT: <NAME>' line, or None if missing/unknown."""
    matches = _NEXT_RE.findall(text)
    if not matches:
        return None
    name = matches[-1].upper()
    return NAME_TO_LETTER.get(name)


def first_agent(topology: str) -> str:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}")
    return "P"


def next_agent(topology: str, current: str, text: str, rng: random.Random) -> Route:
    """Routing decision after agent `current` produced `text` (and did not finish the task)."""
    if topology == "pipeline":
        i = PIPELINE.index(current)
        return Route(PIPELINE[i + 1], "fixed chain") if i + 1 < len(PIPELINE) \
            else Route(None, "end of chain")
    if topology == "random":
        others = [a for a in AGENTS if a != current]
        return Route(rng.choice(others), "uniform random")
    if topology == "debate":
        table = {"P": "C", "C": "R", "R": "D", "A": "C", "T": "R"}
        if current == "D":
            return Route("C", "decider asked for revision")
        return Route(table[current], "debate protocol")
    if topology == "selector":
        nxt = parse_next(text)
        if nxt is not None and nxt != current:
            return Route(nxt, f"NEXT: {AGENTS[nxt].name}")
        fb = "C" if current == "P" else "P"
        why = "missing NEXT line" if nxt is None else "agent named itself"
        return Route(fb, f"fallback ({why})", fallback=True)
    raise ValueError(f"unknown topology {topology!r}")
