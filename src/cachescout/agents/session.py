"""One multi-agent GSM8K session: agents talk, call tools, route, and finally answer.

Engine-agnostic: needs a `client` with `async generate(prompt_ids, max_tokens, request_id,
session_id) -> GenResult`. The vLLM client lives in run.py; tests use a fake client.
"""

import asyncio
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from cachescout.agents.definitions import AGENTS
from cachescout.agents.gsm8k import Problem, is_correct
from cachescout.agents.prompting import Message, PromptBuilder
from cachescout.agents.routing import first_agent, next_agent, parse_final_answer
from cachescout.agents.tools import Scratchpad, execute, parse_tool_call
from cachescout.metrics import TurnRecord


@dataclass
class GenResult:
    text: str
    output_tokens: int
    cached_tokens: int
    t_send: float
    t_first: float | None
    t_done: float


class LLMClient(Protocol):
    async def generate(self, prompt_ids: list[int], max_tokens: int, request_id: str,
                       session_id: str) -> GenResult: ...


@dataclass
class SessionConfig:
    topology: str
    max_tokens: int = 128
    max_calls: int = 14
    max_tool_calls: int = 2          # per agent invocation
    think_s: float = 0.2             # pause between agent invocations


@dataclass
class CallLog:
    turn_idx: int
    agent: str
    request_id: str
    prompt_tokens: int
    dropped_messages: int
    output: str
    output_tokens: int
    cached_tokens: int
    t_send: float
    t_first: float | None
    t_done: float
    force_final: bool = False
    tool: dict[str, Any] | None = None        # {"tool", "argument", "result", "ok"}
    tool_parse_failure: bool = False           # a CALL line that could not be executed
    route: dict[str, Any] | None = None        # {"next", "reason", "fallback"}
    prompt_ids: list[int] | None = None        # kept for record/replay (stored separately)


@dataclass
class SessionResult:
    session_id: str
    problem_id: str
    question: str
    gold: str
    topology: str
    t_arrival: float
    t_end: float = 0.0
    final_answer: str | None = None
    correct: bool = False
    capped: bool = False
    calls: list[CallLog] = field(default_factory=list)

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    @property
    def fallbacks(self) -> int:
        return sum(1 for c in self.calls if c.route and c.route.get("fallback"))

    @property
    def routing_decisions(self) -> int:
        return sum(1 for c in self.calls if c.route is not None)

    @property
    def trimmed_calls(self) -> int:
        return sum(1 for c in self.calls if c.dropped_messages > 0)

    @property
    def tool_parse_failures(self) -> int:
        return sum(1 for c in self.calls if c.tool_parse_failure)

    def turn_records(self) -> list[TurnRecord]:
        return [TurnRecord(c.request_id, c.prompt_tokens, c.cached_tokens, c.output_tokens,
                           c.t_send, c.t_first, c.t_done, session_id=self.session_id,
                           agent_id=c.agent, turn_idx=c.turn_idx) for c in self.calls]

    def agent_sequence(self) -> list[str]:
        """Agent per invocation (consecutive calls of one invocation collapsed)."""
        seq: list[str] = []
        for c in self.calls:
            if not seq or seq[-1] != c.agent:
                seq.append(c.agent)
        return seq

    def to_dict(self, include_prompt_ids: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_prompt_ids:
            for c in d["calls"]:
                c.pop("prompt_ids", None)
        d.update(num_calls=self.num_calls, fallbacks=self.fallbacks,
                 routing_decisions=self.routing_decisions, trimmed_calls=self.trimmed_calls,
                 tool_parse_failures=self.tool_parse_failures,
                 agent_sequence=self.agent_sequence())
        return d


async def run_session(client: LLMClient, builder: PromptBuilder, problem: Problem,
                      cfg: SessionConfig, session_id: str, rng: random.Random,
                      clock: Any = time.perf_counter) -> SessionResult:
    """Run one session to completion (FINAL ANSWER from the DECIDER, end of chain, or call cap)."""
    res = SessionResult(session_id, problem.problem_id, problem.question, problem.gold,
                        cfg.topology, t_arrival=clock())
    history: list[Message] = []
    scratch = Scratchpad()
    agent = first_agent(cfg.topology)
    n = 0
    done = False
    while n < cfg.max_calls and not done:
        if n == cfg.max_calls - 1 and agent != "D":
            agent = "D"                                    # last call: force the DECIDER
        force = n == cfg.max_calls - 1
        tool_calls = 0
        while True:                                        # one agent invocation
            built = builder.build(agent, problem.question, history, force_final=force)
            rid = f"{session_id}|t{n}"
            g = await client.generate(built.token_ids, cfg.max_tokens, rid, session_id)
            text = g.text.strip()
            log = CallLog(n, agent, rid, len(built.token_ids), built.dropped_messages, text,
                          g.output_tokens, g.cached_tokens, g.t_send, g.t_first, g.t_done,
                          force_final=force, prompt_ids=built.token_ids)
            res.calls.append(log)
            history.append(Message(agent, text))
            n += 1
            call = parse_tool_call(text)
            if call is not None:
                result, ok = execute(call, AGENTS[agent].tools, scratch)
                log.tool = {"tool": call.tool, "argument": call.argument, "result": result,
                            "ok": ok}
                log.tool_parse_failure = not ok
                history.append(Message(f"tool:{call.tool}", result))
                if tool_calls < cfg.max_tool_calls and n < cfg.max_calls - 1:
                    tool_calls += 1
                    continue                               # same agent continues
            break
        if agent == "D":
            final = parse_final_answer(text)
            if final is not None:
                res.final_answer = final
                done = True
                break
            if force:
                break
        route = next_agent(cfg.topology, agent, text, rng)
        log.route = {"next": route.next_agent, "reason": route.reason,
                     "fallback": route.fallback}
        if route.next_agent is None:
            break
        agent = route.next_agent
        if cfg.think_s > 0:
            await asyncio.sleep(cfg.think_s)
    res.capped = not done and n >= cfg.max_calls
    res.correct = is_correct(res.final_answer, problem.gold)
    res.t_end = clock()
    return res
