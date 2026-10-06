"""Real multi-agent workload: tools, parsing, routing, trimming, fingerprints, sessions (no GPU)."""

import asyncio
import random

import pytest

from cachescout.agents.definitions import AGENTS, anchor_text
from cachescout.agents.gsm8k import Problem, gold_answer, is_correct, parse_number, select_problems
from cachescout.agents.prompting import Message, PromptBuilder
from cachescout.agents.routing import (
    PIPELINE,
    next_agent,
    parse_final_answer,
    parse_next,
)
from cachescout.agents.session import GenResult, SessionConfig, run_session
from cachescout.agents.summary import agent_run_summary
from cachescout.agents.tools import Scratchpad, ToolCall, calculate, execute, parse_tool_call

NAMES = {k: v.name for k, v in AGENTS.items()}


def fake_tokenize(messages: list[dict[str, str]]) -> list[int]:
    """Deterministic fake chat template: header + one token per character (prefix-stable)."""
    out: list[int] = []
    for m in messages:
        out += [1, {"system": 2, "user": 3, "assistant": 4}[m["role"]]]
        out += [ord(ch) % 5000 + 10 for ch in m["content"]] + [5]
    return out + [1, 4]


# ---- tools ----

@pytest.mark.parametrize(("expr", "want"), [("48 / 2 + 48", "72"), ("(3+4)*2", "14"),
                                            ("10 % 4", "2"), ("2**10", "1024"), ("7/2", "3.5"),
                                            ("1,200 * 3", "3600"), ("$5 * 4", "20"),
                                            ("-3 + 1", "-2"),
                                            ("(30 * 20) - (26 * 15) result: 210", "210"),
                                            ("24 - 3 = 21", "21")])
def test_calculator(expr: str, want: str) -> None:
    assert calculate(expr) == want


@pytest.mark.parametrize("bad", ["__import__('os')", "open('x')", "a + 1", "1/0", "2**99", "",
                                 "[1,2]"])
def test_calculator_rejects_unsafe(bad: str) -> None:
    result, ok = execute(ToolCall("calculator", bad), ("calculator",), Scratchpad())
    assert not ok and result.startswith("error:")


def test_scratchpad_and_permissions() -> None:
    sp = Scratchpad()
    assert execute(ToolCall("scratchpad", "write eggs=16"), ("scratchpad",), sp) == \
        ("saved note 1", True)
    assert execute(ToolCall("scratchpad", "read"), ("scratchpad",), sp)[0] == "1. eggs=16"
    res, ok = execute(ToolCall("calculator", "1+1"), ("scratchpad",), sp)
    assert not ok and "not available" in res


def test_parse_tool_call() -> None:
    assert parse_tool_call("I will compute.\nCALL calculator: 16 - 3 - 4") == \
        ToolCall("calculator", "16 - 3 - 4")
    assert parse_tool_call("**CALL scratchpad: read**".replace("**", "")) == \
        ToolCall("scratchpad", "read")
    assert parse_tool_call("no tool here") is None


# ---- parsing / scoring ----

def test_parse_next_and_final() -> None:
    assert parse_next("blah\nNEXT: REVIEWER") == "R"
    assert parse_next("NEXT: [Coder]") == "C"
    assert parse_next("NEXT: SOMEONE") is None
    assert parse_next("no line") is None
    assert parse_final_answer("FINAL ANSWER: $1,080") == "1080"
    assert parse_final_answer("REVISE: wrong") is None


def test_gsm8k_scoring_and_disjoint_seeds() -> None:
    assert gold_answer("blah\n#### 1,234") == "1234"
    assert parse_number("about 18 dollars") == "18"
    assert is_correct("18.0", "18") and not is_correct(None, "18") and not is_correct("17", "18")
    probs = [Problem("test", i, f"q{i}", str(i)) for i in range(100)]
    s1, s2 = select_problems(probs, 1, 20), select_problems(probs, 2, 20)
    assert len({p.index for p in s1} & {p.index for p in s2}) == 0
    assert select_problems(probs, 1, 20) == s1


# ---- routing ----

def test_pipeline_and_debate_rules() -> None:
    rng = random.Random(0)
    seq, cur = ["P"], "P"
    while True:
        r = next_agent("pipeline", cur, "", rng)
        if r.next_agent is None:
            break
        seq.append(r.next_agent)
        cur = r.next_agent
    assert tuple(seq) == PIPELINE
    assert next_agent("debate", "P", "", rng).next_agent == "C"
    assert next_agent("debate", "C", "", rng).next_agent == "R"
    assert next_agent("debate", "R", "", rng).next_agent == "D"
    assert next_agent("debate", "D", "REVISE: x", rng).next_agent == "C"


def test_random_excludes_self_and_selector_fallback() -> None:
    rng = random.Random(1)
    assert all(next_agent("random", "C", "", rng).next_agent != "C" for _ in range(200))
    r = next_agent("selector", "A", "work\nNEXT: CODER", rng)
    assert r.next_agent == "C" and not r.fallback
    r = next_agent("selector", "A", "no next line here", rng)  # round-robin: A -> C
    assert r.next_agent == "C" and r.fallback
    r = next_agent("selector", "P", "NEXT: PLANNER", rng)       # self -> fallback P -> A
    assert r.next_agent == "A" and r.fallback
    r = next_agent("selector", "D", "REVISE: x", rng)           # D -> P (wraps around)
    assert r.next_agent == "P" and r.fallback
    for text, want in (("done\n[PLANNER]: DECIDER", "D"), ("DECIDER", "D"),
                       ("ok\n[REVIEWER]", "R")):
        r = next_agent("selector", "C", text, rng)
        assert r.next_agent == want and r.lenient and not r.fallback
    r = next_agent("selector", "C", "The answer is 18 dollars", rng)
    assert r.fallback


# ---- prompts / trimming / fingerprints ----

def test_trimming_keeps_anchor_and_task() -> None:
    b = PromptBuilder(fake_tokenize, "selector", max_model_len=1400, max_tokens=100, names=NAMES)
    hist = [Message("C", "x" * 150) for _ in range(12)]
    built = b.build("R", "What is 2+2?", hist)
    assert built.dropped_messages > 0 and len(built.token_ids) <= 1300
    head = fake_tokenize([{"role": "system", "content": anchor_text("R", "selector")},
                          {"role": "user", "content": "Task: What is 2+2?"}])[:-2]
    assert built.token_ids[:len(head)] == head
    small = b.build("R", "What is 2+2?", [Message("C", "ok")])
    assert small.dropped_messages == 0


def test_selector_rule_lists_teammates_not_self() -> None:
    text = anchor_text("P", "selector")
    assert "NEXT: <NAME>" in text and "ANALYST, CODER, TESTER, REVIEWER or DECIDER" in text
    assert "PLANNER," not in text.split("chosen from:")[1]


def test_fingerprints_distinct_fake_tokenizer() -> None:
    for topo in ("selector", "pipeline"):
        fps = {tuple(fake_tokenize([{"role": "system", "content": anchor_text(a, topo)}])[:32])
               for a in AGENTS}
        assert len(fps) == 6


def test_fingerprints_distinct_real_tokenizer() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                         local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not cached")
    from cachescout.agents.prompting import hf_chat_tokenizer

    t = hf_chat_tokenizer(tok)
    for topo in ("selector", "debate"):
        fps, lengths = set(), []
        for a in AGENTS:
            ids = t([{"role": "system", "content": anchor_text(a, topo)},
                     {"role": "user", "content": "Task: x"}])
            fps.add(tuple(ids[:32]))
            lengths.append(len(tok(anchor_text(a, topo))["input_ids"]))
            assert "Alibaba" not in tok.decode(ids)       # Qwen default system prompt absent
        assert len(fps) == 6
        assert all(150 <= n <= 300 for n in lengths), lengths


# ---- full session with a fake engine ----

class FakeClient:
    """Canned replies per agent; counts calls; returns a deterministic GenResult."""

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.t = 0.0
        self.prompts: list[list[int]] = []

    async def generate(self, prompt_ids, max_tokens, request_id, session_id) -> GenResult:
        self.prompts.append(list(prompt_ids))
        agent = next(k for k, v in AGENTS.items()
                     if prompt_ids[2:2 + len(v.name)] == [ord(c) % 5000 + 10 for c in v.name])
        queue = self.replies.get(agent) or ["(nothing to add)"]
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        self.t += 1.0
        return GenResult(text, len(text.split()), 0, self.t, self.t + 0.1, self.t + 0.5)


def run(client, topology: str, max_calls: int = 14, question: str = "16 eggs, eats 3, uses 4, "
        "sells at $2. Dollars per day?", gold: str = "18"):
    b = PromptBuilder(fake_tokenize, topology, 1584, 128, NAMES)
    cfg = SessionConfig(topology=topology, max_calls=max_calls, think_s=0.0)
    return asyncio.run(run_session(client, b, Problem("test", 0, question, gold), cfg, "s0",
                                   random.Random(0), clock=lambda: 0.0))


def test_selector_session_with_tool_and_final_answer() -> None:
    client = FakeClient({
        "P": ["1. remaining eggs 2. times price\nNEXT: CODER"],
        "C": ["CALL calculator: (16 - 3 - 4) * 2", "The result is 18.\nNEXT: DECIDER"],
        "D": ["FINAL ANSWER: 18"],
    })
    res = run(client, "selector")
    assert res.final_answer == "18" and res.correct and not res.capped
    assert [c.agent for c in res.calls] == ["P", "C", "C", "D"]
    tool = res.calls[1].tool
    assert tool["result"] == "18" and tool["ok"]
    # the tool result is in the next prompt of the same agent (history really feeds back)
    assert [ord(c) % 5000 + 10 for c in "[tool:calculator] 18"][:10] in \
        [client.prompts[2][i:i + 10] for i in range(len(client.prompts[2]))]
    assert res.fallbacks == 0
    recs = res.turn_records()
    assert [r.agent_id for r in recs] == ["P", "C", "C", "D"] and recs[0].session_id == "s0"


def test_selector_fallbacks_counted_and_cap_forces_decider() -> None:
    client = FakeClient({"P": ["thinking"], "C": ["more thinking"], "D": ["FINAL ANSWER: 17"]})
    res = run(client, "selector", max_calls=5)
    assert res.fallbacks >= 2
    assert res.calls[-1].agent == "D" and res.calls[-1].force_final
    assert res.final_answer == "17" and not res.correct
    s = agent_run_summary([res])
    assert s["fallback_rate"] == 1.0 and s["gsm8k_accuracy"] == 0.0


def test_pipeline_session_visits_chain_once() -> None:
    client = FakeClient({"D": ["FINAL ANSWER: 18"]})
    res = run(client, "pipeline")
    assert res.agent_sequence() == list(PIPELINE) and res.correct


def test_debate_session_loops_until_final() -> None:
    client = FakeClient({"D": ["REVISE: recheck", "FINAL ANSWER: 18"]})
    res = run(client, "debate")
    assert res.agent_sequence() == ["P", "C", "R", "D", "C", "R", "D"] and res.correct


def test_record_replay_roundtrip(tmp_path) -> None:
    import json

    from cachescout.agents.replay import RecordedTrace, write_prompts

    client = FakeClient({"D": ["FINAL ANSWER: 18"]})
    res = run(client, "pipeline")
    payload = {"mode": "agents", "trace_name": "gsm8k_test_pipeline_s1", "system": "vanilla",
               "trace_meta": {"role": "eval"}, "sessions": [res.to_dict()]}
    (tmp_path / "result.json").write_text(json.dumps(payload))
    write_prompts(tmp_path, [{"request_id": c.request_id, "prompt_ids": c.prompt_ids}
                             for c in res.calls])
    tr = RecordedTrace.from_run(tmp_path)
    s = tr.sessions[0]
    assert [t.agent for t in s.turns] == [c.agent for c in res.calls]
    assert tr.prompt(s, 3) == res.calls[3].prompt_ids
    assert tr.meta["name"] == "replay_gsm8k_test_pipeline_s1"
    from cachescout.sim.simulator import SimConfig, simulate

    out = simulate(tr, SimConfig(num_gpu_blocks=200, system="cachescout"))
    assert out["summary"]["num_turns"] == len(res.calls)
