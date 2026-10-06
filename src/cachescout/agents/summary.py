"""Run-level statistics for real-agent sessions (routing, tools, answers, trimming, completion)."""

import statistics
from collections.abc import Sequence
from typing import Any

from cachescout.agents.session import SessionResult
from cachescout.core.learner import TransitionLearner


def routing_R(sessions: Sequence[SessionResult]) -> float:
    """Measured entropy reduction R (paper Eq. 2) from per-session agent transitions."""
    lr = TransitionLearner(1e-6)
    for s in sessions:
        seq = s.agent_sequence()
        for a, b in zip(seq, seq[1:], strict=False):
            if a != b:
                lr.observe(a, b)
    return lr.entropy_reduction()


def agent_run_summary(sessions: Sequence[SessionResult]) -> dict[str, Any]:
    n = len(sessions)
    calls = [c for s in sessions for c in s.calls]
    tool_attempts = [c for c in calls if c.tool is not None]
    decisions = sum(s.routing_decisions for s in sessions)
    completion = [s.t_end - s.t_arrival for s in sessions]
    answered = [s for s in sessions if s.final_answer is not None]
    return {
        "sessions": n,
        "problem_ids": [s.problem_id for s in sessions],
        "calls": len(calls),
        "calls_per_session_mean": len(calls) / n if n else None,
        "final_answer_rate": len(answered) / n if n else None,
        "gsm8k_accuracy": sum(s.correct for s in sessions) / n if n else None,
        "correct": sum(s.correct for s in sessions),
        "capped_sessions": sum(s.capped for s in sessions),
        "routing_decisions": decisions,
        "fallbacks": sum(s.fallbacks for s in sessions),
        "fallback_rate": (sum(s.fallbacks for s in sessions) / decisions) if decisions else None,
        "lenient_routes": sum(s.lenient_routes for s in sessions),
        "strict_next_rate": ((decisions - sum(s.fallbacks for s in sessions)
                              - sum(s.lenient_routes for s in sessions)) / decisions)
        if decisions else None,
        "tool_calls": len(tool_attempts),
        "tool_parse_failures": sum(1 for c in tool_attempts if c.tool_parse_failure),
        "tool_failure_rate": (sum(1 for c in tool_attempts if c.tool_parse_failure)
                              / len(tool_attempts)) if tool_attempts else None,
        "tool_calls_by_tool": {t: sum(1 for c in tool_attempts if c.tool["tool"] == t)
                               for t in sorted({c.tool["tool"] for c in tool_attempts})},
        "trimmed_calls": sum(1 for c in calls if c.dropped_messages > 0),
        "trim_rate": (sum(1 for c in calls if c.dropped_messages > 0) / len(calls))
        if calls else None,
        "prompt_tokens_mean": statistics.mean(c.prompt_tokens for c in calls) if calls else None,
        "prompt_tokens_max": max((c.prompt_tokens for c in calls), default=None),
        "output_tokens_mean": statistics.mean(c.output_tokens for c in calls) if calls else None,
        "session_completion_s_mean": statistics.mean(completion) if completion else None,
        "session_completion_s_median": statistics.median(completion) if completion else None,
        "routing_R_measured": routing_R(sessions),
        "calls_per_agent": {a: sum(1 for c in calls if c.agent == a) for a in "PACTRD"},
    }
