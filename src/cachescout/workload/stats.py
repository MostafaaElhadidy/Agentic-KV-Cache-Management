"""Trace statistics mirroring the paper's motivation study (Sec. 2.2, Figs. 2-4)."""

import math
from collections import Counter, defaultdict
from typing import Any

from cachescout.core.learner import TransitionLearner
from cachescout.workload.trace import Trace


def estimated_service_s(prompt_tokens: int, output_tokens: int) -> float:
    """Rough per-turn service time used only to order dispatches in time (not a measurement)."""
    return 0.05 + 0.0002 * prompt_tokens + 0.02 * output_tokens


def dispatch_order(trace: Trace) -> list[tuple[float, str, int]]:
    """(send_time, session_id, turn_idx) sorted by time, assuming no queueing (stats only)."""
    events = []
    for s in trace.sessions:
        t = s.arrival_s
        for i, turn in enumerate(s.turns):
            events.append((t, s.session_id, i))
            t += estimated_service_s(len(trace.prompt(s, i)), turn.output_tokens) + turn.think_s
    events.sort()
    return events


def _blocks(tokens: list[int], bs: int) -> list[tuple]:
    """Chained block identities (like vLLM block hashes): block i = all tokens up to its end."""
    return [tuple(tokens[: (i + 1) * bs]) for i in range(len(tokens) // bs)]


def trace_stats(trace: Trace, block_size: int = 16) -> dict[str, Any]:
    """Anchor share (Eq. 1, Fig. 2/3b), reuse per block type (Fig. 3a), R (Eq. 2, Fig. 4a),
    online top-1 next-agent accuracy (Fig. 4b), request footprint."""
    sess = {s.session_id: s for s in trace.sessions}
    order = dispatch_order(trace)

    total_prompt = total_anchor = 0
    phi_by_turn: dict[int, list[float]] = defaultdict(list)
    max_blocks = 0
    seen_blocks: Counter = Counter()
    block_kind: dict[int, str] = {}
    for _, sid, i in order:
        s = sess[sid]
        prompt = trace.prompt(s, i)
        anchor_len = len(trace.anchors[s.turns[i].agent])
        total_prompt += len(prompt)
        total_anchor += anchor_len
        phi_by_turn[i + 1].append(anchor_len / len(prompt))
        footprint = math.ceil((len(prompt) + s.turns[i].output_tokens) / block_size)
        max_blocks = max(max_blocks, footprint)
        for j, blk in enumerate(_blocks(prompt, block_size)):
            key = hash(blk)
            seen_blocks[key] += 1
            block_kind[key] = "anchor" if (j + 1) * block_size <= anchor_len else "history"

    reuse: dict[str, list[int]] = {"anchor": [], "history": []}
    for key, n in seen_blocks.items():
        reuse[block_kind[key]].append(n - 1)          # future reuses after first use
    mean_reuse = {k: (sum(v) / len(v) if v else 0.0) for k, v in reuse.items()}

    # R from true per-session transitions (Eq. 2)
    true_lr = TransitionLearner(1e-6)
    for s in trace.sessions:
        for a, b in zip(s.turns, s.turns[1:], strict=False):
            if a.agent != b.agent:
                true_lr.observe(a.agent, b.agent)

    # Online top-1 accuracy (Fig. 4b) for per-session and global (interleaved) streams
    def online_accuracy(scope: str) -> dict[str, Any]:
        lr = TransitionLearner(1e-6)
        prev_by: dict[str, str] = {}
        prev_global: str | None = None
        hits: list[int] = []
        for _, sid, i in order:
            a = sess[sid].turns[i].agent
            prev = prev_by.get(sid) if scope == "session" else prev_global
            if prev is not None and prev != a:
                hits.append(int(lr.predict(prev) == a))
                lr.observe(prev, a)
            prev_by[sid] = a
            prev_global = a
        first50 = hits[:50]
        last = hits[-200:]
        return {"n": len(hits), "acc_first50": sum(first50) / len(first50) if first50 else None,
                "acc_last200": sum(last) / len(last) if last else None,
                "acc_all": sum(hits) / len(hits) if hits else None}

    return {
        "num_sessions": len(trace.sessions),
        "num_turns": len(order),
        "anchor_share": total_anchor / total_prompt if total_prompt else None,   # Fig. 2
        "phi_by_turn": {t: sum(v) / len(v) for t, v in sorted(phi_by_turn.items())},  # Fig. 3b
        "mean_reuse_per_block": mean_reuse,                                       # Fig. 3a
        "anchor_to_history_reuse_ratio": (mean_reuse["anchor"] / mean_reuse["history"]
                                          if mean_reuse["history"] else None),
        "entropy_reduction_R_true": true_lr.entropy_reduction(),                 # Fig. 4a
        "online_accuracy_session": online_accuracy("session"),                   # Fig. 4b
        "online_accuracy_global": online_accuracy("global"),
        "max_request_blocks": max_blocks,                                        # Sec. 5.4
        "mean_prompt_tokens": total_prompt / len(order) if order else None,
        "duration_s_estimate": max((t for t, _, _ in order), default=0.0),
    }
