"""Record-and-replay: turn recorded real sessions into a trace for controlled comparisons.

A recording is the run directory of a live `mode: agents` run (result.json + prompts.jsonl.gz).
`RecordedTrace` exposes the same interface as `cachescout.workload.trace.Trace` (sessions with
arrival_s and turns[agent, output_tokens, think_s], and prompt(session, i)), so the existing
`drive_online` replay driver, the simulator and compare.py work on it unchanged.
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPTS_FILE = "prompts.jsonl.gz"
REPO = Path(__file__).resolve().parents[3]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return path.name


@dataclass
class RecordedTurn:
    agent: str
    output_tokens: int
    think_s: float              # recorded gap between this call's end and the next call's send


@dataclass
class RecordedSession:
    session_id: str
    arrival_s: float
    turns: list[RecordedTurn]
    prompts: list[list[int]]


class RecordedTrace:
    def __init__(self, meta: dict[str, Any], sessions: list[RecordedSession]) -> None:
        self.meta = meta
        self.sessions = sessions

    def prompt(self, session: RecordedSession, turn_idx: int) -> list[int]:
        return list(session.prompts[turn_idx])

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "RecordedTrace":
        run_dir = Path(run_dir)
        res = json.loads((run_dir / "result.json").read_text())
        if res.get("mode") != "agents":
            raise ValueError(f"{run_dir} is not a live agents run")
        prompts: dict[str, list[int]] = {}
        with gzip.open(run_dir / PROMPTS_FILE, "rt") as f:
            for line in f:
                row = json.loads(line)
                prompts[row["request_id"]] = row["prompt_ids"]
        sessions = []
        t0 = min(s["t_arrival"] for s in res["sessions"])
        for s in res["sessions"]:
            calls = s["calls"]
            turns = []
            for i, c in enumerate(calls):
                gap = (calls[i + 1]["t_send"] - c["t_done"]) if i + 1 < len(calls) else 0.0
                turns.append(RecordedTurn(c["agent"], max(int(c["output_tokens"]), 1),
                                          max(gap, 0.0)))
            sessions.append(RecordedSession(
                s["session_id"], s["t_arrival"] - t0, turns,
                [prompts[c["request_id"]] for c in calls]))
        meta = {"name": f"replay_{res['trace_name']}", "role": res.get("trace_meta", {})
                .get("role", "eval"), "recorded_from": _rel(run_dir),
                "recorded_system": res["system"], "source_meta": res.get("trace_meta")}
        return cls(meta, sessions)


def write_prompts(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(run_dir / PROMPTS_FILE, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
