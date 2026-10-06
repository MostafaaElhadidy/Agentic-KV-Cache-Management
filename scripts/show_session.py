"""Print one recorded real-agent session as a readable conversation.

    python scripts/show_session.py results/real/<workload>/<system>/<run>        # first session
    python scripts/show_session.py <run_dir> --session g1-003                     # a given session
    python scripts/show_session.py <run_dir> --list                               # list sessions
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

NAMES = {"P": "PLANNER", "A": "ANALYST", "C": "CODER", "T": "TESTER", "R": "REVIEWER",
         "D": "DECIDER"}


def wrap(text: str, indent: str = "    ") -> str:
    lines = []
    for para in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, 96, initial_indent=indent, subsequent_indent=indent)
                     or [indent])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--session", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    path = Path(args.run_dir)
    res = json.loads((path / "result.json" if path.is_dir() else path).read_text())
    sessions = res.get("sessions")
    if not sessions:
        print("no real-agent sessions in this result (not a mode: agents run)")
        return 1
    if args.list:
        for s in sessions:
            print(f"{s['session_id']}  {s['problem_id']:<11} calls={s['num_calls']:>2} "
                  f"answer={s['final_answer']} gold={s['gold']} correct={s['correct']}")
        return 0
    s = next((x for x in sessions if x["session_id"] == args.session), None) \
        if args.session else sessions[0]
    if s is None:
        print(f"session {args.session} not found (use --list)")
        return 1
    print(f"Session {s['session_id']}  |  problem {s['problem_id']}  |  topology {s['topology']}  "
          f"|  system {res['system']}  |  cache {res.get('num_gpu_blocks_reported')} blocks")
    print("=" * 100)
    print("TASK:")
    print(wrap(s["question"]))
    print(f"    (gold answer: {s['gold']})")
    print("-" * 100)
    for c in s["calls"]:
        tag = " [forced final]" if c.get("force_final") else ""
        trim = f", {c['dropped_messages']} old msgs trimmed" if c["dropped_messages"] else ""
        print(f"[{c['turn_idx']:>2}] {NAMES.get(c['agent'], c['agent'])}{tag}   "
              f"(prompt {c['prompt_tokens']} tok, {c['cached_tokens']} from cache{trim}; "
              f"output {c['output_tokens']} tok)")
        print(wrap(c["output"]))
        if c.get("tool"):
            t = c["tool"]
            status = "ok" if t["ok"] else "FAILED"
            print(f"    -> tool {t['tool']}({t['argument']!r}) = {t['result']!r}  [{status}]")
        if c.get("route"):
            r = c["route"]
            nxt = NAMES.get(r["next"], r["next"]) if r["next"] else "(end)"
            fb = "  [FALLBACK]" if r.get("fallback") else ""
            print(f"    -> next: {nxt}   ({r['reason']}){fb}")
        print()
    print("-" * 100)
    print(f"FINAL ANSWER: {s['final_answer']}   gold: {s['gold']}   correct: {s['correct']}   "
          f"calls: {s['num_calls']}   capped: {s['capped']}   completion: "
          f"{s['t_end'] - s['t_arrival']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
