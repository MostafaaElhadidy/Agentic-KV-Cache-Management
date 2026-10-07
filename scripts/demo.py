"""Run the real agents on YOUR task and print the conversation; optionally compare vanilla vs CacheScout.

    python scripts/demo.py --task "A train travels 60 km/h for 2.5 hours. How far does it go?" \\
        --topology pipeline --system cachescout
    python scripts/demo.py --task "..." --topology selector --compare        # + REPLAY comparison

Single-task mode: one live session of the six agents (same prompts, tools and routing as the real-agent
workload; same vLLM engine, plugin and block budget). Your task has no gold answer, so it is not scored.

--compare: (1) LIVE recording under vanilla of your task plus N background GSM8K *train* sessions (Poisson
arrivals), (2) REPLAY of exactly those recorded prompts under vanilla and under CacheScout at the same
block budget (open-loop: recorded prompts and output lengths, see docs/decisions.md), (3) a side-by-side
table. Only the replay is a controlled comparison.

Reuses cachescout.agents.session.run_session, PromptBuilder and run.VLLMAgentClient. Run GPU work through
scripts/gpu_run.sh (one GPU job at a time).
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cachescout.agents.definitions import AGENTS  # noqa: E402
from cachescout.agents.gsm8k import Problem, load_split, select_problems  # noqa: E402
from cachescout.agents.prompting import PromptBuilder, hf_chat_tokenizer  # noqa: E402
from cachescout.agents.replay import write_prompts  # noqa: E402
from cachescout.agents.routing import TOPOLOGIES  # noqa: E402
from cachescout.agents.session import SessionConfig, SessionResult, run_session  # noqa: E402
from cachescout.config import load_experiment  # noqa: E402
from cachescout.core.runtime import CacheScoutParams  # noqa: E402
from cachescout.core.warmup import WarmupCoordinator  # noqa: E402
from cachescout.metrics import summarize  # noqa: E402
from cachescout.run import VLLMAgentClient, build_engine, provenance  # noqa: E402
from cachescout.sim.simulator import system_flags  # noqa: E402

CONFIG = "configs/experiments/real/local.yaml"
DEMO_TRAIN_SEED = 104        # background problems: GSM8K TRAIN slice not used by tuning (101/102)


def custom_problem(task: str) -> Problem:
    return Problem("custom", 0, task.strip(), "")          # no gold answer


def finalize_custom(res: SessionResult) -> None:
    """A custom task has no gold answer: correctness is not applicable (None), not 'wrong'."""
    if res.problem_id.startswith("custom:"):
        res.correct = None  # type: ignore[assignment]


def write_recording(out_dir: Path, sessions: list[SessionResult], name: str, system: str,
                    blocks: int, extra: dict[str, Any]) -> Path:
    """Write a live run in the same format as run.py's agents mode, so RecordedTrace can replay it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [r for s in sessions for r in s.turn_records()]
    payload = {"experiment": "demo", "system": system, "variant": None, "label": system,
               "trace_name": name, "mode": "agents", "num_gpu_blocks_reported": blocks,
               "trace_meta": {"name": name, "role": "demo", "workload": "gsm8k-agents+custom"},
               "summary": summarize(records).to_dict(),
               "records": [r.to_dict() for r in records],
               "sessions": [s.to_dict() for s in sessions], **extra}
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    write_prompts(out_dir, [{"request_id": c.request_id, "prompt_ids": c.prompt_ids}
                            for s in sessions for c in s.calls])
    return out_dir


async def live_sessions(engine: Any, exp: dict[str, Any], system: str, problems: list[Problem],
                        arrivals: list[float], topology: str) -> list[SessionResult]:
    from transformers import AutoTokenizer

    hw = exp["hardware_cfg"]
    a = exp["agents"]
    _, warm = system_flags(system)
    cfg = SessionConfig(topology=topology, max_tokens=int(a["max_tokens"]),
                        max_calls=int(a["max_calls"]), max_tool_calls=int(a["max_tool_calls"]),
                        think_s=float(a["think_s"]))
    tok = AutoTokenizer.from_pretrained(hw["model"]["name"])
    builder = PromptBuilder(hf_chat_tokenizer(tok), topology, int(hw["vllm"]["max_model_len"]),
                            cfg.max_tokens, {k: v.name for k, v in AGENTS.items()})
    params = CacheScoutParams.from_dict(exp["cachescout"]["params"])
    coord = WarmupCoordinator(params, int(hw["vllm"]["block_size"]), [1001, 1002, 1003, 1004],
                              float(exp["warmup"]["min_interval_s"])) if warm else None
    client = VLLMAgentClient(engine, coord, int(hw["vllm"]["max_num_seqs"]), params.warmup_prefix)
    t0 = time.perf_counter()

    async def one(i: int, p: Problem) -> SessionResult:
        delay = arrivals[i] - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        res = await run_session(client, builder, p, cfg, f"demo-{i:03d}", random.Random(f"demo-{i}"))
        finalize_custom(res)
        return res

    out = await asyncio.gather(*(one(i, p) for i, p in enumerate(problems)))
    if client.warm_tasks:
        await asyncio.gather(*client.warm_tasks)
    return list(out)


def run_replay(rec_dir: Path, system: str, blocks: int, tag: str) -> Path:
    """Replay a recording in a separate process (fresh engine), exactly like the campaign does."""
    cmd = [sys.executable, "-m", "cachescout.run", "--config", CONFIG, "--system", system,
           "--blocks", str(blocks), "--trace", str(rec_dir.relative_to(REPO)), "--tag", tag,
           "--set", "mode=online", "--set", "experiment=demo"]
    env_path = str(REPO / "src")
    print(f"[demo] REPLAY under {system}: {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True, env={**os.environ, "PYTHONPATH": env_path})
    name = json.loads((rec_dir / "result.json").read_text())["trace_name"]
    return REPO / "results" / "demo" / f"replay_{name}" / system / f"b{blocks}_{tag}" / "result.json"


def compare_table(paths: dict[str, Path]) -> str:
    rows = []
    for system, p in paths.items():
        r = json.loads(p.read_text())
        s = r["summary"]
        custom = [x for x in r["records"] if x["session_id"] == "demo-000"]
        rows.append((system, s["hit_rate"], s["ttft"]["mean"] * 1e3,
                     s["per_turn_latency"]["mean"] * 1e3, s["total_cached_tokens"],
                     s["total_prompt_tokens"], sum(x["cached_tokens"] for x in custom),
                     sum(x["prompt_tokens"] for x in custom), str(p.relative_to(REPO))))
    lines = ["REPLAY comparison (identical recorded prompts for both systems; open-loop)",
             f"{'system':<11} {'hit rate':>8} {'TTFT mean':>10} {'latency':>9} "
             f"{'cached tok (all)':>17} {'cached tok (your task)':>23}"]
    for sysn, hit, ttft, lat, cached, prompt, c_cached, c_prompt, _ in rows:
        lines.append(f"{sysn:<11} {hit * 100:>7.1f}% {ttft:>8.1f}ms {lat:>7.1f}ms "
                     f"{cached:>8}/{prompt:<8} {c_cached:>11}/{c_prompt:<11}")
    lines += ["sources:"] + [f"  {r[0]}: {r[-1]}" for r in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="your question (any word problem)")
    ap.add_argument("--topology", choices=TOPOLOGIES, default="pipeline")
    ap.add_argument("--system", choices=("vanilla", "cachescout"), default="vanilla",
                    help="single-task mode only")
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--compare", action="store_true",
                    help="live vanilla recording + replay under vanilla and cachescout")
    ap.add_argument("--background", type=int, default=20, help="GSM8K train sessions (--compare)")
    ap.add_argument("--arrival-rate", type=float, default=0.2)
    args = ap.parse_args()

    from show_session import print_session

    exp = load_experiment(REPO / CONFIG, repo_root=REPO)
    hw = exp["hardware_cfg"]
    hw["vllm"]["num_gpu_blocks_override"] = args.blocks
    hw["vllm"]["max_model_len"] = int(exp["max_model_len"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    system = "vanilla" if args.compare else args.system
    problems = [custom_problem(args.task)]
    arrivals = [0.0]
    if args.compare:
        problems += select_problems(load_split("train"), DEMO_TRAIN_SEED, args.background)
        rng, t = random.Random(f"demo-arrivals-{stamp}"), 0.0
        for _ in problems[1:]:
            t += rng.expovariate(args.arrival_rate)
            arrivals.append(t)
    out_dir = REPO / "results" / "demo" / f"demo_{stamp}" / system / f"b{args.blocks}_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] LIVE {system}, topology={args.topology}, blocks={args.blocks}, "
          f"sessions={len(problems)} (your task + {len(problems) - 1} GSM8K train)", flush=True)
    engine = build_engine(hw, exp, system, out_dir / "engine_runtime_stats.json")
    try:
        sessions = asyncio.run(live_sessions(engine, exp, system, problems, arrivals,
                                             args.topology))
    finally:
        engine.shutdown()
    write_recording(out_dir, sessions, f"demo_{stamp}", system, args.blocks,
                    {"provenance": provenance(hw), "demo_task": args.task,
                     "topology": args.topology})
    print()
    print_session(sessions[0].to_dict(), system, args.blocks)
    print(f"\n[demo] live run saved to {out_dir.relative_to(REPO)}/result.json")
    if not args.compare:
        return 0
    paths = {s: run_replay(out_dir, s, args.blocks, f"demo_{stamp}") for s in
             ("vanilla", "cachescout")}
    print()
    print(compare_table(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
