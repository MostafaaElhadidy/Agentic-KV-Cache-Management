"""Experiment runner: replay a trace on vLLM 0.31 with or without CacheScout (M1/M4/M5).

    python -m cachescout.run --config configs/experiments/main/local.yaml --system cachescout
    python -m cachescout.run --config configs/experiments/main/local.yaml --system vanilla

The ON/OFF switch:
- CLI `--system vanilla|cachescout|eviction_only|warmup_only|continuum|lru_hook` (wins), else
- config `system:` if present, else
- config `cachescout.enabled: true|false` (+ `cachescout.eviction`/`cachescout.warmup`: ablations).
Everything else (trace, model, seed, cache budget) comes from the same config, so runs differ only
in the system. `vanilla` uses vLLM's stock scheduler; the others load CacheScoutScheduler via
`scheduler_cls` and the CACHESCOUT_CONFIG env var.

Modes: `online` (Poisson session arrivals from the trace, streaming TTFT, warmups),
`sequential` (one request at a time in trace dispatch order; used to cross-check the simulator)
and `agents` (REAL multi-agent GSM8K sessions generated live; see cachescout.agents). `trace:` may
also point to the run directory of a live `agents` run: the recorded prompts are then replayed
(record-and-replay) with `online` or `sequential`.
Results: results/<experiment>/<system>/<tag>/result.json (+ engine-side runtime stats).
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from cachescout.config import apply_env, llm_kwargs, load_experiment
from cachescout.core.runtime import CacheScoutParams
from cachescout.core.warmup import WarmupCoordinator
from cachescout.metrics import TurnRecord, summarize
from cachescout.sim.simulator import SYSTEMS, system_flags
from cachescout.workload.stats import dispatch_order
from cachescout.workload.trace import Trace

MODES = ("online", "sequential", "agents")

REPO = Path(__file__).resolve().parents[2]
SCHEDULER_CLS = "cachescout.vllm_plugin.scheduler.CacheScoutScheduler"


def resolve_system(exp: dict[str, Any], cli_system: str | None) -> str:
    """CLI flag > config `system` > config `cachescout.enabled/eviction/warmup`."""
    if cli_system:
        system = cli_system
    elif exp.get("system"):
        system = exp["system"]
    else:
        cs = exp.get("cachescout") or {}
        if not cs.get("enabled", False):
            system = "vanilla"
        else:
            ev, wu = cs.get("eviction", True), cs.get("warmup", True)
            if not (ev or wu):
                raise ValueError("cachescout.enabled with eviction=false and warmup=false")
            system = {(True, True): "cachescout", (True, False): "eviction_only",
                      (False, True): "warmup_only"}[(bool(ev), bool(wu))]
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; choose from {SYSTEMS}")
    return system


def provenance(hw: dict[str, Any]) -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) \
        if shutil.which("nvidia-smi") else "unknown"
    return {"git_commit": run(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(run(["git", "status", "--porcelain", "--untracked-files=no"])),
            "python": sys.version.split()[0], "vllm": version("vllm"), "torch": version("torch"),
            "transformers": version("transformers"), "gpu": gpu,
            "hardware_profile": hw.get("name"),
            "timestamp": datetime.now().isoformat(timespec="seconds")}


def build_engine(hw: dict[str, Any], exp: dict[str, Any], system: str, stats_path: Path) -> Any:
    """Create an in-process AsyncLLM (engine core in a spawned process) for `system`."""
    apply_env(hw)
    policy, _ = system_flags(system)
    src = str(REPO / "src")
    os.environ["PYTHONPATH"] = src + (os.pathsep + os.environ["PYTHONPATH"]
                                      if os.environ.get("PYTHONPATH") else "")
    kwargs = llm_kwargs(hw, seed=int(exp.get("seed", 0)), log_stats=False)
    if policy is not None:
        params = {**(exp.get("cachescout", {}).get("params") or {}), "policy": policy}
        if system == "continuum":
            params["continuum_ttl_s"] = float((exp.get("continuum") or {}).get("ttl_s", 0.3))
        os.environ["CACHESCOUT_CONFIG"] = json.dumps({"params": params,
                                                      "stats_path": str(stats_path)})
        kwargs["scheduler_cls"] = SCHEDULER_CLS
    else:
        os.environ.pop("CACHESCOUT_CONFIG", None)
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))


async def _one_request(engine: Any, ids: list[int], out_tokens: int, rid: str,
                       session: str | None) -> tuple[float, float | None, float, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)
    t_send = time.perf_counter()
    t_first = None
    final = None
    async for out in engine.generate(TokensPrompt(prompt_token_ids=ids), sp, rid,
                                     session_id=session):
        if t_first is None and out.outputs and len(out.outputs[0].token_ids) > 0:
            t_first = time.perf_counter()
        final = out
    return t_send, t_first, time.perf_counter(), final


async def drive_online(engine: Any, trace: Trace, exp: dict[str, Any], warm: bool,
                       block_size: int, max_num_seqs: int) -> dict[str, Any]:
    """Sessions arrive at their trace times; turns run back-to-back with think gaps; after each
    turn the warmup coordinator may issue a background warmup (Sec. 3.4)."""
    cs_params = CacheScoutParams.from_dict((exp.get("cachescout") or {}).get("params"))
    wcfg = exp.get("warmup") or {}
    coord = (WarmupCoordinator(cs_params, block_size, wcfg.get("minimal_prompt", [1001, 1002,
                                                                               1003, 1004]),
                               float(wcfg.get("min_interval_s", 1.0))) if warm else None)
    records: list[TurnRecord] = []
    inflight = {"fg": 0, "warm": 0}
    warm_tasks: list[asyncio.Task] = []
    wid = [0]
    t0 = time.perf_counter()

    async def warmup(tokens: list[int], sid: str) -> None:
        rid = f"{cs_params.warmup_prefix}{wid[0]}"
        wid[0] += 1
        inflight["warm"] += 1
        try:
            ts, tf, td, final = await _one_request(engine, tokens, 1, rid, None)
            records.append(TurnRecord(rid, len(tokens), int(final.num_cached_tokens or 0), 1,
                                      ts, tf, td, session_id=sid, is_warmup=True))
        finally:
            inflight["warm"] -= 1

    async def session(s: Any) -> None:
        delay = s.arrival_s - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        for i, turn in enumerate(s.turns):
            ids = trace.prompt(s, i)
            inflight["fg"] += 1
            try:
                ts, tf, td, final = await _one_request(engine, ids, turn.output_tokens,
                                                       f"{s.session_id}|t{i}", s.session_id)
            finally:
                inflight["fg"] -= 1
            records.append(TurnRecord(f"{s.session_id}|t{i}", len(ids),
                                      int(final.num_cached_tokens or 0),
                                      len(final.outputs[0].token_ids), ts, tf, td,
                                      session_id=s.session_id, agent_id=turn.agent, turn_idx=i))
            if coord is not None:
                coord.observe(ids, s.session_id)
                # "idle periods" approximation: a free sequence slot and no warmup in flight
                if inflight["warm"] == 0 and inflight["fg"] < max_num_seqs:
                    d = coord.maybe_warmup(s.session_id, time.perf_counter() - t0)
                    if d is not None:
                        warm_tasks.append(asyncio.create_task(warmup(d.tokens, s.session_id)))
            await asyncio.sleep(turn.think_s)

    await asyncio.gather(*(session(s) for s in trace.sessions))
    if warm_tasks:
        await asyncio.gather(*warm_tasks)
    return {"records": records,
            "warmups_issued": coord.issued if coord else 0,
            "warmups_gated": coord.gated if coord else 0,
            "coordinator_R": coord.learner.entropy_reduction() if coord else None}


async def drive_sequential(engine: Any, trace: Trace) -> dict[str, Any]:
    """One request at a time in trace dispatch order (no warmups); for simulator cross-checks."""
    sess = {s.session_id: s for s in trace.sessions}
    records: list[TurnRecord] = []
    for _, sid, i in dispatch_order(trace):
        s = sess[sid]
        ids = trace.prompt(s, i)
        ts, tf, td, final = await _one_request(engine, ids, s.turns[i].output_tokens,
                                               f"{sid}|t{i}", sid)
        records.append(TurnRecord(f"{sid}|t{i}", len(ids), int(final.num_cached_tokens or 0),
                                  len(final.outputs[0].token_ids), ts, tf, td, session_id=sid,
                                  agent_id=s.turns[i].agent, turn_idx=i))
    return {"records": records, "warmups_issued": 0, "warmups_gated": 0, "coordinator_R": None}


class VLLMAgentClient:
    """LLMClient for cachescout.agents.session on the in-process AsyncLLM. After each call the
    warmup coordinator (if enabled) observes the real prompt and may issue a background warmup."""

    def __init__(self, engine: Any, coord: WarmupCoordinator | None, max_num_seqs: int,
                 warmup_prefix: str) -> None:
        self.engine, self.coord, self.max_num_seqs = engine, coord, max_num_seqs
        self.warmup_prefix = warmup_prefix
        self.inflight = {"fg": 0, "warm": 0}
        self.warm_tasks: list[asyncio.Task] = []
        self.warm_records: list[TurnRecord] = []
        self.t0 = time.perf_counter()
        self._wid = 0

    async def generate(self, prompt_ids: list[int], max_tokens: int, request_id: str,
                       session_id: str) -> Any:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from cachescout.agents.session import GenResult

        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        self.inflight["fg"] += 1
        t_send, t_first, final = time.perf_counter(), None, None
        try:
            async for out in self.engine.generate(TokensPrompt(prompt_token_ids=prompt_ids), sp,
                                                  request_id, session_id=session_id):
                if t_first is None and out.outputs and len(out.outputs[0].token_ids) > 0:
                    t_first = time.perf_counter()
                final = out
        finally:
            self.inflight["fg"] -= 1
        t_done = time.perf_counter()
        if self.coord is not None:
            self.coord.observe(prompt_ids, session_id)
            if self.inflight["warm"] == 0 and self.inflight["fg"] < self.max_num_seqs:
                d = self.coord.maybe_warmup(session_id, t_done - self.t0)
                if d is not None:
                    self.warm_tasks.append(asyncio.create_task(self._warm(d.tokens, session_id)))
        return GenResult(final.outputs[0].text, len(final.outputs[0].token_ids),
                         int(final.num_cached_tokens or 0), t_send, t_first, t_done)

    async def _warm(self, tokens: list[int], sid: str) -> None:
        rid = f"{self.warmup_prefix}{self._wid}"
        self._wid += 1
        self.inflight["warm"] += 1
        try:
            ts, tf, td, final = await _one_request(self.engine, tokens, 1, rid, None)
            self.warm_records.append(TurnRecord(rid, len(tokens), int(final.num_cached_tokens
                                                                       or 0), 1, ts, tf, td,
                                                session_id=sid, is_warmup=True))
        finally:
            self.inflight["warm"] -= 1


def agents_workload_name(a: dict[str, Any]) -> str:
    return f"gsm8k_{a.get('split', 'test')}_{a['topology']}_s{int(a['problem_seed'])}"


async def drive_agents(engine: Any, exp: dict[str, Any], warm: bool, block_size: int,
                       max_num_seqs: int, max_model_len: int) -> dict[str, Any]:
    """Live closed-loop multi-agent GSM8K sessions with Poisson arrivals (seeded per problem
    seed, so different seeds get different problems AND arrival times)."""
    import random

    from transformers import AutoTokenizer

    from cachescout.agents.definitions import AGENTS
    from cachescout.agents.gsm8k import load_split, select_problems
    from cachescout.agents.prompting import PromptBuilder, hf_chat_tokenizer
    from cachescout.agents.session import SessionConfig, run_session

    a = exp["agents"]
    seed = int(a["problem_seed"])
    problems = select_problems(load_split(a.get("split", "test")), seed, int(a["num_sessions"]))
    tok = AutoTokenizer.from_pretrained(exp["hardware_cfg"]["model"]["name"])
    cfg = SessionConfig(topology=a["topology"], max_tokens=int(a.get("max_tokens", 128)),
                        max_calls=int(a.get("max_calls", 14)),
                        max_tool_calls=int(a.get("max_tool_calls", 2)),
                        think_s=float(a.get("think_s", 0.2)))
    builder = PromptBuilder(hf_chat_tokenizer(tok), cfg.topology, max_model_len, cfg.max_tokens,
                            {k: v.name for k, v in AGENTS.items()})
    cs_params = CacheScoutParams.from_dict((exp.get("cachescout") or {}).get("params"))
    wcfg = exp.get("warmup") or {}
    coord = (WarmupCoordinator(cs_params, block_size, wcfg.get("minimal_prompt", [1001, 1002,
                                                                               1003, 1004]),
                               float(wcfg.get("min_interval_s", 1.0))) if warm else None)
    client = VLLMAgentClient(engine, coord, max_num_seqs, cs_params.warmup_prefix)
    arng = random.Random(f"arrivals-{seed}")
    arrivals, t = [], 0.0
    for _ in problems:
        t += arng.expovariate(float(a.get("arrival_rate", 0.2)))
        arrivals.append(t)
    t0 = time.perf_counter()

    async def one(i: int, problem: Any) -> Any:
        delay = arrivals[i] - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        return await run_session(client, builder, problem, cfg, f"g{seed}-{i:03d}",
                                 random.Random(f"route-{seed}-{i}"))

    sessions = await asyncio.gather(*(one(i, p) for i, p in enumerate(problems)))
    if client.warm_tasks:
        await asyncio.gather(*client.warm_tasks)
    records = [r for s in sessions for r in s.turn_records()] + client.warm_records
    return {"records": records, "sessions": sessions,
            "warmups_issued": coord.issued if coord else 0,
            "warmups_gated": coord.gated if coord else 0,
            "coordinator_R": coord.learner.entropy_reduction() if coord else None}


def load_trace(path: Path) -> Any:
    """A synthetic trace JSON, or the run directory / result.json of a live agents run
    (record-and-replay)."""
    from cachescout.agents.replay import RecordedTrace

    if path.is_dir() or path.name == "result.json":
        return RecordedTrace.from_run(path if path.is_dir() else path.parent)
    return Trace.load(path)


def set_dotted(exp: dict[str, Any], assignment: str) -> None:
    """Apply `a.b.c=value` (value parsed as YAML) to the experiment config."""
    import yaml

    key, _, raw = assignment.partition("=")
    if not key or not _:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    node = exp
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def apply_variant(exp: dict[str, Any], variant: str) -> str:
    """A named variant from the config's `variants:` table: {system, params} overrides.
    Returns the system to run. Used for extra ablations (e.g. literal Alg. 1, tau = 0)."""
    spec = (exp.get("variants") or {}).get(variant)
    if spec is None:
        raise ValueError(f"variant {variant!r} not defined in config `variants:`")
    cs = exp.setdefault("cachescout", {})
    cs["params"] = {**(cs.get("params") or {}), **(spec.get("params") or {})}
    return spec["system"]


def run(config: str, system_cli: str | None, overrides: dict[str, Any],
        variant: str | None = None, sets: list[str] | None = None) -> Path:
    exp = load_experiment(REPO / config, repo_root=REPO)
    for k, v in overrides.items():
        if v is not None:
            exp[k] = v
    for assignment in sets or []:
        set_dotted(exp, assignment)
    if variant is not None:
        system_cli = apply_variant(exp, variant)
    hw = exp["hardware_cfg"]
    if exp.get("num_gpu_blocks_override") is not None:
        hw["vllm"]["num_gpu_blocks_override"] = int(exp["num_gpu_blocks_override"])
    if exp.get("max_model_len") is not None:
        hw["vllm"]["max_model_len"] = int(exp["max_model_len"])
    system = resolve_system(exp, system_cli)
    _, warm = system_flags(system)
    mode = exp.get("mode", "online")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "agents":
        trace = None
        trace_name = agents_workload_name(exp["agents"])
        trace_meta: dict[str, Any] = {"name": trace_name, "role": "tune" if exp["agents"].get(
            "split", "test") == "train" else "eval", "workload": "gsm8k-agents",
            **exp["agents"]}
    else:
        trace = load_trace(REPO / exp["trace"])
        trace_name = trace.meta.get("name", Path(exp["trace"]).stem)
        trace_meta = trace.meta
    tag = exp.get("tag") or datetime.now().strftime("%Y%m%d-%H%M%S")
    blocks = hw["vllm"].get("num_gpu_blocks_override")
    label = variant or system
    out_dir = (REPO / "results" / exp["experiment"] / trace_name / label / f"b{blocks}_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "engine_runtime_stats.json"
    print(f"[run] experiment={exp['experiment']} system={system} workload={trace_name} "
          f"blocks={blocks} mode={mode} -> {out_dir.relative_to(REPO)}", flush=True)
    engine = build_engine(hw, exp, system, stats_path)
    t_start = time.perf_counter()
    try:
        bs = int(hw["vllm"]["block_size"])
        if mode == "agents":
            res = asyncio.run(drive_agents(engine, exp, warm, bs, int(hw["vllm"]["max_num_seqs"]),
                                           int(hw["vllm"]["max_model_len"])))
        elif mode == "sequential":
            res = asyncio.run(drive_sequential(engine, trace))
        else:
            res = asyncio.run(drive_online(engine, trace, exp, warm, bs,
                                           int(hw["vllm"]["max_num_seqs"])))
        num_blocks = engine.vllm_config.cache_config.num_gpu_blocks
    finally:
        engine.shutdown()
    wall = time.perf_counter() - t_start
    records = res["records"]
    summary = summarize(records, block_size=bs)
    engine_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    payload = {
        "experiment": exp["experiment"], "system": system, "variant": variant,
        "label": label, "trace_name": trace_name, "mode": mode,
        "config_path": config, "config": exp, "trace_meta": trace_meta,
        "num_gpu_blocks_reported": num_blocks, "wall_s": wall,
        "provenance": provenance(hw), "summary": summary.to_dict(),
        "warmups_issued": res["warmups_issued"], "warmups_gated": res["warmups_gated"],
        "coordinator_R": res["coordinator_R"], "engine_runtime": engine_stats,
        "records": [r.to_dict() for r in records],
    }
    if mode == "agents":
        from cachescout.agents.replay import write_prompts
        from cachescout.agents.summary import agent_run_summary

        sessions = res["sessions"]
        payload["agents_summary"] = agent_run_summary(sessions)
        payload["sessions"] = [x.to_dict() for x in sessions]
        write_prompts(out_dir, [{"request_id": c.request_id, "prompt_ids": c.prompt_ids}
                                for x in sessions for c in x.calls])
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    s = summary
    print(f"[run] done in {wall:.0f}s: turns={s.num_turns} hit_rate={s.hit_rate:.4f} "
          f"ttft_mean={s.ttft.mean * 1000 if s.ttft else float('nan'):.1f}ms "
          f"latency_mean={s.per_turn_latency.mean * 1000:.1f}ms "
          f"throughput={s.throughput_turns_per_s:.2f} turns/s warmups={res['warmups_issued']}",
          flush=True)
    if mode == "agents":
        a = payload["agents_summary"]
        print(f"[run] agents: sessions={a['sessions']} accuracy={a['gsm8k_accuracy']:.3f} "
              f"final_answer_rate={a['final_answer_rate']:.3f} fallback_rate="
              f"{a['fallback_rate']} tool_failure_rate={a['tool_failure_rate']} "
              f"trim_rate={a['trim_rate']:.3f} R={a['routing_R_measured']:.2f}", flush=True)
    print(f"[run] saved {(out_dir / 'result.json').relative_to(REPO)}", flush=True)
    return out_dir / "result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--system", choices=SYSTEMS, default=None)
    parser.add_argument("--blocks", type=int, default=None, help="num_gpu_blocks_override")
    parser.add_argument("--trace", default=None)
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--variant", default=None, help="named entry of the config `variants:`")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set agents.topology=random")
    args = parser.parse_args()
    if args.variant and args.system:
        parser.error("use either --system or --variant")
    run(args.config, args.system, {"num_gpu_blocks_override": args.blocks, "trace": args.trace,
                                   "mode": args.mode, "tag": args.tag}, variant=args.variant,
        sets=args.set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
