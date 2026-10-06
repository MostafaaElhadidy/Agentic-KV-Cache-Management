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

Modes: `online` (Poisson session arrivals from the trace, streaming TTFT, warmups) and
`sequential` (one request at a time in trace dispatch order; used to cross-check the simulator).
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
            "git_dirty": bool(run(["git", "status", "--porcelain"])),
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


def run(config: str, system_cli: str | None, overrides: dict[str, Any]) -> Path:
    exp = load_experiment(REPO / config, repo_root=REPO)
    for k, v in overrides.items():
        if v is not None:
            exp[k] = v
    hw = exp["hardware_cfg"]
    if exp.get("num_gpu_blocks_override") is not None:
        hw["vllm"]["num_gpu_blocks_override"] = int(exp["num_gpu_blocks_override"])
    if exp.get("max_model_len") is not None:
        hw["vllm"]["max_model_len"] = int(exp["max_model_len"])
    system = resolve_system(exp, system_cli)
    _, warm = system_flags(system)
    trace_path = REPO / exp["trace"]
    trace = Trace.load(trace_path)
    tag = exp.get("tag") or datetime.now().strftime("%Y%m%d-%H%M%S")
    blocks = hw["vllm"].get("num_gpu_blocks_override")
    out_dir = REPO / "results" / exp["experiment"] / f"{system}" / f"b{blocks}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "engine_runtime_stats.json"
    print(f"[run] experiment={exp['experiment']} system={system} trace={exp['trace']} "
          f"blocks={blocks} mode={exp.get('mode', 'online')} -> {out_dir.relative_to(REPO)}",
          flush=True)
    engine = build_engine(hw, exp, system, stats_path)
    t_start = time.perf_counter()
    try:
        bs = int(hw["vllm"]["block_size"])
        if exp.get("mode", "online") == "sequential":
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
        "experiment": exp["experiment"], "system": system, "mode": exp.get("mode", "online"),
        "config_path": config, "config": exp, "trace_meta": trace.meta,
        "num_gpu_blocks_reported": num_blocks, "wall_s": wall,
        "provenance": provenance(hw), "summary": summary.to_dict(),
        "warmups_issued": res["warmups_issued"], "warmups_gated": res["warmups_gated"],
        "coordinator_R": res["coordinator_R"], "engine_runtime": engine_stats,
        "records": [r.to_dict() for r in records],
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    s = summary
    print(f"[run] done in {wall:.0f}s: turns={s.num_turns} hit_rate={s.hit_rate:.4f} "
          f"ttft_mean={s.ttft.mean * 1000 if s.ttft else float('nan'):.1f}ms "
          f"latency_mean={s.per_turn_latency.mean * 1000:.1f}ms "
          f"throughput={s.throughput_turns_per_s:.2f} turns/s warmups={res['warmups_issued']}",
          flush=True)
    print(f"[run] saved {(out_dir / 'result.json').relative_to(REPO)}", flush=True)
    return out_dir / "result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--system", choices=SYSTEMS, default=None)
    parser.add_argument("--blocks", type=int, default=None, help="num_gpu_blocks_override")
    parser.add_argument("--trace", default=None)
    parser.add_argument("--mode", choices=("online", "sequential"), default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    run(args.config, args.system, {"num_gpu_blocks_override": args.blocks, "trace": args.trace,
                                   "mode": args.mode, "tag": args.tag})
    return 0


if __name__ == "__main__":
    sys.exit(main())
