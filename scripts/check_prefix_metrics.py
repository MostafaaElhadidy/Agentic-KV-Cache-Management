"""M1 check: verify prefix-cache accounting and LRU eviction on vanilla vLLM. LOADS THE MODEL.

Run by hand only (docs/PLAN.md M1):
    python scripts/check_prefix_metrics.py --config configs/experiments/m1_metrics_check/local.yaml
Add --plan-only to print the block arithmetic without loading anything.

Checks (all requests sent strictly one at a time, never batched):
  (a) shared prefix: B shares a 64-token prefix with A  -> B.cached == 64
  (b) identical prompt of N tokens sent twice           -> 2nd cached == floor((N-1)/16)*16
  (c) LRU eviction: A, A again, fillers > usable blocks, A -> final A.cached == 0
  (d) per-request totals vs vLLM's prefix-cache counters (small tolerance, gap explained)
Results (config, provenance, records, summary) are saved to results/m1_metrics_check/<timestamp>/.
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.config import llm_kwargs, load_experiment  # noqa: E402
from cachescout.kv_blocks import plan_eviction  # noqa: E402
from cachescout.metrics import TurnRecord, max_cacheable_tokens, summarize  # noqa: E402
from cachescout.metrics.collectors import prefix_counters, record_from_request_output  # noqa: E402


def provenance(hw: dict[str, Any]) -> dict[str, Any]:
    """Git commit, package versions, GPU name."""
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    gpu = "unknown"
    if shutil.which("nvidia-smi"):
        gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "python": sys.version.split()[0],
        "vllm": version("vllm"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "gpu": gpu,
        "hardware_profile": hw.get("name"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


class Runner:
    """Sends one request at a time and records it."""

    def __init__(self, llm: Any, sampling_params: Any, rng: random.Random, lo: int, hi: int):
        self.llm, self.sp, self.rng, self.lo, self.hi = llm, sampling_params, rng, lo, hi
        self.records: list[TurnRecord] = []
        self.preemptions = 0

    def tokens(self, n: int) -> list[int]:
        """Fresh random token IDs (distinct content, so no accidental prefix sharing)."""
        return [self.rng.randrange(self.lo, self.hi) for _ in range(n)]

    def send(self, ids: list[int], tag: str) -> TurnRecord:
        from vllm.inputs import TokensPrompt

        t_send = time.perf_counter()
        out = self.llm.generate([TokensPrompt(prompt_token_ids=ids)], self.sp, use_tqdm=False)[0]
        t_done = time.perf_counter()
        stats = getattr(out, "metrics", None)
        self.preemptions += int(getattr(stats, "num_preemptions", 0) or 0)
        r = record_from_request_output(out, t_send, t_done, agent_id=tag)
        self.records.append(r)
        print(f"    {tag:<12} prompt={r.prompt_tokens:>5} cached={r.cached_tokens:>5}", flush=True)
        return r

    def reset_cache(self) -> None:
        if not self.llm.reset_prefix_cache():
            print("    WARNING: reset_prefix_cache() returned False", flush=True)


def check(results: list[dict], name: str, actual: int, expected: int, note: str) -> None:
    ok = actual == expected
    results.append({"check": name, "expected": expected, "actual": actual, "ok": ok,
                    "note": note})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected {expected}, got {actual} ({note})",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/m1_metrics_check/local.yaml")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the block arithmetic and exit without importing vLLM")
    args = parser.parse_args()
    exp = load_experiment(REPO / args.config, repo_root=REPO)
    hw = exp["hardware_cfg"]
    v = hw["vllm"]
    bs, mml, max_tokens = v["block_size"], v["max_model_len"], exp["max_tokens"]
    num_blocks = v.get("num_gpu_blocks_override")
    if num_blocks is None:
        print("num_gpu_blocks_override must be set for this check (fixed block budget).")
        return 2
    c = exp["checks"]
    sp_cfg, ip_cfg, ev_cfg = c["shared_prefix"], c["identical_prompt"], c["eviction"]
    plan = plan_eviction(num_gpu_blocks=num_blocks, block_size=bs, max_model_len=mml,
                         max_tokens=max_tokens,
                         target_prompt_tokens=ev_cfg["target_prompt_tokens"],
                         filler_prompt_tokens=ev_cfg["filler_prompt_tokens"],
                         margin_blocks=ev_cfg["margin_blocks"])
    pre, suf = sp_cfg["prefix_tokens"], sp_cfg["suffix_tokens"]
    n_ident = ip_cfg["prompt_tokens"]
    n_target, n_filler = ev_cfg["target_prompt_tokens"], ev_cfg["filler_prompt_tokens"]
    exp_a = pre - pre % bs
    exp_b = max_cacheable_tokens(n_ident, bs)
    exp_c_warm = max_cacheable_tokens(n_target, bs)

    print("=== Block arithmetic (before loading the model) ===")
    print(f"num_gpu_blocks={num_blocks} -> usable={plan.usable} (1 null block reserved); "
          f"block_size={bs}; max_model_len={mml}; max_tokens={max_tokens}")
    print("Blocks per request (upper bound) = ceil((prompt + max_tokens) / block_size)")
    print(f"(a) A = {pre}+{suf} = {pre + suf} tokens, B = same {pre}-token prefix + different "
          f"{suf}. Expect A.cached=0, B.cached={exp_a} ({exp_a // bs} full shared blocks)")
    print(f"(b) identical {n_ident}-token prompt twice. Expect 2nd cached = "
          f"floor(({n_ident}-1)/{bs})*{bs} = {exp_b} (last-token rule, vllm_internals.md §2)")
    print(f"(c) target A = {n_target} tokens -> {plan.target_blocks} blocks; filler = {n_filler} "
          f"tokens -> ceil(({n_filler}+{max_tokens})/{bs}) = {plan.filler_blocks_each} blocks "
          f"each ({n_filler + max_tokens} <= {mml} = max_model_len)")
    print(f"    need >= usable {plan.usable} + target {plan.target_blocks} + margin "
          f"{ev_cfg['margin_blocks']} = {plan.required_filler_blocks} filler blocks -> "
          f"{plan.num_fillers} fillers x {plan.filler_blocks_each} = "
          f"{plan.total_filler_blocks} blocks > {plan.usable} usable")
    print(f"    sequence: reset, A (expect 0), A again (expect {exp_c_warm}), "
          f"F1..F{plan.num_fillers} (expect 0), A (expect 0)")
    print(f"(d) sum(prompt) vs delta vllm:prefix_cache_queries, sum(cached) vs delta "
          f"vllm:prefix_cache_hits; tolerance {exp['counter_tolerance_tokens']} tokens")
    print(flush=True)
    if args.plan_only:
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(**llm_kwargs(hw, seed=exp["seed"], log_stats=True))
    actual_blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    print(f"[KV cache] vLLM reports num_gpu_blocks={actual_blocks} (planned {num_blocks})")
    if actual_blocks != num_blocks:
        print("Block count differs from the plan; aborting.")
        return 2

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    lo, hi = exp["token_id_range"]
    run = Runner(llm, sampling, random.Random(exp["seed"]), lo, hi)
    results: list[dict] = []
    before = prefix_counters(llm.get_metrics())

    print("\n(a) shared prefix")
    run.reset_cache()
    prefix = run.tokens(pre)
    a = run.send(prefix + run.tokens(suf), "a/A")
    b = run.send(prefix + run.tokens(suf), "a/B")
    check(results, "a: A first send", a.cached_tokens, 0, "fresh content after reset")
    check(results, "a: B shares prefix", b.cached_tokens, exp_a, f"{pre}-token shared prefix")

    print("\n(b) identical prompt")
    run.reset_cache()
    ids = run.tokens(n_ident)
    first = run.send(ids, "b/first")
    second = run.send(ids, "b/second")
    check(results, "b: first send", first.cached_tokens, 0, "fresh content after reset")
    check(results, "b: repeat", second.cached_tokens, exp_b, "floor((N-1)/bs)*bs")

    print("\n(c) LRU eviction")
    run.reset_cache()
    target = run.tokens(n_target)
    t1 = run.send(target, "c/A")
    t2 = run.send(target, "c/A-again")
    fillers_cached = sum(
        run.send(run.tokens(n_filler), f"c/F{i + 1}").cached_tokens
        for i in range(plan.num_fillers)
    )
    t3 = run.send(target, "c/A-final")
    check(results, "c: A first send", t1.cached_tokens, 0, "fresh content after reset")
    check(results, "c: A again (cached)", t2.cached_tokens, exp_c_warm, "before fillers")
    check(results, "c: fillers", fillers_cached, 0, "distinct random content")
    check(results, "c: A after fillers (evicted)", t3.cached_tokens, 0,
          f"{plan.total_filler_blocks} filler blocks > {plan.usable} usable")

    print("\n(d) per-request totals vs vLLM counters")
    delta = prefix_counters(llm.get_metrics()) - before
    sum_prompt = sum(r.prompt_tokens for r in run.records)
    sum_cached = sum(r.cached_tokens for r in run.records)
    tol = exp["counter_tolerance_tokens"]
    for label, ours, theirs in (("queries", sum_prompt, delta.queries),
                                ("hits", sum_cached, delta.hits)):
        gap = theirs - ours
        ok = abs(gap) <= tol
        results.append({"check": f"d: {label}", "expected": ours, "actual": theirs, "ok": ok,
                        "note": f"gap {gap}, tolerance {tol}"})
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: per-request sum={ours}, "
              f"vLLM counter delta={theirs}, gap={gap}")
    if delta.queries != sum_prompt or delta.hits != sum_cached:
        print("  Possible causes of a gap: preempted requests are counted in separate "
              f"preempted_* fields, not these counters (preemptions seen: {run.preemptions}); "
              "num_cached_tokens counts the first scheduling only; requests that skip the cache "
              "lookup are not counted; external/connector hits are separate counters.")

    summary = summarize(run.records, block_size=bs, by_agent=False)
    print(f"\nSummary: turns={summary.num_turns} hit_rate={summary.hit_rate:.3f} "
          f"(vLLM ceiling {summary.max_possible_hit_rate:.3f}); "
          f"mean latency={summary.per_turn_latency.mean * 1000:.1f} ms; "
          f"mean TTFT (vLLM-reported)="
          + (f"{summary.ttft.mean * 1000:.1f} ms" if summary.ttft else "n/a"))
    passed = all(r["ok"] for r in results)
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'} "
          f"({sum(r['ok'] for r in results)}/{len(results)} checks)")

    out_dir = REPO / "results" / "m1_metrics_check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {k: val for k, val in exp.items()},
        "config_path": args.config,
        "provenance": provenance(hw),
        "plan": {**plan.__dict__, "total_filler_blocks": plan.total_filler_blocks},
        "checks": results,
        "counters_delta": delta.__dict__,
        "preemptions": run.preemptions,
        "summary": summary.to_dict(),
        "records": [r.to_dict() for r in run.records],
        "passed": passed,
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved {out_dir.relative_to(REPO)}/result.json")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
