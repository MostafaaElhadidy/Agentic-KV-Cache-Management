"""Tune CacheScout's unspecified constants in the simulator on TUNING traces only (M3).

The paper gives no values for epsilon, tau, E_max, lambda, delta, R_min (open_questions B1).
Objective: mean hit rate over cache budgets {100, 150, 200} on the tuning traces
(selector_tune, debate_tune). Evaluation traces are never touched here.

    python scripts/tune_constants.py            # eviction constants (system=eviction_only)
    python scripts/tune_constants.py --warmup   # then R_min / warmup interval (system=cachescout)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.sim.simulator import SimConfig, simulate  # noqa: E402
from cachescout.workload.trace import Trace  # noqa: E402

TUNE_TRACES = ("selector_tune", "debate_tune")
BUDGETS = (100, 150, 200)


def evaluate(traces: dict[str, Trace], system: str, params: dict, **cfg) -> dict:
    hits = {}
    for name, tr in traces.items():
        for b in BUDGETS:
            out = simulate(tr, SimConfig(num_gpu_blocks=b, system=system, params=params, **cfg))
            hits[f"{name}@{b}"] = out["summary"]["hit_rate"]
    return {"params": params, "cfg": cfg, "hits": hits,
            "mean_hit": sum(hits.values()) / len(hits)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    traces = {n: Trace.load(REPO / "results" / "traces" / f"{n}.json") for n in TUNE_TRACES}
    for tr in traces.values():
        assert tr.meta.get("role") == "tune", "tuning must only use tuning traces"
    out_dir = REPO / "results" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    baseline = evaluate(traces, "vanilla", {})
    print(f"vanilla mean hit = {baseline['mean_hit']:.4f}")
    rows = []
    if not args.warmup:
        scopes = [("global", 8), ("session", 4), ("session", 8), ("session", 16)]
        grid = itertools.product((0.1, 0.2, 0.3, 0.5), (2, 3, 4, 6), (0.001, 0.005, 0.02, 0.1),
                                 (0.01, 0.1), scopes)
        for tau, e_max, lam, delta, (scope, active) in grid:
            p = {"tau": tau, "e_max": e_max, "lam": lam, "delta": delta, "scope": scope,
                 "max_active_sessions": active, "session_aggregate": "mean",
                 "epsilon": 0.01, "fingerprint_blocks": 2}
            rows.append(evaluate(traces, "eviction_only", p))
        name = "eviction_sweep.json"
    else:
        best = json.loads((out_dir / "eviction_sweep.json").read_text())["best"]["params"]
        for r_min, interval in itertools.product((0.0, 0.2, 0.3, 0.4, 0.5, 0.7), (0.5, 1.0, 2.0)):
            p = {**best, "r_min": r_min}
            rows.append(evaluate(traces, "cachescout", p, warmup_min_interval_s=interval))
        name = "warmup_sweep.json"
    rows.sort(key=lambda r: -r["mean_hit"])
    for r in rows[:8]:
        print(f"{r['mean_hit']:.4f}  {r['params']}  {r['cfg']}")
    (out_dir / name).write_text(json.dumps({"baseline_vanilla": baseline, "best": rows[0],
                                            "all": rows, "tune_traces": TUNE_TRACES,
                                            "budgets": BUDGETS}, indent=2))
    print(f"{len(rows)} configs in {time.time() - t0:.0f}s -> results/tuning/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
