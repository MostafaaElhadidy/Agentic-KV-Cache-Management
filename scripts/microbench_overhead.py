"""Runtime-overhead microbenchmark (paper Sec. 5.5, Fig. 15): state size and hot-path latency
for 6, 12 and 24 agents. Pure CPU; run when no GPU job is running.

    python scripts/microbench_overhead.py   -> results/microbench/overhead.json
"""

import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.core.runtime import CacheScoutParams, CacheScoutRuntime, Candidate  # noqa: E402

PARAMS = {"scope": "session", "max_active_sessions": 8, "session_aggregate": "mean",
          "block_mapping": "anchor_only", "tau": 0.1, "e_max": 6, "lam": 0.005, "delta": 0.01}


def bench(num_agents: int, dispatches: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rt = CacheScoutRuntime(CacheScoutParams(**PARAMS))
    # sparse random transition structure: each agent has 3 likely successors
    succ = {a: rng.sample([b for b in range(num_agents) if b != a], 3) for a in range(num_agents)}
    observe_ns, select_ns, sizes = [], [], {}
    cur = {f"s{i}": 0 for i in range(8)}
    cands = [Candidate(b, True, 16) for b in range(255)]
    for b in range(255):
        rt.on_blocks_used([b], rng.randrange(num_agents), f"s{b % 8}", new=True)
        rt.on_blocks_used([b], None, f"s{(b + 1) % 8}")
    for n in range(1, dispatches + 1):
        sid = f"s{rng.randrange(8)}"
        nxt = rng.choice(succ[cur[sid]]) if rng.random() < 0.8 else rng.randrange(num_agents)
        cur[sid] = nxt
        t0 = time.perf_counter_ns()
        rt.observe_dispatch(f"r{n}", f"fp{nxt}", sid)
        observe_ns.append(time.perf_counter_ns() - t0)
        rt.on_step()
        t0 = time.perf_counter_ns()
        rt.select_victims(cands, 4)
        select_ns.append(time.perf_counter_ns() - t0)
        if n in (10, 100, 1000, 10000):
            sizes[n] = rt.learner.state_size_bytes()

    def us(xs: list[int], q: float) -> float:
        return statistics.quantiles(xs, n=100)[int(q) - 1] / 1e3

    return {"num_agents": num_agents, "dispatches": dispatches,
            "state_bytes_by_dispatches": sizes,
            "observe_us": {"mean": statistics.mean(observe_ns) / 1e3, "p50": us(observe_ns, 50),
                           "p99": us(observe_ns, 99)},
            "select_255_candidates_us": {"mean": statistics.mean(select_ns) / 1e3,
                                         "p50": us(select_ns, 50), "p99": us(select_ns, 99)}}


def main() -> int:
    out = [bench(a, 10000) for a in (6, 12, 24)]
    for r in out:
        print(f"agents={r['num_agents']:>2} state={r['state_bytes_by_dispatches']} "
              f"observe mean/p99={r['observe_us']['mean']:.1f}/{r['observe_us']['p99']:.1f} us "
              f"select(255) mean/p99={r['select_255_candidates_us']['mean']:.1f}/"
              f"{r['select_255_candidates_us']['p99']:.1f} us")
    d = REPO / "results" / "microbench"
    d.mkdir(parents=True, exist_ok=True)
    (d / "overhead.json").write_text(json.dumps({"python": sys.version.split()[0],
                                                 "params": PARAMS, "runs": out}, indent=2))
    print("paper Fig. 15: state < 25 KB @24 agents; ObserveTouch ~1 us; PredictSurvival <= 6 us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
