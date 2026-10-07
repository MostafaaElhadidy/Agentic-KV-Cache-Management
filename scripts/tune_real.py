"""Re-tune CacheScout constants in the simulator on recorded REAL sessions (tuning problems only).

    python scripts/tune_real.py      -> results/tuning/real_eviction_sweep.json
    python scripts/tune_real.py --config configs/experiments/real/cloud.yaml   (remote box)

Uses the vanilla recordings of GSM8K TRAIN problems (problem seed 101), one per topology. Evaluation
problems (test split) are never used. Objective: mean hit rate over 100/150/200 blocks. The current
constants (configs/experiments/real/local.yaml) are evaluated too; new constants are recommended
only if they beat the current ones by at least MIN_GAIN.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.config import load_experiment  # noqa: E402
from cachescout.run import load_trace  # noqa: E402
from cachescout.sim.simulator import SimConfig, simulate  # noqa: E402

TOPOS = ("pipeline", "random", "debate", "selector")
BUDGETS = (100, 150, 200)
MIN_GAIN = 0.002   # 0.2 pp mean hit rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/real/local.yaml")
    args = ap.parse_args()
    exp = load_experiment(REPO / args.config, repo_root=REPO)
    current = dict(exp["cachescout"]["params"])
    traces = {}
    for t in TOPOS:
        tr = load_trace(REPO / f"results/{exp['experiment']}/gsm8k_train_{t}_s101/vanilla/b100_rec")
        assert tr.meta["role"] == "tune", f"{t}: tuning must use TRAIN recordings only"
        traces[t] = tr

    def evaluate(system: str, params: dict) -> dict:
        hits = {}
        for name, tr in traces.items():
            for b in BUDGETS:
                out = simulate(tr, SimConfig(num_gpu_blocks=b, system=system, params=params))
                hits[f"{name}@{b}"] = out["summary"]["hit_rate"]
        return {"params": params, "hits": hits, "mean_hit": sum(hits.values()) / len(hits)}

    t0 = time.time()
    vanilla = evaluate("vanilla", {})
    cur = evaluate("eviction_only", current)
    print(f"vanilla {vanilla['mean_hit']:.4f}  current constants {cur['mean_hit']:.4f}")
    rows = []
    for tau, e_max, lam, delta, scope, mapping in itertools.product(
            (0.1, 0.2, 0.3, 0.5), (2, 3, 4, 6), (0.001, 0.005, 0.02, 0.1), (0.01, 0.1),
            ("global", "session"), ("all", "anchor_only")):
        p = {**current, "tau": tau, "e_max": e_max, "lam": lam, "delta": delta, "scope": scope,
             "block_mapping": mapping}
        rows.append(evaluate("eviction_only", p))
    rows.sort(key=lambda r: -r["mean_hit"])
    best = rows[0]
    adopt = best["mean_hit"] - cur["mean_hit"] >= MIN_GAIN
    for r in rows[:6]:
        print(f"{r['mean_hit']:.4f} {r['params']}")
    print(f"best {best['mean_hit']:.4f} vs current {cur['mean_hit']:.4f} -> "
          f"{'ADOPT new constants' if adopt else 'keep current constants'} (min gain {MIN_GAIN})")
    suffix = "" if exp["experiment"] == "real" else f"_{exp['experiment']}"
    out = REPO / "results" / "tuning" / f"real_eviction_sweep{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"vanilla": vanilla, "current": cur, "best": best, "adopt": adopt,
                               "min_gain": MIN_GAIN, "all": rows, "budgets": BUDGETS,
                               "recordings": {t: tr.meta["recorded_from"]
                                              for t, tr in traces.items()}}, indent=2))
    print(f"{len(rows)} configs in {time.time() - t0:.0f}s -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
