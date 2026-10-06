"""Generate the standard trace set and print/save statistics vs the paper (M2).

    python scripts/make_traces.py --config configs/traces/local.yaml

Writes results/traces/<name>.json and results/traces/stats.json. Tuning and evaluation traces use
different seeds (same agents/anchors): constants are tuned only on the tuning traces.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.config import load_yaml  # noqa: E402
from cachescout.workload.stats import trace_stats  # noqa: E402
from cachescout.workload.topologies import PAPER_R  # noqa: E402
from cachescout.workload.trace import WorkloadProfile, generate_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/traces/local.yaml")
    args = parser.parse_args()
    cfg = load_yaml(REPO / args.config)
    out_dir = REPO / cfg.get("out_dir", "results/traces")
    profile = WorkloadProfile(**cfg.get("profile", {}))
    all_stats = {}
    for spec in cfg["traces"]:
        trace = generate_trace(topology=spec["topology"], num_sessions=spec["num_sessions"],
                               arrival_rate=spec["arrival_rate"], seed=spec["seed"],
                               profile=profile, token_id_range=tuple(cfg["token_id_range"]))
        trace.meta["name"] = spec["name"]
        trace.meta["role"] = spec.get("role", "eval")
        trace.save(out_dir / f"{spec['name']}.json")
        st = trace_stats(trace)
        st["paper_R"] = PAPER_R[spec["topology"]]
        all_stats[spec["name"]] = st
        acc = st["online_accuracy_session"]
        print(f"{spec['name']:<20} turns={st['num_turns']:>4} "
              f"anchor_share={st['anchor_share']:.2f} "
              f"R={st['entropy_reduction_R_true']:.2f} (paper {st['paper_R']:.2f}) "
              f"acc50={acc['acc_first50'] or 0:.2f} max_blocks={st['max_request_blocks']} "
              f"reuse anchor/history={st['mean_reuse_per_block']['anchor']:.1f}/"
              f"{st['mean_reuse_per_block']['history']:.1f}")
    (out_dir / "stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"Saved {out_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
