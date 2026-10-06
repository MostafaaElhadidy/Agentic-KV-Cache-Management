"""Compare per-request cached tokens of a sequential vLLM run with the simulator (M4 verify).

    python scripts/crosscheck_sim_vs_gpu.py results/<exp>/<system>/<run>/result.json

Writes crosscheck.json next to the GPU result.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.sim.simulator import SimConfig, simulate_sequential  # noqa: E402
from cachescout.workload.trace import Trace  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1]).resolve()
    gpu = json.loads(path.read_text())
    assert gpu["mode"] == "sequential", "cross-check needs a --mode sequential GPU run"
    cfg = gpu["config"]
    trace = Trace.load(REPO / cfg["trace"])
    params = (cfg.get("cachescout") or {}).get("params") or {}
    sim = simulate_sequential(trace, SimConfig(num_gpu_blocks=gpu["num_gpu_blocks_reported"],
                                               system=gpu["system"], params=params))
    g = {r["request_id"]: r["cached_tokens"] for r in gpu["records"]}
    s = {r["request_id"]: r["cached_tokens"] for r in sim["records"]}
    common = [k for k in s if k in g]
    exact = sum(1 for k in common if g[k] == s[k])
    diff = [abs(g[k] - s[k]) for k in common]
    out = {"gpu_result": str(path.relative_to(REPO)), "system": gpu["system"],
           "requests": len(common), "exact_matches": exact,
           "exact_fraction": exact / len(common) if common else None,
           "mean_abs_diff_tokens": sum(diff) / len(diff) if diff else None,
           "gpu_hit_rate": gpu["summary"]["hit_rate"], "sim_hit_rate": sim["summary"]["hit_rate"]}
    (path.parent / "crosscheck.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
