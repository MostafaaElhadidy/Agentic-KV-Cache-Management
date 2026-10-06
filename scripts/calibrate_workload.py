"""Calibrate the synthetic workload size using VANILLA vLLM behaviour only (M2/M3).

Policy-independent calibration target (docs/decisions.md): at the paper's cache budgets, vanilla
LRU hit rate should fall in the paper's vanilla range (Fig. 14a: 64.4% at 100 blocks -> 76.6% at
200 blocks), while keeping anchor share 53-62% (Fig. 2), phi at turn 12 within 43-60% (Fig. 3b)
and max request footprint <= ~93 blocks (Sec. 5.4). Uses the TUNING trace seed only; CacheScout is
never run here, so the calibration cannot favour it.

    python scripts/calibrate_workload.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.sim.simulator import SimConfig, simulate  # noqa: E402
from cachescout.workload.stats import trace_stats  # noqa: E402
from cachescout.workload.trace import WorkloadProfile, generate_trace  # noqa: E402


def scaled(scale: float) -> WorkloadProfile:
    base = WorkloadProfile()

    def sc(x: int) -> int:
        return max(1, round(x * scale))

    def sr(r: tuple[int, int]) -> tuple[int, int]:
        return (round(r[0] * scale), max(round(r[1] * scale), round(r[0] * scale)))

    return WorkloadProfile(
        name=f"scaled-{scale:.2f}",
        anchor_tokens={a: sc(v) for a, v in base.anchor_tokens.items()},
        task_tokens=sr(base.task_tokens),
        header_tokens=base.header_tokens,
        output_tokens={a: sr(v) for a, v in base.output_tokens.items()},
        tool_tokens={a: sr(v) for a, v in base.tool_tokens.items()},
        calls_per_invocation=base.calls_per_invocation,
        turns=base.turns,
        max_prompt_tokens=base.max_prompt_tokens,
        think_s=base.think_s,
        tool_s=base.tool_s,
    )


def main() -> int:
    rows = []
    for scale in (1.0, 0.8, 0.65, 0.5, 0.4, 0.33):
        prof = scaled(scale)
        trace = generate_trace(topology="selector", num_sessions=60, arrival_rate=0.5, seed=101,
                               profile=prof)
        st = trace_stats(trace)
        hits = {}
        for blocks in (100, 150, 200):
            out = simulate(trace, SimConfig(num_gpu_blocks=blocks, system="vanilla"))
            hits[blocks] = out["summary"]["hit_rate"]
        phi12 = st["phi_by_turn"].get(12)
        row = {"scale": scale, "vanilla_hit": hits, "anchor_share": st["anchor_share"],
               "phi_turn12": phi12, "max_blocks": st["max_request_blocks"],
               "mean_prompt": st["mean_prompt_tokens"], "profile": asdict(prof)}
        rows.append(row)
        print(f"scale={scale:.2f} vanilla hit 100/150/200 = {hits[100]:.3f}/{hits[150]:.3f}/"
              f"{hits[200]:.3f}  anchor_share={st['anchor_share']:.2f} "
              f"phi@12={phi12 if phi12 is None else round(phi12, 2)} "
              f"max_blocks={st['max_request_blocks']} mean_prompt={st['mean_prompt_tokens']:.0f}")
    out_dir = REPO / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "workload_calibration.json").write_text(json.dumps(rows, indent=2))
    print("paper vanilla target: 0.644 @100 -> 0.766 @200 (Fig. 14a)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
