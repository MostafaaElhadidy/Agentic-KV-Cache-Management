"""Summarise the two box-only ablations from saved result.json files (no GPU, no model).

    python scripts/summarize_box_ablations.py --exp real_cloud              # print both tables
    python scripts/summarize_box_ablations.py --exp real_cloud \
        --out results/report_real_cloud/ablations.md

1. Prefetch gate on/off (paper Fig. 14b, Sec. 5.4: "disabling the gate increases per-turn latency by
   up to 22%"). Pairs `cachescout` with the `gate_off` variant (r_min = 0) on the same replay
   recording and budget (stage `gate_ablation` of run_real_campaign.sh). Ratio = mean per-turn
   latency ungated / gated, averaged over seeds first (> 1 means the gate helps).
2. Peak throughput (paper Fig. 10b): live runs at several arrival rates (stage `rate_sweep`, tags
   `rate<r>`); peak = the maximum completed turns/s over the rates, per system.

Per-turn latency here is per LLM call (docs/decisions.md, "Per-turn latency"). Every number comes
from a result.json under results/<exp>/; nothing is computed from the paper.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REPLAY_RE = re.compile(r"^replay_gsm8k_(?P<split>\w+?)_(?P<topo>[a-z]+)_s(?P<seed>\d+)$")
RUN_RE = re.compile(r"^b(?P<blocks>\d+)_(?P<tag>.+)$")


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def gate_rows(root: Path, tag: str = "replay") -> list[dict[str, Any]]:
    """One row per (topology, blocks) with both cachescout and gate_off on the same seeds."""
    pairs: dict[tuple[str, int], list[tuple[dict, dict]]] = defaultdict(list)
    for wdir in sorted(root.glob("replay_gsm8k_*")):
        m = REPLAY_RE.match(wdir.name)
        if not m:
            continue
        for on_path in sorted((wdir / "cachescout").glob(f"b*_{tag}/result.json")):
            run = on_path.parent.name
            off_path = wdir / "gate_off" / run / "result.json"
            if not off_path.exists():
                continue
            blocks = int(RUN_RE.match(run)["blocks"])
            pairs[(m["topo"], blocks)].append((_load(on_path), _load(off_path)))
    rows = []
    for (topo, blocks), runs in sorted(pairs.items()):
        lat_on = _mean([on["summary"]["per_turn_latency"]["mean"] for on, _ in runs])
        lat_off = _mean([off["summary"]["per_turn_latency"]["mean"] for _, off in runs])
        rows.append({
            "topology": topo, "blocks": blocks, "seeds": len(runs),
            "lat_on_ms": 1000 * lat_on, "lat_off_ms": 1000 * lat_off, "lat_ratio": lat_off / lat_on,
            "hit_on": _mean([on["summary"]["hit_rate"] for on, _ in runs]),
            "hit_off": _mean([off["summary"]["hit_rate"] for _, off in runs]),
            "issued_on": sum(on.get("warmups_issued", 0) for on, _ in runs),
            "issued_off": sum(off.get("warmups_issued", 0) for _, off in runs),
            "gated_on": sum(on.get("warmups_gated", 0) for on, _ in runs),
            "gated_off": sum(off.get("warmups_gated", 0) for _, off in runs),
        })
    return rows


def gate_markdown(rows: list[dict[str, Any]]) -> str:
    out = ["## Prefetch gate on/off (paper Fig. 14b, Sec. 5.4: ungated up to +22% latency)",
           "",
           "| topology | blocks | seeds | per-turn latency gated (ms) | ungated (ms) "
           "| ungated / gated | hit gated | hit ungated | warmups issued gated / ungated "
           "| gated decisions |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['topology']} | {r['blocks']} | {r['seeds']} | {r['lat_on_ms']:.1f} | "
                   f"{r['lat_off_ms']:.1f} | {r['lat_ratio']:.2f}x | {100 * r['hit_on']:.1f}% | "
                   f"{100 * r['hit_off']:.1f}% | {r['issued_on']} / {r['issued_off']} | "
                   f"{r['gated_on']} |")
    if not rows:
        out.append("| (no complete cachescout / gate_off pairs found) |||||||||| ")
    return "\n".join(out)


def sweep_rows(root: Path) -> list[dict[str, Any]]:
    """Live runs tagged `rate<r>` (any workload, vanilla and cachescout)."""
    rows = []
    for path in sorted(root.glob("gsm8k_*/*/b*_rate*/result.json")):
        m = RUN_RE.match(path.parent.name)
        try:
            rate = float(m["tag"].removeprefix("rate"))
        except ValueError:
            continue
        s = _load(path)["summary"]
        rows.append({"workload": path.parents[2].name, "system": path.parents[1].name,
                     "blocks": int(m["blocks"]), "rate": rate,
                     "throughput": s["throughput_turns_per_s"],
                     "lat_ms": 1000 * s["per_turn_latency"]["mean"], "hit": s["hit_rate"]})
    rows.sort(key=lambda r: (r["workload"], r["blocks"], r["system"], r["rate"]))
    return rows


def peaks(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], tuple[float, float]]:
    """(workload, blocks, system) -> (peak turns/s, arrival rate where it occurred)."""
    best: dict[tuple[str, int, str], tuple[float, float]] = {}
    for r in rows:
        k = (r["workload"], r["blocks"], r["system"])
        if r["throughput"] is not None and (k not in best or r["throughput"] > best[k][0]):
            best[k] = (r["throughput"], r["rate"])
    return best


def sweep_markdown(rows: list[dict[str, Any]]) -> str:
    out = ["## Arrival-rate sweep and peak throughput (paper Fig. 10b)", "",
           "| workload | blocks | system | arrival rate (sessions/s) | turns/s "
           "| per-turn latency (ms) | hit rate |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        thr = "n/a" if r["throughput"] is None else f"{r['throughput']:.3f}"
        out.append(f"| {r['workload']} | {r['blocks']} | {r['system']} | {r['rate']:g} | {thr} | "
                   f"{r['lat_ms']:.1f} | {100 * r['hit']:.1f}% |")
    out += ["", "| workload | blocks | system | peak turns/s | at rate |", "|---|---|---|---|---|"]
    for (w, b, sysname), (thr, rate) in sorted(peaks(rows).items()):
        out.append(f"| {w} | {b} | {sysname} | {thr:.3f} | {rate:g} |")
    if not rows:
        out.append("| (no rate<r> runs found) |||||")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="real_cloud", help="experiment name under results/")
    ap.add_argument("--results-root", default=None, help="override results/<exp> (tests)")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args()
    root = Path(args.results_root) if args.results_root else REPO / "results" / args.exp
    if not root.is_dir():
        print(f"no results directory {root}", file=sys.stderr)
        return 2
    text = gate_markdown(gate_rows(root)) + "\n\n" + sweep_markdown(sweep_rows(root)) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
