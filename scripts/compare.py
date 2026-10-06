"""Run or load vanilla vs CacheScout (and ablations) on the same trace; print a side-by-side table,
check trends against the paper, and save plots + data to results/ (M5).

    CFG=configs/experiments/main/local.yaml
    # cache-size sweep like paper Fig. 14 (runs missing GPU jobs one at a time via gpu_run.sh)
    python scripts/compare.py --config $CFG --blocks 100,150,200 --run
    # just load existing results and print the table
    python scripts/compare.py --config $CFG --blocks 100,150,200
    # another trace / subset of systems / extra variants
    python scripts/compare.py --config $CFG --trace results/traces/debate_eval.json \
        --systems vanilla,cachescout --blocks 100 --run
    # simulator only (no GPU): same table from the pure-Python simulator
    python scripts/compare.py --config $CFG --blocks 100,150,200 --sim

Every number printed comes from a result.json (GPU) or is computed by the simulator (--sim), and the
output JSON lists the source file of each cell.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cachescout.config import load_experiment  # noqa: E402
from cachescout.workload.trace import Trace  # noqa: E402

DEFAULT_SYSTEMS = "vanilla,continuum,warmup_only,eviction_only,cachescout"
# Fixed slot per entity (dataviz rule: color follows the entity), reference palette (light).
COLORS = {"vanilla": "#2a78d6", "continuum": "#eb6834", "warmup_only": "#1baf7a",
          "eviction_only": "#eda100", "cachescout": "#e87ba4", "cachescout_literal": "#008300",
          "no_prediction": "#4a3aa7", "lru_hook": "#e34948"}
MARKERS = {"vanilla": "o", "continuum": "s", "warmup_only": "^", "eviction_only": "D",
           "cachescout": "P", "cachescout_literal": "X", "no_prediction": "v", "lru_hook": "*"}
# Paper Fig. 14a (cache budget sweep), Llama-3.1-8B, for direction/ordering comparison only.
PAPER_FIG14 = {"vanilla": {100: 0.644, 200: 0.766}, "cachescout": {100: 0.87, 200: 0.87}}


def result_path(exp: dict[str, Any], trace_name: str, label: str, blocks: int, tag: str) -> Path:
    return REPO / "results" / exp["experiment"] / trace_name / label / f"b{blocks}_{tag}" / \
        "result.json"


def run_gpu(config: str, trace: str | None, label: str, blocks: int, tag: str,
            variants: set[str], timeout_s: int) -> int:
    sel = ["--variant", label] if label in variants else ["--system", label]
    cmd = [str(REPO / "scripts" / "gpu_run.sh"), f"{Path(config).parent.name}_{label}_b{blocks}"
           f"_{(Path(trace).stem if trace else 'default')}_{tag}", str(timeout_s),
           sys.executable, "-m", "cachescout.run", "--config", config, *sel,
           "--blocks", str(blocks), "--tag", tag]
    if trace:
        cmd += ["--trace", trace]
    print("[compare] running:", " ".join(cmd[3:]), flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode


def row_from_summary(s: dict[str, Any]) -> dict[str, Any]:
    def ms(stats: dict | None, key: str) -> float | None:
        return None if not stats else stats[key] * 1000.0

    return {"hit_rate": s["hit_rate"], "ttft_mean_ms": ms(s["ttft"], "mean"),
            "ttft_median_ms": ms(s["ttft"], "median"), "ttft_p99_ms": ms(s["ttft"], "p99"),
            "latency_mean_ms": ms(s["per_turn_latency"], "mean"),
            "latency_p99_ms": ms(s["per_turn_latency"], "p99"),
            "throughput_tps": s["throughput_turns_per_s"], "turns": s["num_turns"]}


def sim_row(exp: dict[str, Any], trace: Trace, label: str, blocks: int) -> dict[str, Any]:
    from cachescout.sim.simulator import SimConfig, simulate

    params = dict((exp.get("cachescout") or {}).get("params") or {})
    system = label
    if label in (exp.get("variants") or {}):
        spec = exp["variants"][label]
        system = spec["system"]
        params.update(spec.get("params") or {})
    if system == "continuum":
        params["continuum_ttl_s"] = float((exp.get("continuum") or {}).get("ttl_s", 0.3))
    out = simulate(trace, SimConfig(num_gpu_blocks=blocks, system=system, params=params,
                                    warmup_min_interval_s=float((exp.get("warmup") or {})
                                                                .get("min_interval_s", 1.0))))
    row = row_from_summary(out["summary"])
    row["source"] = "simulator"
    return row


def fmt(v: float | None, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def trend_checks(table: dict[int, dict[str, dict]], blocks: list[int]) -> list[dict[str, Any]]:
    """Direction/ordering checks against the paper (absolute values are not expected to match)."""
    checks = []

    def add(name: str, ok: bool | None, detail: str, ref: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "paper": ref})

    def get(b: int, label: str, key: str) -> float | None:
        return (table.get(b, {}).get(label) or {}).get(key)

    have = lambda label: all(get(b, label, "hit_rate") is not None for b in blocks)  # noqa: E731
    if have("cachescout") and have("vanilla"):
        gains = [get(b, "cachescout", "hit_rate") - get(b, "vanilla", "hit_rate") for b in blocks]
        add("CacheScout hit rate > vanilla at every budget", all(g > 0 for g in gains),
            "gains (pp): " + ", ".join(f"{b}:{g * 100:+.1f}" for b, g in zip(blocks, gains,
                                                                             strict=True)),
            "Fig. 8a (+10..18 pp), Fig. 14a")
        if len(blocks) > 1:
            add("gain shrinks as the cache grows", gains[0] > gains[-1],
                f"{blocks[0]}: {gains[0] * 100:+.1f} pp vs {blocks[-1]}: {gains[-1] * 100:+.1f} pp",
                "Fig. 14a (vanilla 64.4%->76.6%, CacheScout flat 86-87%)")
        t = [(get(b, "cachescout", "ttft_mean_ms"), get(b, "vanilla", "ttft_mean_ms"))
             for b in blocks]
        if all(x is not None and y is not None for x, y in t):
            add("CacheScout mean TTFT < vanilla at every budget", all(x < y for x, y in t),
                ", ".join(f"{b}:{(x / y - 1) * 100:+.1f}%" for b, (x, y) in zip(blocks, t,
                                                                               strict=True)),
                "Fig. 8b (-18..45%)")
        lat = [(get(b, "cachescout", "latency_mean_ms"), get(b, "vanilla", "latency_mean_ms"))
               for b in blocks]
        add("CacheScout mean per-turn latency < vanilla at every budget",
            all(x < y for x, y in lat),
            ", ".join(f"{b}:{(x / y - 1) * 100:+.1f}%" for b, (x, y) in zip(blocks, lat,
                                                                           strict=True)),
            "Fig. 10a (-29..38%)")
    if have("eviction_only") and have("warmup_only") and have("vanilla"):
        ev = [get(b, "eviction_only", "hit_rate") - get(b, "vanilla", "hit_rate") for b in blocks]
        wu = [get(b, "warmup_only", "hit_rate") - get(b, "vanilla", "hit_rate") for b in blocks]
        add("eviction-only gain >= warmup-only gain (eviction is the main source)",
            all(e >= w for e, w in zip(ev, wu, strict=True)),
            "evict/warm (pp): " + ", ".join(f"{b}:{e * 100:+.1f}/{w * 100:+.1f}"
                                            for b, e, w in zip(blocks, ev, wu, strict=True)),
            "Sec. 5.3, Fig. 12a")
        add("warmup alone adds <= 1 pp hit rate", all(w <= 0.01 for w in wu),
            ", ".join(f"{b}:{w * 100:+.1f}" for b, w in zip(blocks, wu, strict=True)),
            "Sec. 5.3 ('at most one percentage point')")
    if have("cachescout") and have("continuum"):
        d = [get(b, "cachescout", "hit_rate") - get(b, "continuum", "hit_rate") for b in blocks]
        add("CacheScout hit rate > Continuum at every budget", all(x > 0 for x in d),
            ", ".join(f"{b}:{x * 100:+.1f} pp" for b, x in zip(blocks, d, strict=True)),
            "Fig. 8a")
    return checks


def plot(table: dict[int, dict[str, dict]], blocks: list[int], labels: list[str], title: str,
         path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b29", "#6f6e69", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    metrics = [("hit_rate", "KV-cache hit rate (%)", 100.0),
               ("ttft_mean_ms", "Mean TTFT (ms)", 1.0),
               ("ttft_p99_ms", "P99 TTFT (ms)", 1.0),
               ("latency_mean_ms", "Mean per-turn latency (ms)", 1.0),
               ("throughput_tps", "Throughput (turns/s)", 1.0)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.6),
                             facecolor="#fcfcfb")
    for ax, (key, ylabel, scale) in zip(axes, metrics, strict=True):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for label in labels:
            xs = [b for b in blocks if (table[b].get(label) or {}).get(key) is not None]
            ys = [table[b][label][key] * scale for b in xs]
            if not xs:
                continue
            is_base = label == "vanilla"     # baseline drawn on top, dashed, so ties stay visible
            ax.plot(xs, ys, color=COLORS.get(label, muted), linewidth=2,
                    linestyle=(0, (5, 2)) if is_base else "-",
                    marker=MARKERS.get(label, "o"), markersize=8, markeredgecolor="#fcfcfb",
                    markeredgewidth=1.5, label=label, zorder=5 if is_base else 3)
        if key == "hit_rate":
            for ref, style in (("vanilla", (0, (4, 3))), ("cachescout", (0, (1, 2)))):
                pts = sorted(PAPER_FIG14[ref].items())
                ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], color=muted,
                        linewidth=1.2, linestyle=style, zorder=2)
                ax.annotate(f"paper {ref}", (pts[-1][0], pts[-1][1] * 100), fontsize=7,
                            color=muted, xytext=(4, 0), textcoords="offset points",
                            va="center")
        if key != "hit_rate":
            ax.set_ylim(bottom=0)            # magnitudes: zero baseline, no exaggerated noise
        ax.set_xlabel("GPU cache budget (blocks)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(blocks)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=len(lbls), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.10, fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--systems", default=DEFAULT_SYSTEMS,
                        help="comma list of systems and/or config variants")
    parser.add_argument("--blocks", default="100,150,200")
    parser.add_argument("--trace", default=None, help="trace json (default: config trace)")
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--run", action="store_true", help="run missing GPU results")
    parser.add_argument("--sim", action="store_true", help="use the simulator instead of GPU")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    exp = load_experiment(REPO / args.config, repo_root=REPO)
    trace_file = args.trace or exp["trace"]
    trace = Trace.load(REPO / trace_file)
    trace_name = trace.meta.get("name", Path(trace_file).stem)
    if trace.meta.get("role") == "tune":
        print("WARNING: this is a TUNING trace; do not report it as an evaluation result")
    labels = [s.strip() for s in args.systems.split(",") if s.strip()]
    variants = set((exp.get("variants") or {}).keys())
    blocks = [int(b) for b in args.blocks.split(",")]
    table: dict[int, dict[str, dict]] = {b: {} for b in blocks}
    for b in blocks:
        for label in labels:
            if args.sim:
                table[b][label] = sim_row(exp, trace, label, b)
                continue
            path = result_path(exp, trace_name, label, b, args.tag)
            if not path.exists() and args.run:
                rc = run_gpu(args.config, args.trace, label, b, args.tag, variants, args.timeout)
                if rc != 0:
                    print(f"[compare] GPU run failed (rc={rc}) for {label} @ {b}; stopping.")
                    return rc
            if path.exists():
                res = json.loads(path.read_text())
                row = row_from_summary(res["summary"])
                row["source"] = str(path.relative_to(REPO))
                row["warmups_issued"] = res.get("warmups_issued")
                table[b][label] = row
            else:
                table[b][label] = None
    kind = "SIMULATOR" if args.sim else "GPU (vLLM 0.31.0)"
    print(f"\n=== {kind} | experiment={exp['experiment']} trace={trace_name} tag={args.tag} ===")
    hdr = (f"{'blocks':>6} {'system':<19} {'hit%':>6} {'dHit':>6} {'TTFT':>7} {'TTFTmed':>7} "
           f"{'TTFTp99':>7} {'dTTFT':>7} {'lat':>7} {'dLat':>7} {'thr/s':>6}")
    print(hdr)
    md = ["| blocks | system | hit rate | Δhit vs vanilla | TTFT mean (ms) | TTFT median |"
          " TTFT P99 | Δ TTFT | latency mean (ms) | Δ latency | throughput (turns/s) | source |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in blocks:
        base = table[b].get("vanilla")
        for label in labels:
            r = table[b].get(label)
            if r is None:
                print(f"{b:>6} {label:<19} (missing)")
                continue
            dh = (r["hit_rate"] - base["hit_rate"]) * 100 if base else None
            dt = ((r["ttft_mean_ms"] / base["ttft_mean_ms"] - 1) * 100
                  if base and r["ttft_mean_ms"] and base["ttft_mean_ms"] else None)
            dl = ((r["latency_mean_ms"] / base["latency_mean_ms"] - 1) * 100
                  if base and r["latency_mean_ms"] else None)
            print(f"{b:>6} {label:<19} {r['hit_rate'] * 100:>6.1f} {fmt(dh, '+6.1f'):>6} "
                  f"{fmt(r['ttft_mean_ms'], '7.1f'):>7} {fmt(r['ttft_median_ms'], '7.1f'):>7} "
                  f"{fmt(r['ttft_p99_ms'], '7.1f'):>7} {fmt(dt, '+7.1f'):>7} "
                  f"{fmt(r['latency_mean_ms'], '7.1f'):>7} {fmt(dl, '+7.1f'):>7} "
                  f"{fmt(r['throughput_tps'], '6.2f'):>6}")
            md.append(f"| {b} | {label} | {r['hit_rate'] * 100:.1f}% | {fmt(dh, '+.1f')} pp | "
                      f"{fmt(r['ttft_mean_ms'], '.1f')} | {fmt(r['ttft_median_ms'], '.1f')} | "
                      f"{fmt(r['ttft_p99_ms'], '.1f')} | {fmt(dt, '+.1f')}% | "
                      f"{fmt(r['latency_mean_ms'], '.1f')} | {fmt(dl, '+.1f')}% | "
                      f"{fmt(r['throughput_tps'], '.2f')} | `{r['source']}` |")
    checks = trend_checks(table, blocks)
    print("\nTrend checks vs paper (direction/ordering only):")
    for c in checks:
        mark = "PASS" if c["ok"] else ("FAIL" if c["ok"] is not None else "n/a")
        print(f"  [{mark}] {c['check']}: {c['detail']}   ({c['paper']})")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPO / "results" / exp["experiment"] / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{trace_name}_{'sim' if args.sim else 'gpu'}_{args.tag}_{stamp}"
    (out_dir / f"{base_name}.json").write_text(json.dumps(
        {"kind": kind, "config": args.config, "trace": trace_file, "blocks": blocks,
         "labels": labels, "table": {str(b): table[b] for b in blocks}, "checks": checks},
        indent=2))
    (out_dir / f"{base_name}.md").write_text("\n".join(md) + "\n")
    complete = [lab for lab in labels if all(table[b].get(lab) for b in blocks)]
    if complete:
        plot(table, blocks, complete, f"{kind}: {trace_name}", out_dir / f"{base_name}.png")
        print(f"\nSaved results/{exp['experiment']}/compare/{base_name}.{{json,md,png}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
