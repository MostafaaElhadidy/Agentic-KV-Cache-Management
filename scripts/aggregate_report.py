"""Aggregate all local results into report tables + figures (M5). Every value carries its source path.

    python scripts/aggregate_report.py   -> results/report/{summary.json, tables.md, fig_*.png}
"""

import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = RES / "report"
BLOCKS = (100, 150, 200)
SEED_TRACES = ("selector_eval", "selector_eval_s2", "selector_eval_s3")


def load(trace: str, label: str, blocks: int, exp: str = "main", tag: str = "eval") -> dict | None:
    p = RES / exp / trace / label / f"b{blocks}_{tag}" / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    s = r["summary"]
    return {"hit": s["hit_rate"], "ttft_ms": s["ttft"]["mean"] * 1e3,
            "ttft_med_ms": s["ttft"]["median"] * 1e3, "ttft_p99_ms": s["ttft"]["p99"] * 1e3,
            "lat_ms": s["per_turn_latency"]["mean"] * 1e3,
            "thr": s["throughput_turns_per_s"], "turns": s["num_turns"],
            "warmups": r.get("warmups_issued"), "engine": r.get("engine_runtime"),
            "src": str(p.relative_to(REPO))}


def rel(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    md: list[str] = []

    # 1. main table (seed 1)
    systems = ["vanilla", "continuum", "warmup_only", "eviction_only", "cachescout",
               "cachescout_literal", "no_prediction"]
    main_rows = {}
    md += ["## Main sweep (selector_eval, seed 1, GPU)", "",
           "| blocks | system | hit rate | Δ hit (pp) | TTFT mean (ms) | Δ TTFT | TTFT P99 (ms) | "
           "latency mean (ms) | Δ latency | throughput (turns/s) | source |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in BLOCKS:
        base = load("selector_eval", "vanilla", b)
        for s in systems:
            r = load("selector_eval", s, b)
            if r is None:
                continue
            main_rows[f"{s}@{b}"] = r
            md.append(f"| {b} | {s} | {r['hit'] * 100:.1f}% | {(r['hit'] - base['hit']) * 100:+.1f} | "
                      f"{r['ttft_ms']:.1f} | {rel(r['ttft_ms'], base['ttft_ms']):+.1f}% | "
                      f"{r['ttft_p99_ms']:.1f} | {r['lat_ms']:.1f} | "
                      f"{rel(r['lat_ms'], base['lat_ms']):+.1f}% | {r['thr']:.2f} | `{r['src']}` |")
    summary["main"] = {k: {kk: vv for kk, vv in v.items() if kk != "engine"}
                       for k, v in main_rows.items()}

    # 2. seed-averaged gains (3 evaluation seeds)
    md += ["", "## CacheScout vs vanilla, mean ± std over 3 evaluation seeds (GPU)", "",
           "| blocks | system | vanilla hit | system hit | Δ hit (pp) | Δ TTFT mean | Δ latency mean | "
           "per-seed Δ hit (pp) |", "|---|---|---|---|---|---|---|---|"]
    seeds: dict[str, Any] = {}
    for s in ("eviction_only", "cachescout"):
        for b in BLOCKS:
            pairs = [(load(t, "vanilla", b), load(t, s, b)) for t in SEED_TRACES]
            pairs = [(v, c) for v, c in pairs if v and c]
            if len(pairs) < 2:
                continue
            dh = [(c["hit"] - v["hit"]) * 100 for v, c in pairs]
            dt = [rel(c["ttft_ms"], v["ttft_ms"]) for v, c in pairs]
            dl = [rel(c["lat_ms"], v["lat_ms"]) for v, c in pairs]
            vh = statistics.mean(v["hit"] for v, _ in pairs) * 100
            ch = statistics.mean(c["hit"] for _, c in pairs) * 100
            seeds[f"{s}@{b}"] = {"n": len(pairs), "vanilla_hit": vh, "hit": ch,
                                 "dhit_mean": statistics.mean(dh), "dhit_std": statistics.stdev(dh),
                                 "dttft_mean": statistics.mean(dt), "dttft_std": statistics.stdev(dt),
                                 "dlat_mean": statistics.mean(dl), "dlat_std": statistics.stdev(dl),
                                 "per_seed_dhit": dh,
                                 "sources": [p[1]["src"] for p in pairs] + [p[0]["src"] for p in pairs]}
            x = seeds[f"{s}@{b}"]
            md.append(f"| {b} | {s} | {vh:.1f}% | {ch:.1f}% | {x['dhit_mean']:+.1f} ± {x['dhit_std']:.1f} | "
                      f"{x['dttft_mean']:+.1f} ± {x['dttft_std']:.1f}% | "
                      f"{x['dlat_mean']:+.1f} ± {x['dlat_std']:.1f}% | "
                      f"{', '.join(f'{d:+.1f}' for d in dh)} |")
    summary["seeds"] = seeds

    # 2b. absolute means over the 3 evaluation seeds (README results table)
    readme: dict[str, Any] = {}
    for s in ("vanilla", "cachescout"):
        for b in BLOCKS:
            runs = [load(t, s, b) for t in SEED_TRACES]
            runs = [r for r in runs if r]
            if len(runs) < 3:
                continue
            readme[f"{s}@{b}"] = {
                "n_seeds": len(runs),
                "hit_mean": statistics.mean(r["hit"] for r in runs),
                "ttft_ms_mean": statistics.mean(r["ttft_ms"] for r in runs),
                "lat_ms_mean": statistics.mean(r["lat_ms"] for r in runs),
                "thr_mean": statistics.mean(r["thr"] for r in runs),
                "sources": [r["src"] for r in runs]}
    summary["readme_table"] = readme

    # 3. topologies
    md += ["", "## Coordination topologies (GPU, CacheScout vs vanilla)", "",
           "| trace | true R (trace) | paper R | blocks | vanilla hit | CacheScout hit | Δ hit (pp) | "
           "Δ TTFT | source (CacheScout) |", "|---|---|---|---|---|---|---|---|---|"]
    stats = json.loads((RES / "traces" / "stats.json").read_text())
    topo = {}
    for t in ("pipeline_eval", "debate_eval", "selector_eval", "random_eval"):
        for b in (100, 150):
            v, c = load(t, "vanilla", b), load(t, "cachescout", b)
            if not (v and c):
                continue
            topo[f"{t}@{b}"] = {"vanilla": v["hit"], "cachescout": c["hit"],
                                "dhit": (c["hit"] - v["hit"]) * 100,
                                "dttft": rel(c["ttft_ms"], v["ttft_ms"]),
                                "R_true": stats[t]["entropy_reduction_R_true"],
                                "R_paper": stats[t]["paper_R"], "src": [v["src"], c["src"]]}
            x = topo[f"{t}@{b}"]
            md.append(f"| {t} | {x['R_true']:.2f} | {x['R_paper']:.2f} | {b} | {v['hit'] * 100:.1f}% | "
                      f"{c['hit'] * 100:.1f}% | {x['dhit']:+.1f} | {x['dttft']:+.1f}% | `{c['src']}` |")
    summary["topologies"] = topo

    # 4. load sweep
    md += ["", "## Load sweep at 150 blocks (GPU)", "",
           "| arrival rate (sessions/s) | vanilla hit | CacheScout hit | vanilla TTFT (ms) | "
           "CacheScout TTFT (ms) | vanilla thr | CacheScout thr | source (CacheScout) |",
           "|---|---|---|---|---|---|---|---|"]
    load_rows = {}
    for t, rate in (("selector_eval", 0.5), ("selector_eval_r1", 1.0), ("selector_eval_r2", 2.0),
                    ("selector_eval_r4", 4.0)):
        v, c = load(t, "vanilla", 150), load(t, "cachescout", 150)
        if v and c:
            load_rows[str(rate)] = {"vanilla": {k: v[k] for k in ("hit", "ttft_ms", "lat_ms", "thr")},
                                    "cachescout": {k: c[k] for k in ("hit", "ttft_ms", "lat_ms", "thr")},
                                    "src": [v["src"], c["src"]]}
            md.append(f"| {rate} | {v['hit'] * 100:.1f}% | {c['hit'] * 100:.1f}% | {v['ttft_ms']:.0f} | "
                      f"{c['ttft_ms']:.0f} | {v['thr']:.2f} | {c['thr']:.2f} | `{c['src']}` |")
    summary["load"] = load_rows

    # 5. M4 cross-checks
    md += ["", "## Hook verification: sequential GPU vs simulator (tuning trace, 100 blocks)", "",
           "| system | requests | exact matches | GPU hit | simulator hit | source |",
           "|---|---|---|---|---|---|"]
    xc = {}
    for s in ("vanilla", "lru_hook", "eviction_only"):
        p = RES / "tune_gpu" / s / "b100_seq" / "crosscheck.json"
        if p.exists():
            x = json.loads(p.read_text())
            xc[s] = {**x, "src": str(p.relative_to(REPO))}
            md.append(f"| {s} | {x['requests']} | {x['exact_matches']} ({x['exact_fraction'] * 100:.1f}%) | "
                      f"{x['gpu_hit_rate'] * 100:.2f}% | {x['sim_hit_rate'] * 100:.2f}% | "
                      f"`{xc[s]['src']}` |")
    summary["crosscheck"] = xc

    # 6. engine-side learner stats + overhead
    eng = {}
    for b in BLOCKS:
        r = load("selector_eval", "cachescout", b)
        if r and r["engine"]:
            e = r["engine"]
            eng[str(b)] = {k: e.get(k) for k in ("prediction_accuracy", "entropy_reduction_R",
                                                 "dispatches", "warmups_seen", "observe_us_mean",
                                                 "select_us_mean", "learner_state_bytes")}
            eng[str(b)]["warmups_issued"] = r["warmups"]
            eng[str(b)]["src"] = r["src"]
    summary["engine"] = eng
    mb = RES / "microbench" / "overhead.json"
    if mb.exists():
        summary["microbench"] = {**json.loads(mb.read_text()), "src": str(mb.relative_to(REPO))}

    # 7. trace statistics vs paper motivation numbers
    summary["trace_stats"] = {t: {k: stats[t][k] for k in
                                  ("anchor_share", "entropy_reduction_R_true", "paper_R",
                                   "max_request_blocks", "mean_reuse_per_block",
                                   "online_accuracy_session", "online_accuracy_global")}
                              for t in ("selector_eval", "pipeline_eval", "debate_eval",
                                        "random_eval")}
    summary["trace_stats"]["src"] = "results/traces/stats.json"

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    figures(seeds, topo)
    print("\n".join(md))
    print("\nSaved results/report/{summary.json, tables.md, fig_cache_sweep.png, fig_topologies.png}")
    return 0


def figures(seeds: dict[str, Any], topo: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, bg = "#2b2b29", "#6f6e69", "#e6e5e0", "#fcfcfb"
    colors = {"vanilla": "#2a78d6", "eviction_only": "#eda100", "cachescout": "#e87ba4"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "text.color": ink, "axes.labelcolor": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    # Figure A: seed-averaged hit rate vs budget (cf. paper Fig. 14a)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.grid(True, axis="y", color=grid)
    ax.set_axisbelow(True)
    van = [seeds[f"cachescout@{b}"]["vanilla_hit"] for b in BLOCKS]
    lines = {"vanilla": van,
             "eviction_only": [seeds[f"eviction_only@{b}"]["hit"] for b in BLOCKS],
             "cachescout": [seeds[f"cachescout@{b}"]["hit"] for b in BLOCKS]}
    for (name, ys), m in zip(lines.items(), ("o", "D", "P"), strict=True):
        ax.plot(BLOCKS, ys, color=colors[name], linewidth=2, marker=m, markersize=8,
                markeredgecolor=bg, markeredgewidth=1.5, label=f"ours: {name}",
                linestyle=(0, (5, 2)) if name == "vanilla" else "-", zorder=4)
    ax.plot([100, 200], [64.4, 76.6], color=muted, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.plot([100, 200], [87, 87], color=muted, linestyle=(0, (1, 2)), linewidth=1.2)
    ax.annotate("paper vanilla", (200, 76.6), xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=muted, va="center")
    ax.annotate("paper CacheScout", (200, 87), xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=muted, va="center")
    ax.set_xticks(BLOCKS)
    ax.set_xlabel("GPU cache budget (blocks)")
    ax.set_ylabel("KV-cache hit rate (%), mean of 3 seeds")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Cache-size sweep (Selector, GPU) vs paper Fig. 14a", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cache_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Figure B: hit-rate gain per topology (cf. paper Fig. 4a / Tab. 1)
    order = ["pipeline_eval", "debate_eval", "selector_eval", "random_eval"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=bg)
    ax.set_facecolor(bg)
    ax.grid(True, axis="y", color=grid)
    ax.set_axisbelow(True)
    width = 0.36
    for k, (b, col) in enumerate(((100, "#e87ba4"), (150, "#4a3aa7"))):
        xs = [i + (k - 0.5) * width for i in range(len(order))]
        ys = [topo.get(f"{t}@{b}", {}).get("dhit", 0.0) for t in order]
        ax.bar(xs, ys, width=width - 0.04, color=col, label=f"{b} blocks", zorder=3)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{y:+.1f}", (x, y), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=ink)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{t.split('_')[0]}\nR={topo.get(t + '@100', {}).get('R_true', 0):.2f}"
                        for t in order])
    ax.axhline(0, color=muted, linewidth=0.8)
    ax.set_ylabel("CacheScout − vanilla hit rate (pp)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Gain by coordination topology (GPU, seed 1)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_topologies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
