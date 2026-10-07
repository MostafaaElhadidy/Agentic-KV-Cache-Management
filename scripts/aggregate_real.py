"""Aggregate the REAL multi-agent campaign into report tables + figures. Every value cites its source.

    python scripts/aggregate_real.py                    -> results/report_real/...  (laptop campaign)
    python scripts/aggregate_real.py --exp real_cloud   -> results/report_real_cloud/...  (remote box)

Replay (controlled: identical recorded prompts for both systems) is the headline; live closed-loop
runs are reported separately (outputs diverge between systems, so differences are not attributable to
CacheScout alone).
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
EXP = "real"                      # results/<EXP>/... ; set by --exp
OUT = RES / "report_real"
TOPOS = ("pipeline", "random", "debate", "selector")
SEEDS = (1, 2, 3)
BLOCKS = (100, 150, 200)
SYSTEMS = ("vanilla", "cachescout")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def session_completion(records: list[dict]) -> list[float]:
    """Per-session completion time from first send to last completion (non-warmup records)."""
    first: dict[str, float] = {}
    last: dict[str, float] = defaultdict(float)
    for r in records:
        if r.get("is_warmup") or r.get("session_id") is None:
            continue
        sid = r["session_id"]
        first[sid] = min(first.get(sid, r["t_send"]), r["t_send"])
        last[sid] = max(last[sid], r["t_done"])
    return [last[s] - first[s] for s in first]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    s = r["summary"]
    comp = session_completion(r["records"])
    out = {"hit": s["hit_rate"], "ttft_ms": s["ttft"]["mean"] * 1e3,
           "ttft_p99_ms": s["ttft"]["p99"] * 1e3, "lat_ms": s["per_turn_latency"]["mean"] * 1e3,
           "thr": s["throughput_turns_per_s"], "turns": s["num_turns"],
           "completion_s": statistics.mean(comp) if comp else None,
           "warmups": r.get("warmups_issued"), "git_dirty": r["provenance"]["git_dirty"],
           "src": rel(path)}
    if r.get("agents_summary"):
        a = r["agents_summary"]
        out.update(accuracy=a["gsm8k_accuracy"], final_rate=a["final_answer_rate"],
                   fallback_rate=a["fallback_rate"], strict_next=a.get("strict_next_rate"),
                   trim_rate=a["trim_rate"], R=a["routing_R_measured"], calls=a["calls"],
                   tool_calls=a["tool_calls"], tool_fail=a["tool_parse_failures"],
                   capped=a["capped_sessions"], problem_ids=a["problem_ids"],
                   prompt_mean=a["prompt_tokens_mean"], prompt_max=a["prompt_tokens_max"])
    eng = r.get("engine_runtime") or {}
    out["engine_pred_acc"] = eng.get("prediction_accuracy")
    return out


def replay_path(t: str, seed: int, system: str, b: int) -> Path:
    return RES / EXP / f"replay_gsm8k_test_{t}_s{seed}" / system / f"b{b}_replay" / "result.json"


def live_path(t: str, seed: int, system: str, b: int) -> Path:
    return RES / EXP / f"gsm8k_test_{t}_s{seed}" / system / f"b{b}_eval" / "result.json"


def opt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def ms(xs: list[float]) -> tuple[float, float]:
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def paired(kind: str, t: str, b: int) -> dict[str, Any] | None:
    fn = replay_path if kind == "replay" else live_path
    pairs = [(load(fn(t, s, "vanilla", b)), load(fn(t, s, "cachescout", b))) for s in SEEDS]
    pairs = [(v, c) for v, c in pairs if v and c]
    if not pairs:
        return None
    d = {"n_seeds": len(pairs),
         "vanilla_hit": statistics.mean(v["hit"] for v, _ in pairs),
         "cachescout_hit": statistics.mean(c["hit"] for _, c in pairs)}
    for key, name, pct in (("hit", "dhit_pp", False), ("ttft_ms", "dttft_pct", True),
                           ("lat_ms", "dlat_pct", True), ("completion_s", "dcompletion_pct", True),
                           ("thr", "dthr_pct", True)):
        vals = [((c[key] / v[key] - 1) * 100) if pct else (c[key] - v[key]) * 100
                for v, c in pairs if v[key] and c[key] is not None]
        d[name] = ms(vals) if vals else None
        d[name + "_per_seed"] = vals
    for key in ("ttft_ms", "lat_ms", "completion_s", "thr"):
        d[f"vanilla_{key}"] = statistics.mean(v[key] for v, _ in pairs if v[key] is not None)
        d[f"cachescout_{key}"] = statistics.mean(c[key] for _, c in pairs if c[key] is not None)
    if kind == "live":
        for sysname, idx in (("vanilla", 0), ("cachescout", 1)):
            d[f"{sysname}_accuracy"] = statistics.mean(p[idx]["accuracy"] for p in pairs)
            d[f"{sysname}_calls"] = statistics.mean(p[idx]["calls"] for p in pairs)
    d["engine_pred_acc"] = statistics.mean(c["engine_pred_acc"] for _, c in pairs
                                           if c["engine_pred_acc"] is not None) \
        if any(c["engine_pred_acc"] is not None for _, c in pairs) else None
    d["sources"] = [x["src"] for p in pairs for x in p]
    d["any_git_dirty"] = any(x["git_dirty"] for p in pairs for x in p)
    return d


def fmt(pair: tuple[float, float] | None, unit: str = "") -> str:
    return "n/a" if pair is None else f"{pair[0]:+.1f} ± {pair[1]:.1f}{unit}"


def main() -> int:
    global EXP, OUT
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="real", help="experiment folder under results/ (real_cloud on a box)")
    EXP = ap.parse_args().exp
    OUT = RES / ("report_real" if EXP == "real" else f"report_{EXP}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"workload": {}, "replay": {}, "live": {}}
    md: list[str] = []

    # workload statistics from the vanilla live recordings @100 blocks (3 seeds)
    md += ["## Workload statistics (vanilla live recordings, 100 blocks, mean over 3 seeds)", "",
           "| topology | measured R | LLM calls/run | GSM8K accuracy | FINAL ANSWER rate | selector fallback | "
           "strict NEXT | tool calls/run (failures) | trim rate | prompt tokens mean / max | sources |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in TOPOS:
        runs = [load(live_path(t, s, "vanilla", 100)) for s in SEEDS]
        runs = [r for r in runs if r]
        if not runs:
            continue
        w = {k: statistics.mean(r[k] for r in runs) for k in
             ("R", "calls", "accuracy", "final_rate", "trim_rate", "tool_calls", "tool_fail",
              "prompt_mean")}
        w["prompt_max"] = max(r["prompt_max"] for r in runs)
        w["fallback_rate"] = statistics.mean(r["fallback_rate"] for r in runs) \
            if t == "selector" else None
        w["strict_next"] = statistics.mean(r["strict_next"] for r in runs) \
            if t == "selector" else None
        w["problem_ids"] = {f"seed{s}": r["problem_ids"] for s, r in zip(SEEDS, runs, strict=False)}
        w["sources"] = [r["src"] for r in runs]
        summary["workload"][t] = w
        md.append(f"| {t} | {w['R']:.2f} | {w['calls']:.0f} | {w['accuracy']:.2f} | "
                  f"{w['final_rate']:.2f} | "
                  f"{opt(w['fallback_rate'])} | {opt(w['strict_next'])} | "
                  f"{w['tool_calls']:.1f} ({w['tool_fail']:.1f}) | {w['trim_rate']:.3f} | "
                  f"{w['prompt_mean']:.0f} / {w['prompt_max']} | `{runs[0]['src']}` (+2) |")

    for kind, title in (("replay", "REPLAY (controlled; identical recorded prompts) — headline"),
                        ("live", "LIVE closed-loop (outputs diverge between systems; not "
                                 "attributable to CacheScout alone)")):
        md += ["", f"## {title}", "", "Mean ± std over 3 seeds of CacheScout − vanilla.", "",
               "| topology | blocks | vanilla hit | CacheScout hit | Δ hit (pp) | Δ TTFT | Δ per-turn "
               "latency | Δ session completion | Δ throughput |"
               + (" accuracy vanilla / CacheScout |" if kind == "live" else ""),
               "|---|---|---|---|---|---|---|---|---|" + ("---|" if kind == "live" else "")]
        for t in TOPOS:
            for b in BLOCKS:
                d = paired(kind, t, b)
                if d is None:
                    continue
                summary[kind][f"{t}@{b}"] = d
                line = (f"| {t} | {b} | {d['vanilla_hit'] * 100:.1f}% | "
                        f"{d['cachescout_hit'] * 100:.1f}% | {fmt(d['dhit_pp'])} | "
                        f"{fmt(d['dttft_pct'], '%')} | {fmt(d['dlat_pct'], '%')} | "
                        f"{fmt(d['dcompletion_pct'], '%')} | {fmt(d['dthr_pct'], '%')} |")
                if kind == "live":
                    line += f" {d['vanilla_accuracy']:.2f} / {d['cachescout_accuracy']:.2f} |"
                md.append(line)
        # all-topology means per budget
        md += ["", f"{kind} — mean over all topologies and seeds:", "",
               "| blocks | Δ hit (pp) | Δ TTFT | Δ latency | Δ completion |", "|---|---|---|---|---|"]
        for b in BLOCKS:
            ds = [summary[kind].get(f"{t}@{b}") for t in TOPOS]
            ds = [d for d in ds if d]
            if not ds:
                continue
            agg = {k: statistics.mean(x for d in ds for x in d[k + "_per_seed"])
                   for k in ("dhit_pp", "dttft_pct", "dlat_pct", "dcompletion_pct")}
            summary[kind][f"all@{b}"] = agg
            md.append(f"| {b} | {agg['dhit_pp']:+.1f} | {agg['dttft_pct']:+.1f}% | "
                      f"{agg['dlat_pct']:+.1f}% | {agg['dcompletion_pct']:+.1f}% |")

    syn = RES / "report" / "summary.json"
    if syn.exists():
        s = json.loads(syn.read_text())
        summary["synthetic_reference"] = {f"selector@{b}": s["seeds"][f"cachescout@{b}"]["dhit_mean"]
                                          for b in BLOCKS} | {"src": rel(syn)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    figures(summary)
    print("\n".join(md))
    print(f"\nSaved {rel(OUT)}/{{summary.json, tables.md, fig_real_*.png}}")
    return 0


def figures(summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, bg = "#2b2b29", "#6f6e69", "#e6e5e0", "#fcfcfb"
    colors = {100: "#e87ba4", 150: "#4a3aa7", 200: "#1baf7a"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "text.color": ink, "axes.labelcolor": ink,
                         "axes.spines.top": False, "axes.spines.right": False})
    for kind in ("replay", "live"):
        data = summary[kind]
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 3.5), facecolor=bg)
        ax.set_facecolor(bg)
        ax.grid(True, axis="y", color=grid)
        ax.set_axisbelow(True)
        width = 0.26
        for k, b in enumerate(BLOCKS):
            xs, ys, es = [], [], []
            for i, t in enumerate(TOPOS):
                d = data.get(f"{t}@{b}")
                if d and d["dhit_pp"]:
                    xs.append(i + (k - 1) * width)
                    ys.append(d["dhit_pp"][0])
                    es.append(d["dhit_pp"][1])
            ax.bar(xs, ys, width=width - 0.03, color=colors[b], label=f"{b} blocks", zorder=3,
                   yerr=es, error_kw={"ecolor": muted, "elinewidth": 1, "capsize": 2})
            for x, y in zip(xs, ys, strict=True):
                ax.annotate(f"{y:+.1f}", (x, y), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.5, color=ink)
        ax.axhline(0, color=muted, linewidth=0.8)
        ax.set_xticks(range(len(TOPOS)))
        wl = summary["workload"]
        ax.set_xticklabels([f"{t}\nR={wl[t]['R']:.2f}" if t in wl else t for t in TOPOS])
        ax.set_ylabel("CacheScout − vanilla hit rate (pp)")
        title = ("Real agents, REPLAY (identical prompts), mean ± std of 3 seeds" if kind == "replay"
                 else "Real agents, LIVE closed-loop (outputs diverge), mean ± std of 3 seeds")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_real_{kind}_gain.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
