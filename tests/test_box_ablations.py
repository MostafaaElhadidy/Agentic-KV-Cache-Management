"""GPU-free tests for scripts/summarize_box_ablations.py (Fig. 14b gate on/off, Fig. 10b peak
throughput) on fake result.json files."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sba", REPO / "scripts/summarize_box_ablations.py")
sba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sba)


def fake(root: Path, workload: str, label: str, run: str, lat: float, thr: float,
         issued: int = 0, gated: int = 0) -> None:
    d = root / workload / label / run
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "summary": {"hit_rate": 0.5, "ttft": {"mean": lat / 2, "median": lat / 4, "p99": lat},
                    "per_turn_latency": {"mean": lat},
                    "throughput_turns_per_s": thr},
        "warmups_issued": issued, "warmups_gated": gated}))


def test_gate_table_ratio(tmp_path: Path) -> None:
    for seed, (on, off) in {1: (1.0, 1.2), 2: (2.0, 2.4)}.items():
        w = f"replay_gsm8k_test_random_s{seed}"
        fake(tmp_path, w, "cachescout", "b100_replay", on, 1.0, issued=3, gated=8)
        fake(tmp_path, w, "gate_off", "b100_replay", off, 1.0, issued=5)
    rows = sba.gate_rows(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert (r["topology"], r["blocks"], r["seeds"]) == ("random", 100, 2)
    assert abs(r["lat_ratio"] - 1.2) < 1e-9          # mean 1.8 s ungated / 1.5 s gated
    assert r["gated_on"] == 16 and r["gated_off"] == 0 and r["issued_off"] == 10  # summed
    assert "1.20x" in sba.gate_markdown(rows)


def test_gate_rows_skip_incomplete_pairs(tmp_path: Path) -> None:
    fake(tmp_path, "replay_gsm8k_test_debate_s1", "cachescout", "b150_replay", 1.0, 1.0)
    assert sba.gate_rows(tmp_path) == []


def test_sweep_peak(tmp_path: Path) -> None:
    w = "gsm8k_test_selector_s1"
    for rate, (v, c) in {"0.2": (0.5, 0.5), "1": (1.6, 1.9), "4": (1.5, 2.1)}.items():
        fake(tmp_path, w, "vanilla", f"b100_rate{rate}", 1.0, v)
        fake(tmp_path, w, "cachescout", f"b100_rate{rate}", 1.0, c)
    fake(tmp_path, w, "vanilla", "b100_eval", 1.0, 9.9)       # not part of the sweep
    rows = sba.sweep_rows(tmp_path)
    assert [r["rate"] for r in rows if r["system"] == "vanilla"] == [0.2, 1.0, 4.0]
    peaks = sba.peaks(rows)
    assert peaks[("gsm8k_test_selector_s1", 100, "vanilla")] == (1.6, 1.0)
    assert peaks[("gsm8k_test_selector_s1", 100, "cachescout")] == (2.1, 4.0)
    assert "peak" in sba.sweep_markdown(rows).lower()


def test_sweep_reports_ttft_and_every_rate(tmp_path: Path) -> None:
    w = "gsm8k_test_selector_s1"
    for rate in ("0.2", "8"):
        fake(tmp_path, w, "vanilla", f"b100_rate{rate}", 2.0, 1.0)
        fake(tmp_path, w, "cachescout", f"b100_rate{rate}", 2.0, 1.0)   # no gain: still reported
    rows = sba.sweep_rows(tmp_path)
    assert rows[0]["ttft_ms"] == 1000.0 and rows[0]["ttft_median_ms"] == 500.0
    md = sba.sweep_markdown(rows)
    assert "TTFT" in md and "| 8 |" in md and "| 0.2 |" in md
    assert md.count("| gsm8k_test_selector_s1 | 100 |") == 4 + 2  # 4 rate rows + 2 peak rows
