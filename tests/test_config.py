"""Tests for config loading (no GPU)."""

from pathlib import Path

from cachescout.config import llm_kwargs, load_experiment

REPO = Path(__file__).resolve().parents[1]


def test_local_metrics_check_config_resolves() -> None:
    exp = load_experiment(REPO / "configs/experiments/m1_metrics_check/local.yaml", repo_root=REPO)
    hw = exp["hardware_cfg"]
    assert hw["vllm"]["num_gpu_blocks_override"] == 256
    assert hw["vllm"]["enforce_eager"] is True
    kw = llm_kwargs(hw, seed=0, log_stats=True)
    assert kw["disable_log_stats"] is False
    assert kw["max_model_len"] == 2048
    assert kw["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_experiment_override_of_block_budget() -> None:
    exp = load_experiment(REPO / "configs/experiments/m1_metrics_check/cloud.yaml", repo_root=REPO)
    assert exp["hardware_cfg"]["vllm"]["num_gpu_blocks_override"] == 256
    assert exp["hardware_cfg"]["vllm"]["enforce_eager"] is False
