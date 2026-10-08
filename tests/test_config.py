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


def test_apply_env(monkeypatch) -> None:
    from cachescout.config import apply_env

    monkeypatch.delenv("CS_TEST_VAR", raising=False)
    apply_env({"env": {"CS_TEST_VAR": 1}})
    import os

    assert os.environ["CS_TEST_VAR"] == "1"


def test_llama_profiles_share_one_model_file() -> None:
    """All Llama-3.1-8B hardware profiles read the model name from one file, so switching
    mirror -> meta-llama is a one-line edit (docs/decisions.md, model mirror)."""
    from cachescout.config import load_yaml

    model = load_yaml(REPO / "configs/models/llama31_8b.yaml")
    assert model["name"] in ("unsloth/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct")
    for name in ("cloud", "box_24gb", "box_48gb", "box_80gb"):
        hw = load_yaml(REPO / f"configs/hardware/{name}.yaml")
        assert hw["model"] == model, name
        assert llm_kwargs(hw, seed=0, log_stats=False)["model"] == model["name"]


def test_model_file_conflicts_with_inline_model(tmp_path) -> None:
    import pytest

    from cachescout.config import load_yaml

    (tmp_path / "m.yaml").write_text("name: x\ndtype: bfloat16\n")
    (tmp_path / "hw.yaml").write_text("model_file: m.yaml\nmodel: {name: y}\n")
    with pytest.raises(ValueError):
        load_yaml(tmp_path / "hw.yaml")
