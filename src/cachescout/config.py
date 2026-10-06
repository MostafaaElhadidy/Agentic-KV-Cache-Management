"""Load experiment + hardware YAML configs and build vLLM `LLM(...)` keyword arguments.

Engineering choice: hardware/model settings live only in configs/hardware/*.yaml; an experiment
config points to one via `hardware:` and may override `num_gpu_blocks_override`.
"""

from pathlib import Path
from typing import Any

import yaml

VLLM_KEYS = (
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "block_size",
    "enable_prefix_caching",
    "enforce_eager",
    "num_gpu_blocks_override",
    "tensor_parallel_size",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def load_experiment(path: str | Path, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Load an experiment config and attach its resolved hardware config under `hardware_cfg`.

    `hardware:` is resolved relative to `repo_root` (default: current directory). An
    experiment-level `num_gpu_blocks_override` replaces the hardware value.
    """
    exp = load_yaml(path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    hw = load_yaml(root / exp["hardware"])
    if "num_gpu_blocks_override" in exp:
        hw.setdefault("vllm", {})["num_gpu_blocks_override"] = exp["num_gpu_blocks_override"]
    exp["hardware_cfg"] = hw
    return exp


def llm_kwargs(hw: dict[str, Any], *, seed: int, log_stats: bool) -> dict[str, Any]:
    """Keyword arguments for `vllm.LLM` from a hardware config.

    `log_stats=True` sets `disable_log_stats=False`, needed for `LLM.get_metrics()` and
    `RequestOutput.metrics` (docs/vllm_internals.md §3-4).
    """
    v = hw["vllm"]
    kwargs: dict[str, Any] = {
        "model": hw["model"]["name"],
        "dtype": hw["model"]["dtype"],
        "seed": seed,
        "disable_log_stats": not log_stats,
    }
    kwargs.update({k: v[k] for k in VLLM_KEYS if k in v})
    return kwargs
