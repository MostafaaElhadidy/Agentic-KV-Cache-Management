"""Opt-in vLLM generation smoke test. Loads the model; run by hand only, while watching memory.

Usage (from repo root, env active):
    python scripts/smoke_vllm_generate.py --config configs/hardware/local.yaml

Prints GPU memory before/after loading and the number of KV-cache blocks vLLM allocated.
Not part of `pytest`.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cachescout.config import load_yaml  # noqa: E402

MIB = 1024**2


def gpu_mem_mib(device: int = 0) -> tuple[float, float]:
    """Device-wide (free, total) GPU memory in MiB, same quantities as torch.cuda.mem_get_info.

    Engineering choice: read via NVML instead of torch.cuda.mem_get_info, because the latter
    creates a CUDA context in this parent process (~0.3-0.5 GB VRAM held for the whole run) and
    forces vLLM to switch to the 'spawn' start method. NVML reads the same counters without one.
    """
    import pynvml

    pynvml.nvmlInit()
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
        return info.free / MIB, info.total / MIB
    finally:
        pynvml.nvmlShutdown()


def report(label: str) -> None:
    """Print free/used/total GPU memory."""
    free, total = gpu_mem_mib()
    print(f"[GPU {label}] free={free:.0f} MiB used={total - free:.0f} MiB total={total:.0f} MiB",
          flush=True)


def main() -> None:
    """Load the model per the hardware config, generate a few tokens, report memory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hardware/local.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)   # resolves model_file:
    os.environ.update(cfg.get("env", {}))
    v = cfg["vllm"]
    print(f"config={args.config} model={cfg['model']['name']} "
          f"enforce_eager={v['enforce_eager']} "
          f"gpu_memory_utilization={v['gpu_memory_utilization']} "
          f"max_model_len={v['max_model_len']}", flush=True)

    report("before load")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        gpu_memory_utilization=v["gpu_memory_utilization"],
        max_model_len=v["max_model_len"],
        max_num_seqs=v["max_num_seqs"],
        block_size=v["block_size"],
        enable_prefix_caching=v["enable_prefix_caching"],
        enforce_eager=v["enforce_eager"],
        num_gpu_blocks_override=v["num_gpu_blocks_override"],
        seed=0,
    )
    report("after load")

    # The engine core reports num_gpu_blocks back to the frontend (vllm/v1/engine/core_client.py).
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    num_blocks = cache_cfg.num_gpu_blocks or 0
    override = v["num_gpu_blocks_override"]
    usable = max(num_blocks - 1, 0)  # BlockPool permanently reserves one null block
    per_seq = -(-v["max_model_len"] // cache_cfg.block_size)  # ceil(max_model_len / block_size)
    print(f"[KV cache] num_gpu_blocks={num_blocks} (override requested: {override}) "
          f"usable={usable} block_size={cache_cfg.block_size} "
          f"-> {usable * cache_cfg.block_size} tokens; "
          f"one max_model_len sequence needs {per_seq} blocks", flush=True)
    if override is not None:
        status = "OK" if num_blocks == override else "MISMATCH"
        print(f"[KV cache] override check: {status}", flush=True)

    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=16))
    print("GENERATED:", repr(out[0].outputs[0].text), flush=True)
    report("after generate")


if __name__ == "__main__":
    main()
