"""Opt-in vLLM generation smoke test. Loads the model; run by hand only, while watching memory.

Usage (from repo root, env active):
    python scripts/smoke_vllm_generate.py --config configs/hardware/local.yaml

Not part of `pytest`. Not yet run (see docs/PLAN.md M1).
"""

import argparse
import os

import yaml


def main() -> None:
    """Generate a few tokens and print KV-cache info."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hardware/local.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.environ.update(cfg.get("env", {}))

    from vllm import LLM, SamplingParams

    v = cfg["vllm"]
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
    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0, max_tokens=16))
    print("GENERATED:", repr(out[0].outputs[0].text))


if __name__ == "__main__":
    main()
