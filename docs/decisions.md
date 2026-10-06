# Decisions log

Format: date, decision, why, alternatives. Label: **paper** / **interpretation** / **engineering choice**.

## 2026-10-06: Reuse the existing vLLM environment
- **Engineering choice.** Use `~/testLLM/.venv` (uv venv, Python 3.12.15, vLLM 0.31.0, torch 2.13.0+cu132).
  No second environment. Pre-change freeze saved to `docs/env_freeze_before.txt`.
- Install safeguard (user rule): before any install, save the freeze, run `uv pip install --dry-run`, and stop if
  vllm/torch/transformers/xformers/nvidia-*/cuda-* would change.

## 2026-10-06: Installed pytest 9.1.1 and ruff 0.16.10
- **Engineering choice.** The dry run showed only 4 additions (iniconfig, pluggy, pytest, ruff), no changes to existing
  packages. pyyaml 6.0.3 was already present (vLLM dependency).

## 2026-10-06: Local model = Qwen2.5-1.5B-Instruct
- **Engineering choice** (user decision). Already cached; ~3.1 GB bf16 weights. Paper uses Llama-3.1-8B-Instruct (Sec. 5.1).

## 2026-10-06: Control KV budget in blocks, not GB
- **Interpretation.** The paper's budget study is in blocks (Fig. 14: 100–200). Use `num_gpu_blocks_override` so local
  and cloud runs are comparable. `gpu_memory_utilization` only has to leave enough room for weights + activations.

## 2026-10-06: Simulator before vLLM hook
- **Engineering choice.** Validate Eqs. 3–9 / Alg. 1 in a GPU-free block-cache simulator (fast tests), then port to a
  vLLM runtime plugin. Prefer plugin / monkey-patch over editing site-packages (see open_questions A1).

## 2026-10-06: Git tracks results/log only
- **Engineering choice.** `results/` is ignored except `results/log/` (small Markdown/JSON experiment records written
  by `/log-experiment`), so experiment metadata is versioned but bulky outputs are not.

## 2026-10-06: Edit hook runs ruff on the edited file only
- **Engineering choice.** `.claude/settings.json` PostToolUse hook → `scripts/hooks/ruff_on_edit.sh`. Fast (<1 s),
  never touches the GPU. Full test suite is run manually (`pytest -q`).

## 2026-10-06: enforce_eager=True locally (applies to ALL compared systems)
- **Engineering choice.** First run of `scripts/smoke_vllm_generate.py` (local config, enforce_eager=false): the
  model loaded fine (2.98 GiB weights), then failed with CUDA OOM during torch.compile autotuning
  (`InductorError: Failed to run autotuning code block`). RAM was fine; WSL did not crash.
- Fix: `enforce_eager: true` in `configs/hardware/local.yaml`. In vLLM 0.31.0 this sets
  `compilation_config.mode = NONE` (no torch.compile/Inductor autotuning) and `cudagraph_mode = NONE`
  (no CUDA graph capture); see `vllm/config/vllm.py`.
- **Fairness:** eager mode is a hardware-profile setting, so it applies identically to vanilla vLLM, the
  Continuum-style baseline, and CacheScout (all ablations). Never compare an eager run against a compiled run.
  Eager mode slows decoding, so absolute local latency/throughput is pessimistic, but relative comparisons hold.
- `configs/hardware/cloud.yaml` keeps `enforce_eager: false` (paper uses a normal vLLM setup, Sec. 5.1).
  Every result records the flag via its saved config.
- Smoke script reads GPU memory via NVML (pynvml, already installed with vLLM) rather than
  `torch.cuda.mem_get_info`, which reports the same device-wide numbers but would create a CUDA context
  (~0.3–0.5 GB) in the parent process.
