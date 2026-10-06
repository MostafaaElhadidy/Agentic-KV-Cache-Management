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
- (Correction, see next entry: the actual OOM site was the KV-cache allocation, not autotuning. enforce_eager
  stays on; it is harmless for fairness and avoids compile memory spikes.)
- Smoke script reads GPU memory via NVML (pynvml, already installed with vLLM) rather than
  `torch.cuda.mem_get_info`, which reports the same device-wide numbers but would create a CUDA context
  (~0.3–0.5 GB) in the parent process.

## 2026-10-06: Fixed KV budget locally: num_gpu_blocks_override=256, max_model_len=2048
- **Engineering choice.** Root cause of the smoke-test OOM (user-diagnosed): `allocate_kv_cache`
  (`vllm/v1/worker/utils.py`) tried to allocate 1.23 GiB with 3.86 GiB reported free. The KV-cache size vLLM derives
  from `gpu_memory_utilization` is too large to allocate reliably on this WSL setup; it is not the profiling run.
- Fix in `configs/hardware/local.yaml`: `num_gpu_blocks_override: 256` (≈ 256 × 448 KiB = 112 MiB KV for
  Qwen2.5-1.5B: 28 layers × 2 KV heads × 128 dim × K,V × 2 B × 16 tokens) and `max_model_len: 2048`.
- Verified in installed vLLM 0.31.0 source (read-only):
  - `v1/core/kv_cache_utils.py::may_override_num_blocks` replaces the profiled block count with the override;
    `get_kv_cache_config_from_groups` then sets the allocation `size = bytes_per_block * num_blocks`, and
    `v1/worker/utils.py::allocate_kv_cache` allocates exactly that buffer. So the override shrinks the real allocation.
  - `get_kv_cache_configs` logs "Overriding num_gpu_blocks=X with num_gpu_blocks_override=Y" and plans the
    max_model_len admission check against `override × bytes_per_block` **minus one block** (the null block that
    `BlockPool` permanently reserves, `v1/core/block_pool.py`). Usable blocks = override − 1.
  - The final count is sent back to the frontend (`v1/engine/core_client.py`), and the smoke script prints it
    from `llm.llm_engine.vllm_config.cache_config.num_gpu_blocks`.
- Capacity: 255 usable blocks; one 2048-token sequence needs 128 blocks, so ≈ 2 full-length sequences fit at once.
- **Implication for the paper's cache study (Fig. 14, 100–200 blocks):** max_model_len must satisfy
  `ceil(max_model_len / 16) ≤ budget − 1`. For 100 blocks: ≤ 99 × 16 = **1584 tokens** (not 1600, because of the
  null block); for 150: ≤ 2384; for 200: ≤ 3184. The paper notes a ~93-block max request footprint (Sec. 5.4), which
  fits within 99. Experiment configs for the budget sweep must set max_model_len accordingly.
- `gpu_memory_utilization` (0.60) still bounds vLLM's startup memory check and profiling, but no longer sizes the KV cache.
