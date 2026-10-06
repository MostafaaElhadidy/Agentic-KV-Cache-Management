# CacheScout replication

Faithful replication of "Learning Agent Execution for KV-Cache Management in Agentic Serving"
(CacheScout, arXiv:2608.14624, `paper/cachescout.pdf`). Phase 1: local, small scale (8 GB GPU).
Phase 2: cloud GPUs (A100/H100). Same code; hardware/model settings only in `configs/`.

## Environment (verified 2026-10-06)
- Windows 11 + WSL2, Ubuntu 26.04.1, kernel 6.18. Linux/bash only; repo and HF cache stay in `~/` (not `/mnt/c`).
- GPU: RTX 4060 Laptop, 8188 MiB (~370 MiB used by Windows). Driver comes from Windows; never install one in WSL.
- RAM 11 GiB, swap 16 GiB (WSL limits set by the user in `.wslconfig`).
- Python env: `~/testLLM/.venv` (uv venv). Activate: `source ~/testLLM/.venv/bin/activate` (alias `vllmenv`).
- Versions: Python 3.12.15, **vllm 0.31.0**, **torch 2.13.0+cu132** (CUDA 13.2), transformers 5.17.0.
  Paper targets vLLM v0.11, so see `docs/open_questions.md` A1. Full freeze: `docs/env_freeze_before.txt`.
- Local model: `Qwen/Qwen2.5-1.5B-Instruct` (cached in `~/.cache/huggingface`). Don't download others without asking.

## Docs
- `docs/paper_notes.md`: paper summary, equations, Alg. 1, settings table, paper inconsistencies
- `docs/open_questions.md`: ambiguities + recommended defaults (A1 vLLM version is top priority)
- `docs/PLAN.md`: milestones M1–M6 with checkboxes and the Phase 1 success criteria
- `docs/decisions.md`: every non-paper choice (labelled paper / interpretation / engineering choice)
- `docs/project_instructions.md`: text for the claude.ai Project

## Commands
```bash
source ~/testLLM/.venv/bin/activate          # every new terminal
pytest -q                                     # smoke tests (no model loading)
ruff check .                                  # lint
python scripts/smoke_vllm_generate.py --config configs/hardware/local.yaml   # LOADS MODEL: only when user asks
# Baseline benchmark / experiment runner: not implemented yet (M1). Planned:
#   python -m cachescout.run --config configs/experiments/<exp>/local.yaml
```

## Layout
```
paper/        the PDF (source of truth)
docs/         notes, plan, questions, decisions
src/cachescout/  package (empty until M1)
configs/hardware/{local,cloud}.yaml       GPU/model/vLLM settings
configs/experiments/<exp>/{local,cloud}.yaml
scripts/      runnable scripts; scripts/hooks/ for Claude Code hooks
tests/        pytest
results/      outputs (git-ignored) ; results/log/ experiment records (tracked)
.claude/      agents, commands, settings (ruff hook on edit)
```

## Conventions
- Type hints everywhere; docstrings cite the paper (e.g. "Eq. 9, Sec. 3.3") or say "interpretation"/"engineering choice".
- Experiments are YAML-config driven; no hardware or model constants in code.
- Every result is saved with its resolved config, git commit, seed, and package versions.
- Pinned versions in `requirements-extra.txt`; core stack pinned by the existing env.
- Installs: save `uv pip freeze` first, `uv pip install --dry-run`, stop if vllm/torch/transformers/xformers/nvidia/cuda change.

## Rules
- Never fabricate paper details. Log uncertainty in `docs/open_questions.md` and ask the user.
- Run `pytest -q` (and `ruff check .`) before declaring anything done.
- Prefer a patch/plugin over editing vLLM source; log it in `docs/decisions.md`.
- Never reinstall or upgrade vLLM/torch (or install large packages / download models) without asking.
- Don't load models or run vLLM generation unless the user asks for it in that session.
- Keep code hardware-agnostic; GPU/model settings live in configs.
- Update `docs/PLAN.md` checkboxes after each milestone.
- The user is new to Linux/WSL: explain commands simply.
