"""Environment smoke tests: GPU visible, CUDA usable, pinned versions present.

Deliberately does NOT load a model or start vLLM.
That is scripts/smoke_vllm_generate.py, run by hand.
"""

import shutil
import subprocess
from importlib.metadata import version

import pytest

EXPECTED = {"vllm": "0.31.0", "torch": "2.13.0+cu132", "transformers": "5.17.0"}


@pytest.mark.parametrize("pkg", sorted(EXPECTED))
def test_pinned_versions(pkg: str) -> None:
    """Core stack must match CLAUDE.md; never upgraded without asking."""
    assert version(pkg) == EXPECTED[pkg]


def test_nvidia_smi_works() -> None:
    """The Windows driver exposes the GPU to WSL via nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    assert exe is not None, "nvidia-smi not found in WSL"
    out = subprocess.run(
        [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout
    assert out.strip()


@pytest.mark.gpu
def test_torch_cuda_available() -> None:
    """torch sees the GPU and can run a tiny kernel (a few KB of VRAM)."""
    import torch

    assert torch.cuda.is_available()
    x = torch.ones(4, device="cuda")
    assert float((x * 2).sum()) == 8.0


def test_vllm_importable() -> None:
    """Import only; no engine is constructed."""
    import vllm

    assert vllm.__version__ == EXPECTED["vllm"]
