"""Tests for KV-block arithmetic used by the M1 eviction check (no GPU)."""

import pytest

from cachescout.kv_blocks import blocks_for, plan_eviction, usable_blocks


def test_blocks_for() -> None:
    assert blocks_for(200, 8, 16) == 13
    assert blocks_for(1200, 8, 16) == 76
    assert blocks_for(2040, 8, 16) == 128


def test_usable_blocks() -> None:
    assert usable_blocks(256) == 255
    assert usable_blocks(0) == 0


def test_plan_eviction_local_config() -> None:
    p = plan_eviction(num_gpu_blocks=256, block_size=16, max_model_len=2048, max_tokens=8,
                      target_prompt_tokens=200, filler_prompt_tokens=1200, margin_blocks=32)
    assert (p.usable, p.target_blocks, p.filler_blocks_each) == (255, 13, 76)
    assert p.required_filler_blocks == 300
    assert p.num_fillers == 4 and p.total_filler_blocks == 304
    assert p.total_filler_blocks > p.usable


def test_plan_eviction_rejects_filler_longer_than_max_model_len() -> None:
    with pytest.raises(ValueError):
        plan_eviction(num_gpu_blocks=256, block_size=16, max_model_len=2048, max_tokens=8,
                      target_prompt_tokens=200, filler_prompt_tokens=2041, margin_blocks=0)


def test_plan_eviction_rejects_request_bigger_than_pool() -> None:
    with pytest.raises(ValueError):
        plan_eviction(num_gpu_blocks=100, block_size=16, max_model_len=2048, max_tokens=8,
                      target_prompt_tokens=200, filler_prompt_tokens=1600, margin_blocks=0)
