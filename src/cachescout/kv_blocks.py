"""KV-block arithmetic for vLLM's paged cache (block size 16 by default, paper Sec. 2.1)."""

import math
from dataclasses import dataclass


def blocks_for(prompt_tokens: int, max_tokens: int, block_size: int) -> int:
    """Conservative upper bound on blocks one request occupies: ceil((prompt + output) / bs)."""
    return math.ceil((prompt_tokens + max_tokens) / block_size)


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int,
                       dtype_bytes: int = 2) -> int:
    """KV-cache bytes per token: K and V, for every layer and KV head (bf16 = 2 bytes)."""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def kv_bytes_per_block(num_layers: int, num_kv_heads: int, head_dim: int,
                       block_size: int = 16, dtype_bytes: int = 2) -> int:
    return block_size * kv_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes)


def usable_blocks(num_gpu_blocks: int) -> int:
    """vLLM reserves one null block (docs/vllm_internals.md §6)."""
    return max(num_gpu_blocks - 1, 0)


@dataclass(frozen=True)
class EvictionPlan:
    """Sizing of the sequential LRU-eviction check (M1).

    Reasoning: cached free blocks form a FIFO queue; new blocks are taken from its head. When
    requests run strictly one at a time, once the fillers have allocated more than the usable
    pool in total, every block that was free before them (including all of the target's) has
    been taken from the head and its hash evicted. Each filler's partial last block is unhashed
    and re-used first (prepended), which can cost up to one block per filler, so `margin_blocks`
    covers that slack.
    """

    usable: int
    target_blocks: int
    filler_blocks_each: int
    num_fillers: int
    required_filler_blocks: int

    @property
    def total_filler_blocks(self) -> int:
        return self.num_fillers * self.filler_blocks_each


def plan_eviction(
    *,
    num_gpu_blocks: int,
    block_size: int,
    max_model_len: int,
    max_tokens: int,
    target_prompt_tokens: int,
    filler_prompt_tokens: int,
    margin_blocks: int,
) -> EvictionPlan:
    """Choose the number of fillers so their total blocks exceed usable + target + margin."""
    usable = usable_blocks(num_gpu_blocks)
    if filler_prompt_tokens + max_tokens > max_model_len:
        raise ValueError(
            f"filler ({filler_prompt_tokens}+{max_tokens} tokens) "
            f"exceeds max_model_len={max_model_len}"
        )
    if target_prompt_tokens + max_tokens > max_model_len:
        raise ValueError("target prompt exceeds max_model_len")
    target = blocks_for(target_prompt_tokens, max_tokens, block_size)
    filler = blocks_for(filler_prompt_tokens, max_tokens, block_size)
    if filler > usable or target > usable:
        raise ValueError("a single request needs more blocks than the usable pool")
    required = usable + target + margin_blocks
    n = math.ceil(required / filler)
    return EvictionPlan(usable, target, filler, n, required)
