"""Check that the six agents get DISTINCT plugin fingerprints under a model's chat template.

The CacheScout plugin identifies an agent by its first `fingerprint_blocks` KV blocks (default 2 =
32 tokens; Sec. 4 "prefix block hashes"). Chat templates prepend a shared header before the system
text: Qwen2.5 adds 3 tokens (`<|im_start|>system\\n`); Llama 3.1 adds `<|begin_of_text|>`, a role
header and a "Cutting Knowledge Date / Today Date" preamble. If the shared header plus the shared
start of the anchors is longer than the fingerprint window, different agents would collide.
"""

from collections.abc import Callable
from dataclasses import dataclass

from cachescout.agents.definitions import AGENTS, anchor_text


@dataclass(frozen=True)
class FingerprintReport:
    topology: str
    window_tokens: int            # fingerprint_blocks * block_size
    shared_prefix_tokens: int     # tokens identical across ALL six agents' prompts
    first_divergence: dict[str, int]  # per agent: token index where it becomes unique
    distinct: bool                # six different fingerprints inside the window
    min_blocks_needed: int        # smallest fingerprint_blocks that would separate all six


def _common_prefix(seqs: list[list[int]]) -> int:
    n = 0
    for column in zip(*seqs, strict=False):
        if len(set(column)) != 1:
            break
        n += 1
    return n


def fingerprint_report(tokenize: Callable[[list[dict[str, str]]], list[int]], topology: str,
                       fingerprint_blocks: int = 2, block_size: int = 16) -> FingerprintReport:
    prompts = {a: tokenize([{"role": "system", "content": anchor_text(a, topology)},
                            {"role": "user", "content": "Task: x"}]) for a in AGENTS}
    seqs = list(prompts.values())
    shared = _common_prefix(seqs)
    divergence = {}
    for a, ids in prompts.items():
        others = [p for b, p in prompts.items() if b != a]
        divergence[a] = max(_common_prefix([ids, o]) for o in others)
    window = fingerprint_blocks * block_size
    distinct = len({tuple(p[:window]) for p in seqs}) == len(seqs)
    need = max(divergence.values()) // block_size + 1      # block containing the last divergence
    return FingerprintReport(topology, window, shared, divergence, distinct, need)
