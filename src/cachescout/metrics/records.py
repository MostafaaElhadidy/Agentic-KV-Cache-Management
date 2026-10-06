"""One record per agent turn (one LLM call), the unit all metrics are computed from.

Engineering choice: records are independent of how requests are sent (offline `LLM` or
OpenAI server), so `summarize()` is shared by every collector.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TurnRecord:
    """Measurements for a single agent invocation.

    Times are seconds from one clock (`time.perf_counter()` in our collectors); never mix
    with vLLM's internal timestamps (docs/vllm_internals.md §4).

    Attributes:
        request_id: Unique request ID.
        prompt_tokens: Prompt length in tokens.
        cached_tokens: Prompt tokens served from the prefix cache (vLLM `num_cached_tokens`).
        output_tokens: Generated tokens.
        t_send: When the request was submitted.
        t_first_token: When the first output token was available; None if unknown.
        t_done: When the request finished.
        session_id: Session (user request) this turn belongs to.
        agent_id: Agent that issued the turn.
        turn_idx: Position of the turn within its session.
        is_warmup: CacheScout background warmup (paper Sec. 3.4); excluded from hit rate and
            latency, but its time counts toward throughput (docs/decisions.md, interpretation).
        vllm_first_token_latency: vLLM's own TTFT duration (`RequestStateStats`), cross-check only.
    """

    request_id: str
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    t_send: float
    t_first_token: float | None
    t_done: float
    session_id: str | None = None
    agent_id: str | None = None
    turn_idx: int | None = None
    is_warmup: bool = False
    vllm_first_token_latency: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.cached_tokens <= self.prompt_tokens:
            raise ValueError(
                f"{self.request_id}: cached_tokens={self.cached_tokens} not in "
                f"[0, prompt_tokens={self.prompt_tokens}]"
            )
        if self.t_done < self.t_send:
            raise ValueError(f"{self.request_id}: t_done < t_send")
        if self.t_first_token is not None and not (
            self.t_send <= self.t_first_token <= self.t_done
        ):
            raise ValueError(f"{self.request_id}: t_first_token outside [t_send, t_done]")

    @property
    def ttft(self) -> float | None:
        """Time to first token (paper Sec. 5.1): arrival to first generated token."""
        return None if self.t_first_token is None else self.t_first_token - self.t_send

    @property
    def latency(self) -> float:
        """Per-turn latency (paper Sec. 5.1): end-to-end, prefill + decode."""
        return self.t_done - self.t_send

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON results."""
        return asdict(self)
