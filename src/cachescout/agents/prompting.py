"""Chat-template prompts for agent calls, with history trimming to respect max_model_len.

Prompt structure (AutoGen group-chat style, engineering choice): [system: agent anchor] +
[user: "Task: <question>"] + shared history, where every teammate message is a user message
"[NAME]: text" and every tool result a user message "[tool:name] result". The chat template is
applied by the model's tokenizer and the token IDs are sent to vLLM, so prefixes are exact.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from cachescout.agents.definitions import anchor_text

FORCE_FINAL = ("[SYSTEM]: The call limit is reached. DECIDER, reply now with one line "
               "FINAL ANSWER: <number>.")


@dataclass(frozen=True)
class Message:
    speaker: str          # agent letter, or "tool:<name>", or "SYSTEM"
    text: str

    def as_chat(self, names: dict[str, str]) -> dict[str, str]:
        if self.speaker.startswith("tool:"):
            return {"role": "user", "content": f"[{self.speaker}] {self.text}"}
        if self.speaker == "SYSTEM":
            return {"role": "user", "content": self.text}
        return {"role": "user", "content": f"[{names[self.speaker]}]: {self.text}"}


@dataclass(frozen=True)
class BuiltPrompt:
    token_ids: list[int]
    dropped_messages: int      # history messages removed by trimming (0 = no trim)
    history_used: int


class PromptBuilder:
    """Builds token IDs for one agent call. `tokenize` maps chat messages -> token IDs (the real
    one wraps tokenizer.apply_chat_template(..., add_generation_prompt=True))."""

    def __init__(self, tokenize: Callable[[list[dict[str, str]]], list[int]], topology: str,
                 max_model_len: int, max_tokens: int, names: dict[str, str]) -> None:
        self.tokenize = tokenize
        self.topology = topology
        self.budget = max_model_len - max_tokens
        self.names = names

    def build(self, agent: str, question: str, history: Sequence[Message],
              force_final: bool = False) -> BuiltPrompt:
        head = [{"role": "system", "content": anchor_text(agent, self.topology)},
                {"role": "user", "content": f"Task: {question}"}]
        tail = [{"role": "user", "content": FORCE_FINAL}] if force_final else []
        hist = [m.as_chat(self.names) for m in history]
        dropped = 0
        while True:
            ids = self.tokenize(head + hist[dropped:] + tail)
            if len(ids) <= self.budget or dropped >= len(hist):
                break
            dropped += 1                       # drop the oldest message after anchor + task
        if len(ids) > self.budget:
            raise ValueError(f"anchor + task alone exceed the prompt budget ({len(ids)} > "
                             f"{self.budget})")
        return BuiltPrompt(ids, dropped, len(hist) - dropped)


def hf_chat_tokenizer(tokenizer: Any) -> Callable[[list[dict[str, str]]], list[int]]:
    """Wrap a Hugging Face tokenizer's chat template (explicit system message always present, so
    Qwen's default system prompt never appears)."""

    def tokenize(messages: list[dict[str, str]]) -> list[int]:
        assert messages and messages[0]["role"] == "system"
        out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return list(ids)

    return tokenize
