"""A stand-in for AsyncAnthropic that replays scripted responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str = ""
    signature: str = "sig-abc"
    type: str = "thinking"


@dataclass
class ToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class StopDetails:
    category: str
    type: str = "refusal"
    explanation: str = ""


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    stop_details: StopDetails | None = None
    model: str = "claude-opus-5"


def says(text: str) -> FakeMessage:
    return FakeMessage(content=[ThinkingBlock(), TextBlock(text)], stop_reason="end_turn")


def calls(*tools: tuple[str, dict[str, Any]]) -> FakeMessage:
    blocks = [ThinkingBlock()]
    blocks += [
        ToolUseBlock(name=name, input=args, id=f"toolu_{i}")
        for i, (name, args) in enumerate(tools)
    ]
    return FakeMessage(content=blocks, stop_reason="tool_use")


class _Stream:
    def __init__(self, message: FakeMessage) -> None:
        self._message = message

    async def __aenter__(self) -> "_Stream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def get_final_message(self) -> FakeMessage:
        return self._message


class _Messages:
    def __init__(self, owner: "FakeAnthropic") -> None:
        self._owner = owner

    def stream(self, **kwargs: Any) -> _Stream:
        self._owner.requests.append(kwargs)
        if not self._owner.script:
            raise AssertionError(
                f"agent made {len(self._owner.requests)} requests but the script ran out"
            )
        return _Stream(self._owner.script.pop(0))


class _Beta:
    def __init__(self, owner: "FakeAnthropic") -> None:
        self.messages = _Messages(owner)


class FakeAnthropic:
    """Replays `script` one response per request, recording each request."""

    def __init__(self, script: list[FakeMessage]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self)
        self.beta = _Beta(self)
