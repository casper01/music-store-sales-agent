"""Test doubles shared by the test suite."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from sales_agent.agent.contracts import Message, ModelResponse, ToolCallRequest

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder: no network, stable across runs."""

    def __init__(self, dimensions: int = 256, model_name: str = "fake-hashing-embedder") -> None:
        self.dimensions = dimensions
        self.model_name = model_name
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def tool_call(
    name: str, arguments: dict[str, Any] | str, call_id: str = "call-1"
) -> ToolCallRequest:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCallRequest(id=call_id, name=name, arguments_json=raw)


def calls(*requests: ToolCallRequest) -> ModelResponse:
    return ModelResponse(content=None, tool_calls=tuple(requests))


def answer(text: str) -> ModelResponse:
    return ModelResponse(content=text)


Step = ModelResponse | Callable[[list[Message]], ModelResponse]


class ScriptedModel:
    """Returns pre-defined responses in order.

    A step may be a callable receiving the conversation so far, which lets a test build the final
    answer from real tool output instead of hardcoding it.
    """

    def __init__(self, steps: Iterable[Step]) -> None:
        self._steps = iter(steps)
        self.requests: list[tuple[list[Message], list[dict[str, Any]]]] = []

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        self.requests.append(([dict(message) for message in messages], tools))
        step = next(self._steps)
        return step(messages) if callable(step) else step


def tool_outputs(messages: list[Message]) -> list[dict[str, Any]]:
    return [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
