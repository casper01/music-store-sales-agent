"""Contracts between the agent loop, the chat model, and the tools.

The conversation history uses the OpenAI Chat Completions message format (``role``/``content``/
``tool_calls``/``tool_call_id``) as the internal wire format, because it is the de-facto standard
that many providers accept. Model *responses*, however, are normalised into the provider-neutral
:class:`ModelResponse`, so the loop never depends on SDK objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

Message = dict[str, Any]
ToolDefinition = dict[str, Any]


class AgentTool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    id: str
    name: str
    arguments_json: str
    # Opaque provider-specific fields that must be echoed back unchanged in the next request,
    # e.g. Gemini's ``extra_content.google.thought_signature`` (omitting it causes HTTP 400).
    provider_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = ()

    def to_message(self) -> Message:
        message: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    **call.provider_fields,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    def complete(self, messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse: ...


class StopReason(StrEnum):
    ANSWERED = "answered"
    ITERATION_LIMIT = "iteration_limit"
    EMPTY_RESPONSE = "empty_response"


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One executed tool call, kept for tracing, the CLI and evaluation."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentResult:
    answer: str
    stop_reason: StopReason
    iterations: int
    tool_calls: tuple[ToolCallRecord, ...] = field(default_factory=tuple)

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]
