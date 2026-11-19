"""The tool-calling loop.

One iteration = one model call. The flow of a single tool call is:

1. send the conversation + tool definitions (JSON Schema) to the model;
2. the model answers either with text (done) or with one or more ``tool_calls``, each carrying a
   tool name and its arguments as a JSON string;
3. the loop parses the arguments, validates and executes the tool, and appends a ``tool``
   message with the JSON result (or a JSON error the model can react to) under the same
   ``tool_call_id``;
4. go back to 1 with the extended conversation.

The loop stops on a text answer or after ``max_iterations`` model calls. In the latter case it
returns an explicit "limit reached" message instead of guessing an answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from sales_agent.agent.contracts import (
    AgentResult,
    AgentTool,
    ChatModel,
    Message,
    StopReason,
    ToolCallRecord,
    ToolCallRequest,
    ToolDefinition,
)
from sales_agent.agent.prompts import SYSTEM_PROMPT
from sales_agent.errors import SalesAgentError

logger = logging.getLogger(__name__)

ITERATION_LIMIT_ANSWER = (
    "I reached the agent step limit before I could finish, so I do not have a complete, "
    "source-backed answer. Please ask a narrower or more specific question."
)
EMPTY_RESPONSE_ANSWER = "The model returned an empty response, so I have no supported answer."

ToolCallObserver = Callable[[ToolCallRecord], None]
# Called with the iteration number and the full conversation just before each model call.
ModelCallObserver = Callable[[int, list[Message]], None]


class AgentOrchestrator:
    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[AgentTool],
        max_iterations: int,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        max_tool_output_chars: int = 20_000,
        on_tool_call: ToolCallObserver | None = None,
        on_model_call: ModelCallObserver | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if max_tool_output_chars < 100:
            raise ValueError("max_tool_output_chars must be at least 100")
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique")
        self._model = model
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._max_tool_output_chars = max_tool_output_chars
        self._on_tool_call = on_tool_call
        self._on_model_call = on_model_call
        self._definitions: list[ToolDefinition] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return list(self._definitions)

    def ask(self, question: str) -> str:
        """Convenience wrapper returning only the final answer text."""
        return self.run(question).answer

    def run(self, question: str) -> AgentResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty")

        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question.strip()},
        ]
        records: list[ToolCallRecord] = []

        for iteration in range(1, self._max_iterations + 1):
            if self._on_model_call is not None:
                self._on_model_call(iteration, messages)
            response = self._model.complete(messages, self._definitions)
            messages.append(response.to_message())

            if not response.tool_calls:
                content = (response.content or "").strip()
                if content:
                    return AgentResult(content, StopReason.ANSWERED, iteration, tuple(records))
                return AgentResult(
                    EMPTY_RESPONSE_ANSWER, StopReason.EMPTY_RESPONSE, iteration, tuple(records)
                )

            for call in response.tool_calls:
                record, payload = self._execute(call)
                records.append(record)
                if self._on_tool_call is not None:
                    self._on_tool_call(record)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": self._serialize(payload)}
                )

        logger.warning("Agent stopped after %d iterations", self._max_iterations)
        return AgentResult(
            ITERATION_LIMIT_ANSWER, StopReason.ITERATION_LIMIT, self._max_iterations, tuple(records)
        )

    def _execute(self, call: ToolCallRequest) -> tuple[ToolCallRecord, dict[str, Any]]:
        arguments: dict[str, Any] = {}
        try:
            parsed = json.loads(call.arguments_json or "{}")
            if not isinstance(parsed, dict):
                raise SalesAgentError("Tool arguments must be a JSON object")
            arguments = parsed
            tool = self._tools.get(call.name)
            if tool is None:
                raise SalesAgentError(
                    f"Unknown tool {call.name!r}. Available tools: {', '.join(self._tools)}"
                )
            result = tool.execute(arguments)
        except json.JSONDecodeError as exc:
            return self._failure(call.name, arguments, f"Arguments are not valid JSON: {exc}")
        except SalesAgentError as exc:
            return self._failure(call.name, arguments, str(exc))

        logger.info("Tool %s succeeded with arguments %s", call.name, arguments)
        return ToolCallRecord(call.name, arguments, ok=True), result

    @staticmethod
    def _failure(
        name: str, arguments: dict[str, Any], message: str
    ) -> tuple[ToolCallRecord, dict[str, Any]]:
        logger.info("Tool %s failed: %s", name, message)
        return ToolCallRecord(name, arguments, ok=False, error=message), {"error": message}

    def _serialize(self, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= self._max_tool_output_chars:
            return text
        return json.dumps(
            {
                "partial_output": text[: self._max_tool_output_chars],
                "note": (
                    f"Output exceeded {self._max_tool_output_chars} characters and was cut. "
                    "Do not rely on missing rows; aggregate or narrow the query."
                ),
            },
            ensure_ascii=False,
        )
