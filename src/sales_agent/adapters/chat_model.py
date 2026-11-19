"""Chat model adapter for any OpenAI-compatible Chat Completions API.

The default provider is Google Gemini through its OpenAI-compatible endpoint, but OpenAI itself
and other compatible servers work unchanged. Provider differences handled here:

* **Portable tool schemas** - Gemini accepts only a subset of JSON Schema in function
  declarations, so the schema sent to the API keeps just ``type``, ``description``,
  ``properties``, ``required``, ``items`` and ``enum``. The full schema (``minLength``,
  ``additionalProperties`` ...) is still enforced locally by each tool before execution.
* **Opaque per-call fields** - Gemini 3 models attach a "thought signature" to each tool call
  (``extra_content``). Any field outside the standard ones is preserved in
  ``ToolCallRequest.provider_fields`` and echoed back by the agent loop, as the API requires.
* **Missing call ids** - if a provider omits ``id``, a stable one is generated so the tool
  result can still be matched to its call.
"""

from __future__ import annotations

import copy
from typing import Any

import openai

from sales_agent.agent.contracts import (
    Message,
    ModelResponse,
    ToolCallRequest,
    ToolDefinition,
)
from sales_agent.errors import ModelError

_PORTABLE_SCHEMA_KEYS = frozenset(
    {"type", "description", "properties", "required", "items", "enum"}
)
_STANDARD_TOOL_CALL_KEYS = frozenset({"id", "type", "function", "index"})


def portable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON Schema keywords that some providers reject, recursively."""
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _PORTABLE_SCHEMA_KEYS:
            continue
        if key == "properties":
            result[key] = {name: portable_schema(spec) for name, spec in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = portable_schema(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def portable_tools(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    portable = []
    for tool in tools:
        function = dict(tool["function"])
        parameters = portable_schema(function.get("parameters", {}))
        if parameters.get("properties"):
            function["parameters"] = parameters
        else:
            # A tool without arguments: some providers reject an object with no properties.
            function.pop("parameters", None)
        portable.append({"type": "function", "function": function})
    return portable


class OpenAICompatibleChatModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float | None = None,
        timeout_seconds: float = 90.0,
        max_retries: int = 6,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=max_retries
        )

    def complete(self, messages: list[Message], tools: list[ToolDefinition]) -> ModelResponse:
        request: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = portable_tools(tools)
            request["tool_choice"] = "auto"
        if self.temperature is not None:
            request["temperature"] = self.temperature
        try:
            response = self._client.chat.completions.create(**request)
        except openai.AuthenticationError as exc:
            raise ModelError("LLM authentication failed; check LLM_API_KEY") from exc
        except openai.PermissionDeniedError as exc:
            raise ModelError(f"LLM access denied (check the key and model name): {exc}") from exc
        except openai.NotFoundError as exc:
            raise ModelError(
                f"Model {self.model!r} was not found at this endpoint; check LLM_MODEL"
            ) from exc
        except openai.RateLimitError as exc:
            raise ModelError(
                "LLM rate limit or quota exceeded (free tiers allow only a few requests per "
                "minute); wait a minute and try again"
            ) from exc
        except openai.APITimeoutError as exc:
            raise ModelError("LLM request timed out; increase LLM_TIMEOUT_SECONDS") from exc
        except openai.APIConnectionError as exc:
            raise ModelError(f"Could not reach the LLM API: {exc}") from exc
        except openai.OpenAIError as exc:
            raise ModelError(f"LLM request failed: {exc}") from exc

        if not response.choices:
            raise ModelError("LLM returned no choices")
        message = response.choices[0].message
        tool_calls = tuple(
            ToolCallRequest(
                id=getattr(call, "id", None) or f"call_{number}",
                name=call.function.name,
                arguments_json=call.function.arguments or "{}",
                provider_fields=_extra_fields(call),
            )
            for number, call in enumerate(message.tool_calls or [], start=1)
            if getattr(call, "type", "function") == "function"
        )
        return ModelResponse(content=message.content, tool_calls=tool_calls)


def _extra_fields(call: Any) -> dict[str, Any]:
    """Return non-standard fields of an SDK tool-call object (e.g. Gemini ``extra_content``)."""
    extra = getattr(call, "model_extra", None) or {}
    return {
        key: copy.deepcopy(value)
        for key, value in extra.items()
        if key not in _STANDARD_TOOL_CALL_KEYS and value is not None
    }
