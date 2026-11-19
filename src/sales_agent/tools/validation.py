"""Validation of tool arguments against the tool's own JSON Schema.

Only the subset of JSON Schema used by the tool definitions is supported (flat objects with
``required``, ``additionalProperties: false``, primitive ``type`` and string ``minLength``/
``maxLength``). Using the same schema for the model-facing definition and for validation keeps
the two from drifting apart.
"""

from __future__ import annotations

from typing import Any

from sales_agent.errors import ToolInputError

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def validate_arguments(tool_name: str, arguments: Any, schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ToolInputError(f"{tool_name}: arguments must be a JSON object")
    properties: dict[str, dict[str, Any]] = schema.get("properties", {})

    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            raise ToolInputError(f"{tool_name}: unexpected argument(s): {', '.join(unexpected)}")

    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        raise ToolInputError(f"{tool_name}: missing required argument(s): {', '.join(missing)}")

    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            continue
        expected = _TYPES.get(spec.get("type", ""))
        # bool is a subclass of int; never accept it for numeric fields.
        is_bool_for_number = isinstance(value, bool) and spec.get("type") in {"integer", "number"}
        if expected is not None and (not isinstance(value, expected) or is_bool_for_number):
            raise ToolInputError(f"{tool_name}: argument {name!r} must be of type {spec['type']}")
        if isinstance(value, str):
            length = len(value.strip())
            if length < spec.get("minLength", 0):
                raise ToolInputError(f"{tool_name}: argument {name!r} must not be empty")
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                raise ToolInputError(
                    f"{tool_name}: argument {name!r} exceeds {spec['maxLength']} characters"
                )
    return arguments
