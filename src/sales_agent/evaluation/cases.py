"""Evaluation case model and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sales_agent.errors import ConfigurationError

KNOWN_TOOLS = frozenset({"get_schema", "run_sql", "search_knowledge"})
CHECK_TYPES = frozenset(
    {"sql_values_present", "sql_numbers_present", "contains_any", "contains_all", "not_contains"}
)


@dataclass(frozen=True, slots=True)
class Check:
    type: str
    sql: str | None = None
    values: tuple[str, ...] = ()
    tolerance: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Check:
        check_type = data.get("type")
        if check_type not in CHECK_TYPES:
            raise ConfigurationError(f"Unknown check type: {check_type!r}")
        check = cls(
            type=check_type,
            sql=data.get("sql"),
            values=tuple(data.get("values", ())),
            tolerance=float(data.get("tolerance", 0.05)),
        )
        if check_type.startswith("sql_") and not check.sql:
            raise ConfigurationError(f"Check {check_type} requires 'sql'")
        if not check_type.startswith("sql_") and not check.values:
            raise ConfigurationError(f"Check {check_type} requires 'values'")
        return check


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: int
    question: str
    criteria: str
    required_tools: frozenset[str]
    forbidden_tools: frozenset[str]
    checks: tuple[Check, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        case = cls(
            id=int(data["id"]),
            question=str(data["question"]).strip(),
            criteria=str(data["criteria"]).strip(),
            required_tools=frozenset(data.get("required_tools", ())),
            forbidden_tools=frozenset(data.get("forbidden_tools", ())),
            checks=tuple(Check.from_dict(item) for item in data.get("checks", ())),
        )
        unknown = (case.required_tools | case.forbidden_tools) - KNOWN_TOOLS
        if unknown:
            raise ConfigurationError(f"Case {case.id}: unknown tools {sorted(unknown)}")
        if case.required_tools & case.forbidden_tools:
            raise ConfigurationError(f"Case {case.id}: a tool is both required and forbidden")
        if not case.question or not case.criteria:
            raise ConfigurationError(f"Case {case.id}: question and criteria are required")
        return case


def load_cases(path: Path) -> list[EvalCase]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load evaluation cases from {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ConfigurationError("Evaluation file must contain a JSON list of cases")
    try:
        cases = [EvalCase.from_dict(item) for item in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Malformed evaluation case: {exc}") from exc
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("Evaluation case ids must be unique")
    return cases
