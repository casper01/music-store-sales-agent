"""Runs evaluation cases against an agent and grades the results.

Grading is deliberately mechanical and transparent:

* **tool selection** - every ``required_tools`` entry was called successfully at least once,
  and no ``forbidden_tools`` entry was called;
* **safety** - no ``run_sql`` call contained anything but a read query, and the database file's
  SHA-256 is identical before and after the whole run;
* **grounding** - expected values are computed *at evaluation time* by running the case's
  reference SQL against the same database, then looked up in the answer text (accent- and
  case-insensitive; numbers with a tolerance, in English or Polish number formatting).

Keyword checks are heuristics: they catch missing content, not subtle reasoning errors, so the
report keeps every answer for a human to read.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sales_agent.adapters.sqlite_reader import SqliteReader
from sales_agent.agent.contracts import AgentResult, ToolCallRecord
from sales_agent.errors import SalesAgentError
from sales_agent.evaluation.cases import Check, EvalCase

_READ_ONLY_START = re.compile(r"^\s*(select|with|values)\b", re.IGNORECASE)
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_NUMBER_TOKEN = re.compile(r"\d(?:[\d,.\u00a0\u202f]*\d)?")


class QuestionAnswerer(Protocol):
    def run(self, question: str) -> AgentResult: ...


@dataclass(slots=True)
class CaseResult:
    case: EvalCase
    answer: str
    tool_calls: tuple[ToolCallRecord, ...]
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]


@dataclass(slots=True)
class EvalReport:
    results: list[CaseResult]
    database_unchanged: bool

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def all_passed(self) -> bool:
        return self.database_unchanged and self.passed_count == len(self.results)


class Evaluator:
    def __init__(self, agent: QuestionAnswerer, reader: SqliteReader) -> None:
        self._agent = agent
        self._reader = reader

    def run(
        self,
        cases: Sequence[EvalCase],
        on_result: Callable[[CaseResult], None] | None = None,
    ) -> EvalReport:
        before = file_sha256(self._reader.database_path)
        results = []
        for case in cases:
            result = self.evaluate_case(case)
            results.append(result)
            if on_result is not None:
                on_result(result)
        after = file_sha256(self._reader.database_path)
        return EvalReport(results=results, database_unchanged=before == after)

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        try:
            outcome = self._agent.run(case.question)
        except SalesAgentError as exc:
            return CaseResult(case, answer="", tool_calls=(), failures=[f"Agent error: {exc}"])

        result = CaseResult(case, answer=outcome.answer, tool_calls=outcome.tool_calls)
        result.failures.extend(_check_tools(case, outcome.tool_calls))
        for check in case.checks:
            try:
                failure = self._apply_check(check, outcome.answer)
            except SalesAgentError as exc:
                failure = f"Check {check.type} could not run: {exc}"
            if failure:
                result.failures.append(failure)
        return result

    def _apply_check(self, check: Check, answer: str) -> str | None:
        if check.type == "contains_any":
            if not any(_contains(answer, value) for value in check.values):
                return f"Answer mentions none of: {', '.join(check.values)}"
        elif check.type == "contains_all":
            missing = [value for value in check.values if not _contains(answer, value)]
            if missing:
                return f"Answer is missing: {', '.join(missing)}"
        elif check.type == "not_contains":
            present = [value for value in check.values if _contains(answer, value)]
            if present:
                return f"Answer must not mention: {', '.join(present)}"
        elif check.type == "sql_values_present":
            expected = self._reference_values(check)
            missing = [str(value) for value in expected if not _contains(answer, str(value))]
            if missing:
                return f"Answer is missing expected value(s): {', '.join(missing)}"
        elif check.type == "sql_numbers_present":
            found = extract_numbers(answer)
            missing = [
                value
                for value in self._reference_values(check)
                if not _number_present(float(value), found, check.tolerance)
            ]
            if missing:
                formatted = ", ".join(f"{float(value):.2f}" for value in missing)
                return f"Answer is missing expected number(s): {formatted}"
        return None

    def _reference_values(self, check: Check) -> list[Any]:
        assert check.sql is not None
        result = self._reader.run_sql(check.sql)
        if not result.rows:
            raise SalesAgentError(f"Reference SQL returned no rows: {check.sql}")
        first_column = result.columns[0]
        return [row[first_column] for row in result.rows]


def _check_tools(case: EvalCase, calls: Iterable[ToolCallRecord]) -> list[str]:
    calls = list(calls)
    failures = []
    succeeded = {call.name for call in calls if call.ok}
    used = {call.name for call in calls}
    missing = sorted(case.required_tools - succeeded)
    if missing:
        failures.append(f"Required tool(s) not used successfully: {', '.join(missing)}")
    forbidden = sorted(case.forbidden_tools & used)
    if forbidden:
        failures.append(f"Forbidden tool(s) used: {', '.join(forbidden)}")
    for call in calls:
        query = call.arguments.get("query") if call.name == "run_sql" else None
        if isinstance(query, str) and not is_read_query(query):
            failures.append(f"Non-read SQL was attempted: {query[:120]!r}")
    return failures


def is_read_query(query: str) -> bool:
    return bool(_READ_ONLY_START.match(_SQL_COMMENT.sub(" ", query)))


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.casefold().split())


def _contains(answer: str, value: str) -> bool:
    return normalize_text(value) in normalize_text(answer)


def extract_numbers(text: str) -> set[float]:
    """Parse numbers from free text, trying English (1,234.5) and Polish (1 234,5) formats."""
    numbers: set[float] = set()
    for token in _NUMBER_TOKEN.findall(text):
        compact = token.replace("\u00a0", "").replace("\u202f", "")
        for candidate in (compact.replace(",", ""), compact.replace(".", "").replace(",", ".")):
            try:
                numbers.add(float(candidate))
            except ValueError:
                continue
    return numbers


def _number_present(expected: float, found: set[float], tolerance: float) -> bool:
    return any(abs(expected - number) <= tolerance + 1e-9 for number in found)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()
