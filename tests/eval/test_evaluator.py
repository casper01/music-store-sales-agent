"""The grading logic of the evaluator, exercised with stand-in agents."""

import pytest

from sales_agent.adapters.sqlite_reader import SqliteReader
from sales_agent.agent.contracts import AgentResult, StopReason, ToolCallRecord
from sales_agent.errors import ModelError
from sales_agent.evaluation.cases import Check, EvalCase
from sales_agent.evaluation.evaluator import Evaluator, extract_numbers, is_read_query
from sales_agent.evaluation.report import render_eval_report

TOP_COUNTRY_SQL = (
    "SELECT i.BillingCountry FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
    "GROUP BY 1 ORDER BY SUM(il.UnitPrice * il.Quantity) DESC LIMIT 1"
)
TOP_REVENUE_SQL = (
    "SELECT SUM(il.UnitPrice * il.Quantity) FROM Invoice i "
    "JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId GROUP BY i.BillingCountry "
    "ORDER BY 1 DESC LIMIT 1"
)

CASE = EvalCase(
    id=1,
    question="Top country?",
    criteria="Top country with revenue",
    required_tools=frozenset({"run_sql"}),
    forbidden_tools=frozenset({"search_knowledge"}),
    checks=(
        Check("sql_values_present", sql=TOP_COUNTRY_SQL),
        Check("sql_numbers_present", sql=TOP_REVENUE_SQL),
    ),
)


class FixedAgent:
    def __init__(self, answer, tool_calls=()):
        self.result = AgentResult(answer, StopReason.ANSWERED, 2, tuple(tool_calls))

    def run(self, question):
        return self.result


@pytest.fixture
def reader(chinook_database):
    return SqliteReader(chinook_database, max_rows=50, timeout_seconds=5)


@pytest.fixture
def truth(reader):
    country = reader.run_sql(TOP_COUNTRY_SQL).rows[0]["BillingCountry"]
    revenue = next(iter(reader.run_sql(TOP_REVENUE_SQL).rows[0].values()))
    return country, revenue


SQL_CALL = ToolCallRecord("run_sql", {"query": "SELECT ..."}, ok=True)


def test_grounded_answer_with_correct_tools_passes(reader, truth):
    country, revenue = truth
    agent = FixedAgent(f"The top country is {country.upper()} with {revenue:,.2f}.", [SQL_CALL])

    report = Evaluator(agent, reader).run([CASE])

    assert report.results[0].failures == []
    assert report.all_passed
    assert report.database_unchanged


def test_polish_number_formatting_is_accepted(reader, truth):
    country, revenue = truth
    polish = f"{revenue:.2f}".replace(".", ",")
    result = Evaluator(FixedAgent(f"{country}: {polish} USD", [SQL_CALL]), reader).evaluate_case(
        CASE
    )
    assert result.passed


def test_wrong_value_and_missing_tool_are_reported(reader, truth):
    country, _ = truth
    result = Evaluator(FixedAgent(f"{country} earned 1.00"), reader).evaluate_case(CASE)

    assert not result.passed
    assert any("Required tool" in failure for failure in result.failures)
    assert any("expected number" in failure for failure in result.failures)


def test_failed_tool_call_does_not_satisfy_requirement(reader, truth):
    country, revenue = truth
    failed = ToolCallRecord("run_sql", {"query": "SELECT x"}, ok=False, error="no such column")
    result = Evaluator(FixedAgent(f"{country} {revenue:.2f}", [failed]), reader).evaluate_case(CASE)
    assert any("Required tool" in failure for failure in result.failures)


def test_forbidden_tool_and_write_attempt_fail_the_case(reader, truth):
    country, revenue = truth
    calls = [
        SQL_CALL,
        ToolCallRecord("search_knowledge", {"query": "x"}, ok=True),
        ToolCallRecord("run_sql", {"query": "/* hi */ DROP TABLE Invoice"}, ok=False, error="no"),
    ]
    result = Evaluator(FixedAgent(f"{country} {revenue:.2f}", calls), reader).evaluate_case(CASE)

    assert any("Forbidden" in failure for failure in result.failures)
    assert any("Non-read SQL" in failure for failure in result.failures)


def test_agent_errors_become_case_failures(reader):
    class FailingAgent:
        def run(self, question):
            raise ModelError("rate limited")

    result = Evaluator(FailingAgent(), reader).evaluate_case(CASE)
    assert result.failures == ["Agent error: rate limited"]


def test_keyword_checks_are_accent_and_case_insensitive(reader):
    case = EvalCase(
        2,
        "q",
        "c",
        frozenset(),
        frozenset(),
        (
            Check("contains_any", values=("Helena Holý",)),
            Check("contains_all", values=("refund", "30")),
            Check("not_contains", values=("guaranteed",)),
        ),
    )
    agent = FixedAgent("HELENA HOLY: Refunds within 30 days.")
    assert Evaluator(agent, reader).evaluate_case(case).passed


def test_report_lists_every_case(reader, truth):
    country, revenue = truth
    report = Evaluator(FixedAgent(f"{country} {revenue:.2f}", [SQL_CALL]), reader).run([CASE])
    markdown = render_eval_report("test-model", report)
    assert "| 1 | PASS | run_sql | - |" in markdown
    assert "Passed cases: **1/1**" in markdown


@pytest.mark.parametrize(
    ("text", "expected"),
    [("523.06", 523.06), ("2,328.60", 2328.6), ("523,06", 523.06), ("1\u00a0234,5", 1234.5)],
)
def test_extract_numbers_handles_common_formats(text, expected):
    assert any(abs(number - expected) < 1e-9 for number in extract_numbers(f"total: {text}."))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT 1", True),
        ("  with x as (select 1) select * from x", True),
        ("-- comment\nSELECT 1", True),
        ("DELETE FROM Invoice", False),
        ("/* SELECT */ DROP TABLE Invoice", False),
    ],
)
def test_is_read_query(query, expected):
    assert is_read_query(query) is expected
