"""Tools against the real Chinook database (a private copy per test)."""

import pytest

from sales_agent.adapters.sqlite_reader import SqliteReader
from sales_agent.errors import SqlExecutionError
from sales_agent.evaluation.evaluator import file_sha256
from sales_agent.tools.sql import GetSchemaTool, RunSqlTool

REVENUE_BY_COUNTRY = """
    SELECT i.BillingCountry AS country, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue
    FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
    GROUP BY i.BillingCountry ORDER BY revenue DESC
"""


@pytest.fixture
def reader(chinook_database):
    return SqliteReader(chinook_database, max_rows=200, timeout_seconds=5)


def test_schema_exposes_the_chinook_tables(reader):
    tables = {table["table"] for table in GetSchemaTool(reader).execute({})["tables"]}
    assert {"Artist", "Album", "Track", "Invoice", "InvoiceLine", "Customer"} <= tables


def test_invoice_line_revenue_matches_invoice_totals(reader):
    sql = RunSqlTool(reader)
    by_line = sql.execute(
        {"query": "SELECT ROUND(SUM(UnitPrice * Quantity), 2) AS v FROM InvoiceLine"}
    )
    by_invoice = sql.execute({"query": "SELECT ROUND(SUM(Total), 2) AS v FROM Invoice"})
    assert by_line["rows"] == by_invoice["rows"]


def test_country_revenue_ranking_is_ordered_and_consistent(reader):
    rows = RunSqlTool(reader).execute({"query": REVENUE_BY_COUNTRY})["rows"]
    revenues = [row["revenue"] for row in rows]
    total = RunSqlTool(reader).execute(
        {"query": "SELECT ROUND(SUM(UnitPrice * Quantity), 2) AS v FROM InvoiceLine"}
    )["rows"][0]["v"]

    assert revenues == sorted(revenues, reverse=True)
    assert sum(revenues) == pytest.approx(total, abs=0.05)


def test_unknown_country_returns_no_rows_instead_of_a_value(reader):
    result = RunSqlTool(reader).execute(
        {"query": "SELECT SUM(Total) AS revenue FROM Invoice WHERE BillingCountry = 'Atlantis'"}
    )
    assert result["rows"] == [{"revenue": None}]


def test_row_limit_applies_to_large_tables(chinook_database):
    reader = SqliteReader(chinook_database, max_rows=50, timeout_seconds=5)
    result = RunSqlTool(reader).execute({"query": "SELECT TrackId FROM Track"})
    assert result["row_count"] == 50
    assert result["truncated"] is True


def test_delete_invoices_is_rejected_and_database_is_unchanged(chinook_database, reader):
    checksum = file_sha256(chinook_database)

    for statement in ("DROP TABLE Invoice", "DELETE FROM Invoice", "DELETE FROM InvoiceLine"):
        with pytest.raises(SqlExecutionError):
            RunSqlTool(reader).execute({"query": statement})

    assert file_sha256(chinook_database) == checksum
    count = RunSqlTool(reader).execute({"query": "SELECT COUNT(*) AS n FROM Invoice"})
    assert count["rows"][0]["n"] > 0
