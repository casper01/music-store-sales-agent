"""The full agent with real tools (Chinook + Chroma) and a scripted stand-in for the LLM.

Only the model is faked. SQL runs against Chinook and retrieval runs against a real Chroma index
built from knowledge/ with a deterministic embedder, so these tests show that tool results flow
through the loop into the final answer - nothing in the answer is hardcoded.
"""

import pytest

from sales_agent.adapters.sqlite_reader import SqliteReader
from sales_agent.adapters.vector_index import ChromaVectorIndex
from sales_agent.agent.orchestrator import AgentOrchestrator
from sales_agent.evaluation.evaluator import file_sha256
from sales_agent.knowledge.indexer import build_knowledge_index
from sales_agent.tools.knowledge import SearchKnowledgeTool
from sales_agent.tools.sql import GetSchemaTool, RunSqlTool

from conftest import KNOWLEDGE_PATH
from fakes import HashingEmbedder, ScriptedModel, answer, calls, tool_call, tool_outputs

TOP_COUNTRY_SQL = (
    "SELECT i.BillingCountry AS country, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue "
    "FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
    "GROUP BY i.BillingCountry ORDER BY revenue DESC LIMIT 1"
)


@pytest.fixture
def tools(chinook_database, tmp_path):
    reader = SqliteReader(chinook_database, max_rows=200, timeout_seconds=5)
    index = ChromaVectorIndex(tmp_path / "store", HashingEmbedder())
    build_knowledge_index(KNOWLEDGE_PATH, index)
    return [GetSchemaTool(reader), RunSqlTool(reader), SearchKnowledgeTool(index, top_k=2)]


def compose_country_answer(messages):
    _schema, sql_result, knowledge = tool_outputs(messages)
    row = sql_result["rows"][0]
    source = knowledge["matches"][0]
    return answer(
        f"{row['country']} has the highest revenue ({row['revenue']}). "
        f"Per {source['document']}: {source['text']}"
    )


def test_multi_step_question_combines_sql_and_knowledge(tools):
    model = ScriptedModel(
        [
            calls(tool_call("get_schema", {}, "c1")),
            calls(
                tool_call("run_sql", {"query": TOP_COUNTRY_SQL}, "c2"),
                tool_call(
                    "search_knowledge", {"query": "interpreting a top revenue country"}, "c3"
                ),
            ),
            compose_country_answer,
        ]
    )

    result = AgentOrchestrator(model, tools, max_iterations=5).run(
        "Which country generated the most revenue and how should we interpret it?"
    )

    expected = tools[1].execute({"query": TOP_COUNTRY_SQL})["rows"][0]
    assert result.tool_names == ["get_schema", "run_sql", "search_knowledge"]
    assert all(call.ok for call in result.tool_calls)
    assert expected["country"] in result.answer
    assert str(expected["revenue"]) in result.answer
    assert "sales_playbook.md" in result.answer
    assert "invoice count" in result.answer


def test_failed_query_is_corrected_on_the_next_iteration(tools):
    model = ScriptedModel(
        [
            calls(tool_call("run_sql", {"query": "SELECT Country FROM Invoices"})),
            calls(tool_call("run_sql", {"query": "SELECT COUNT(*) AS n FROM Invoice"})),
            lambda messages: answer(f"There are {tool_outputs(messages)[-1]['rows'][0]['n']}."),
        ]
    )

    result = AgentOrchestrator(model, tools, max_iterations=4).run("How many invoices?")

    assert [call.ok for call in result.tool_calls] == [False, True]
    assert "no such table" in result.tool_calls[0].error
    assert result.answer.startswith("There are ")


def test_destructive_sql_from_a_compromised_model_is_blocked(tools, chinook_database):
    checksum = file_sha256(chinook_database)
    model = ScriptedModel(
        [
            calls(tool_call("run_sql", {"query": "DROP TABLE Invoice"})),
            answer("I cannot modify the database; it is read-only."),
        ]
    )

    result = AgentOrchestrator(model, tools, max_iterations=3).run("Delete the invoices table")

    assert result.tool_calls[0].ok is False
    assert "read-only" in result.tool_calls[0].error
    assert file_sha256(chinook_database) == checksum
