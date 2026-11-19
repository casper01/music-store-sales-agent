import pytest

from sales_agent.adapters.sqlite_reader import SqliteReader
from sales_agent.adapters.vector_index import SearchHit
from sales_agent.errors import SqlExecutionError, ToolInputError
from sales_agent.tools.knowledge import SearchKnowledgeTool
from sales_agent.tools.sql import GetSchemaTool, RunSqlTool
from sales_agent.tools.validation import validate_arguments


@pytest.fixture
def reader(sample_database):
    return SqliteReader(sample_database, max_rows=10, timeout_seconds=2)


class StubSearcher:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, top_k):
        self.queries.append((query, top_k))
        return self.hits


class TestValidation:
    SCHEMA = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 5},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def test_accepts_valid_arguments(self):
        assert validate_arguments("t", {"query": "abc", "limit": 3}, self.SCHEMA)

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ([], "JSON object"),
            ({}, "missing required"),
            ({"query": "a", "other": 1}, "unexpected"),
            ({"query": 5}, "type string"),
            ({"query": "a", "limit": True}, "type integer"),
            ({"query": "   "}, "must not be empty"),
            ({"query": "toolong"}, "exceeds"),
        ],
    )
    def test_rejects_invalid_arguments(self, arguments, message):
        with pytest.raises(ToolInputError, match=message):
            validate_arguments("t", arguments, self.SCHEMA)


class TestSqlTools:
    def test_get_schema_returns_serialisable_tables(self, reader):
        tables = GetSchemaTool(reader).execute({})["tables"]
        assert [table["table"] for table in tables] == ["Album", "Artist"]
        assert tables[0]["foreign_keys"][0]["references_table"] == "Artist"

    def test_get_schema_rejects_arguments(self, reader):
        with pytest.raises(ToolInputError):
            GetSchemaTool(reader).execute({"table": "Artist"})

    def test_run_sql_returns_rows(self, reader):
        result = RunSqlTool(reader).execute({"query": "SELECT COUNT(*) AS total FROM Artist"})
        assert result["rows"] == [{"total": 3}]
        assert result["truncated"] is False

    def test_run_sql_description_mentions_row_limit(self, reader):
        assert "10 rows" in RunSqlTool(reader).description

    @pytest.mark.parametrize("arguments", [{}, {"sql": "SELECT 1"}, {"query": 1}, {"query": ""}])
    def test_run_sql_validates_arguments(self, reader, arguments):
        with pytest.raises(ToolInputError):
            RunSqlTool(reader).execute(arguments)

    def test_run_sql_rejects_destructive_query(self, reader):
        with pytest.raises(SqlExecutionError, match="read-only"):
            RunSqlTool(reader).execute({"query": "DROP TABLE Artist"})


class TestSearchKnowledgeTool:
    def test_returns_matches_with_document_names(self):
        hit = SearchHit("business_rules.md", "Rules > Revenue", 0, "Revenue means ...", 0.9)
        searcher = StubSearcher([hit])

        result = SearchKnowledgeTool(searcher, top_k=3).execute({"query": "  revenue  "})

        assert result["matches"][0]["document"] == "business_rules.md"
        assert searcher.queries == [("revenue", 3)]

    def test_reports_when_nothing_matches(self):
        result = SearchKnowledgeTool(StubSearcher([]), top_k=3).execute({"query": "x"})
        assert result["matches"] == []
        assert "No matching" in result["note"]

    @pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "a", "top_k": 2}])
    def test_validates_arguments(self, arguments):
        with pytest.raises(ToolInputError):
            SearchKnowledgeTool(StubSearcher([]), top_k=3).execute(arguments)
