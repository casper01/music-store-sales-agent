from __future__ import annotations

from typing import Any

from sales_agent.adapters.sqlite_reader import MAX_QUERY_LENGTH, SqliteReader
from sales_agent.tools.validation import validate_arguments


class GetSchemaTool:
    name = "get_schema"
    description = (
        "List the database tables with their columns, types, primary keys and foreign keys. "
        "Date/time columns include value_range: the earliest and latest stored value (the "
        "data horizon). Call this before writing SQL against tables you have not inspected yet."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, reader: SqliteReader) -> None:
        self._reader = reader

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validate_arguments(self.name, arguments, self.parameters)
        return {"tables": [table.to_dict() for table in self._reader.get_schema()]}


class RunSqlTool:
    name = "run_sql"

    def __init__(self, reader: SqliteReader) -> None:
        self._reader = reader
        self.description = (
            "Run exactly one read-only SQLite query (SELECT or WITH ... SELECT) and return its "
            f"rows as JSON. At most {reader.max_rows} rows are returned; 'truncated': true means "
            "more rows exist, so aggregate in SQL. Writes, DDL, PRAGMA and ATTACH are rejected."
        )
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single read-only SQLite query.",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validate_arguments(self.name, arguments, self.parameters)
        return self._reader.run_sql(arguments["query"]).to_dict()
