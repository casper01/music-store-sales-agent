"""Read-only access to a SQLite database.

Read-only access is enforced in three independent layers, so a bypass of one is caught by the next:

1. the file is opened with the ``mode=ro`` URI parameter - SQLite refuses every write;
2. ``PRAGMA query_only = ON`` - the connection rejects any statement that would modify data;
3. an authorizer callback whitelists only the actions a plain ``SELECT`` needs, which also blocks
   ``ATTACH``, ``PRAGMA`` and other side-effecting statements before they are prepared.

Resource limits: a wall-clock timeout implemented with a progress handler (SQLite's ``timeout``
argument only covers lock waits, not long-running queries), a maximum number of returned rows,
and a maximum query length. Only one statement per call is accepted.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from sales_agent.errors import SqlExecutionError, ToolInputError

MAX_QUERY_LENGTH = 20_000
_PROGRESS_HANDLER_STEPS = 1_000
_MAX_CELL_CHARS = 2_000
_TEMPORAL_TYPE_MARKERS = ("DATE", "TIME")


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool
    # Smallest and largest stored value, reported for date/time columns so the model knows the
    # data horizon (e.g. that invoices end in 2025) without guessing from the current date.
    value_range: tuple[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class ForeignKey:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True, slots=True)
class TableSchema:
    table: str
    columns: tuple[ColumnInfo, ...]
    foreign_keys: tuple[ForeignKey, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    row_limit: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SqliteReader:
    """Executes validated, read-only queries against a SQLite file."""

    _ALLOWED_ACTIONS = frozenset(
        {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }
    )

    def __init__(self, database_path: Path, max_rows: int, timeout_seconds: float) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.database_path = Path(database_path)
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite3.Connection's own context manager only manages transactions and does not close
        # the connection, so closing is handled explicitly here.
        if not self.database_path.is_file():
            raise SqlExecutionError(f"Database file not found: {self.database_path}")
        try:
            connection = sqlite3.connect(
                f"{self.database_path.resolve().as_uri()}?mode=ro",
                uri=True,
                timeout=self.timeout_seconds,
            )
        except sqlite3.Error as exc:
            raise SqlExecutionError(f"Could not open database: {exc}") from exc
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            yield connection
        finally:
            connection.close()

    @classmethod
    def _authorize(cls, action: int, *_: Any) -> int:
        return sqlite3.SQLITE_OK if action in cls._ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def get_schema(self) -> list[TableSchema]:
        """Return all user tables with their columns and foreign keys."""
        try:
            with self._connect() as connection:
                table_names = [
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_schema "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                ]
                return [self._describe_table(connection, name) for name in table_names]
        except sqlite3.Error as exc:
            raise SqlExecutionError(f"Could not read database schema: {exc}") from exc

    @staticmethod
    def _describe_table(connection: sqlite3.Connection, table_name: str) -> TableSchema:
        quoted = _quote_identifier(table_name)
        columns = tuple(
            ColumnInfo(
                name=row["name"],
                type=row["type"],
                nullable=not row["notnull"],
                primary_key=bool(row["pk"]),
                value_range=_value_range(connection, quoted, row["name"], row["type"]),
            )
            for row in connection.execute(f"PRAGMA table_info({quoted})")
        )
        foreign_keys = tuple(
            ForeignKey(
                column=row["from"], references_table=row["table"], references_column=row["to"]
            )
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted})")
        )
        return TableSchema(table=table_name, columns=columns, foreign_keys=foreign_keys)

    def run_sql(self, query: str) -> QueryResult:
        """Execute one read-only statement and return at most ``max_rows`` rows."""
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        if len(query) > MAX_QUERY_LENGTH:
            raise ToolInputError(f"query exceeds the {MAX_QUERY_LENGTH} character limit")

        deadline = monotonic() + self.timeout_seconds
        try:
            with self._connect() as connection:
                connection.set_authorizer(self._authorize)
                connection.set_progress_handler(
                    lambda: int(monotonic() > deadline), _PROGRESS_HANDLER_STEPS
                )
                cursor = connection.execute(query)
                if cursor.description is None:
                    raise SqlExecutionError("Only read-only queries that return rows are allowed")
                columns = [description[0] for description in cursor.description]
                fetched = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            raise self._translate_error(exc) from exc

        truncated = len(fetched) > self.max_rows
        rows = [
            {key: _to_json_value(value) for key, value in dict(row).items()}
            for row in fetched[: self.max_rows]
        ]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            row_limit=self.max_rows,
        )

    def _translate_error(self, exc: sqlite3.Error) -> SqlExecutionError:
        """Map SQLite errors to messages the model (and a person) can act on."""
        message = str(exc)
        lowered = message.lower()
        if "interrupted" in lowered:
            return SqlExecutionError(
                f"Query timed out after {self.timeout_seconds:g} s; simplify or aggregate it"
            )
        if "not authorized" in lowered or "readonly" in lowered or "query_only" in lowered:
            return SqlExecutionError(
                "Query rejected: the database is read-only; only SELECT statements are permitted"
            )
        if "one statement at a time" in lowered:
            return SqlExecutionError("Query rejected: send exactly one SQL statement per call")
        return SqlExecutionError(f"SQL error: {message}")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _value_range(
    connection: sqlite3.Connection, quoted_table: str, column: str, declared_type: str
) -> tuple[Any, Any] | None:
    """Return ``(MIN, MAX)`` of a date/time column, or ``None`` for other or empty columns."""
    if not any(marker in (declared_type or "").upper() for marker in _TEMPORAL_TYPE_MARKERS):
        return None
    quoted_column = _quote_identifier(column)
    low, high = connection.execute(
        f"SELECT MIN({quoted_column}), MAX({quoted_column}) FROM {quoted_table}"
    ).fetchone()
    return None if low is None else (low, high)


def _to_json_value(value: Any) -> Any:
    """Make a SQLite value JSON-friendly and bounded in size."""
    if isinstance(value, bytes):
        return f"<binary {len(value)} bytes>"
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        return value[:_MAX_CELL_CHARS] + "...<truncated>"
    return value
