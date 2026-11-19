import sqlite3
from pathlib import Path

import pytest

from sales_agent.adapters.sqlite_reader import MAX_QUERY_LENGTH, SqliteReader
from sales_agent.errors import SqlExecutionError, ToolInputError
from sales_agent.evaluation.evaluator import file_sha256


def make_reader(path: Path, max_rows: int = 10, timeout: float = 2.0) -> SqliteReader:
    return SqliteReader(path, max_rows=max_rows, timeout_seconds=timeout)


class TestSchema:
    def test_lists_tables_columns_and_foreign_keys(self, sample_database):
        schema = {table.table: table for table in make_reader(sample_database).get_schema()}

        assert set(schema) == {"Album", "Artist"}
        artist_columns = {column.name: column for column in schema["Artist"].columns}
        assert artist_columns["ArtistId"].primary_key is True
        assert artist_columns["Name"].nullable is False
        assert artist_columns["Name"].type == "TEXT"
        foreign_key = schema["Album"].foreign_keys[0]
        assert (foreign_key.column, foreign_key.references_table) == ("ArtistId", "Artist")

    def test_missing_database_file_gives_clear_error(self, tmp_path):
        with pytest.raises(SqlExecutionError, match="not found"):
            make_reader(tmp_path / "missing.sqlite").get_schema()


class TestQueries:
    def test_returns_rows_and_columns(self, sample_database):
        result = make_reader(sample_database).run_sql("SELECT Name FROM Artist WHERE ArtistId = 1")

        assert result.columns == ["Name"]
        assert result.rows == [{"Name": "Example Artist"}]
        assert result.row_count == 1
        assert result.truncated is False

    def test_supports_joins_and_common_table_expressions(self, sample_database):
        result = make_reader(sample_database).run_sql(
            "WITH a AS (SELECT * FROM Artist) "
            "SELECT a.Name, al.Title FROM a JOIN Album al ON al.ArtistId = a.ArtistId"
        )
        assert result.rows == [{"Name": "Example Artist", "Title": "First Album"}]

    def test_no_results_is_an_empty_successful_result(self, sample_database):
        result = make_reader(sample_database).run_sql(
            "SELECT Name FROM Artist WHERE Name = 'Does Not Exist'"
        )
        assert result.rows == []
        assert result.columns == ["Name"]
        assert result.truncated is False

    def test_invalid_sql_reports_the_database_error(self, sample_database):
        with pytest.raises(SqlExecutionError, match="no such table"):
            make_reader(sample_database).run_sql("SELECT * FROM Missing")

    def test_blob_values_are_summarised(self, sample_database):
        result = make_reader(sample_database).run_sql("SELECT x'0102' AS payload")
        assert result.rows == [{"payload": "<binary 2 bytes>"}]


class TestLimits:
    def test_caps_rows_and_marks_truncation(self, sample_database):
        result = make_reader(sample_database, max_rows=2).run_sql(
            "SELECT ArtistId FROM Artist ORDER BY ArtistId"
        )
        assert [row["ArtistId"] for row in result.rows] == [1, 2]
        assert result.truncated is True
        assert result.row_limit == 2

    def test_exact_row_limit_is_not_marked_truncated(self, sample_database):
        result = make_reader(sample_database, max_rows=3).run_sql("SELECT ArtistId FROM Artist")
        assert result.row_count == 3
        assert result.truncated is False

    def test_long_running_query_times_out(self, sample_database):
        reader = make_reader(sample_database, timeout=0.2)
        endless = (
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
        )
        with pytest.raises(SqlExecutionError, match="timed out"):
            reader.run_sql(endless)

    @pytest.mark.parametrize("query", ["", "   ", None, 42])
    def test_rejects_empty_or_non_string_query(self, sample_database, query):
        with pytest.raises(ToolInputError):
            make_reader(sample_database).run_sql(query)

    def test_rejects_overlong_query(self, sample_database):
        with pytest.raises(ToolInputError, match="character limit"):
            make_reader(sample_database).run_sql("SELECT 1 " + " " * MAX_QUERY_LENGTH)

    @pytest.mark.parametrize(("max_rows", "timeout"), [(0, 1.0), (1, 0)])
    def test_rejects_invalid_limits(self, sample_database, max_rows, timeout):
        with pytest.raises(ValueError):
            SqliteReader(sample_database, max_rows=max_rows, timeout_seconds=timeout)


class TestReadOnly:
    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE Artist SET Name = 'Changed'",
            "DELETE FROM Artist",
            "INSERT INTO Artist (ArtistId, Name) VALUES (99, 'New')",
            "DROP TABLE Artist",
            "CREATE TABLE Hacked (id INTEGER)",
            "ALTER TABLE Artist ADD COLUMN Extra TEXT",
            "PRAGMA query_only = OFF",
            "ATTACH DATABASE ':memory:' AS other",
            "VACUUM",
            "REPLACE INTO Artist (ArtistId, Name) VALUES (1, 'Replaced')",
            "DELETE FROM Artist RETURNING *",
        ],
    )
    def test_rejects_writes_and_leaves_file_unchanged(self, sample_database, statement):
        reader = make_reader(sample_database)
        checksum = file_sha256(sample_database)

        with pytest.raises(SqlExecutionError):
            reader.run_sql(statement)

        assert file_sha256(sample_database) == checksum
        assert reader.run_sql("SELECT COUNT(*) AS n FROM Artist").rows == [{"n": 3}]

    def test_rejects_multiple_statements(self, sample_database):
        with pytest.raises(SqlExecutionError):
            make_reader(sample_database).run_sql("SELECT 1; DROP TABLE Artist")
        assert make_reader(sample_database).run_sql("SELECT COUNT(*) AS n FROM Artist").rows


def test_schema_reports_value_range_of_date_columns(tmp_path):
    path = tmp_path / "dates.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE Invoice (InvoiceId INTEGER PRIMARY KEY, InvoiceDate DATETIME, Note TEXT);
            CREATE TABLE Empty (Id INTEGER PRIMARY KEY, CreatedAt DATE);
            INSERT INTO Invoice VALUES
                (1, '2021-01-01 00:00:00', 'a'), (2, '2025-12-22 00:00:00', 'b');
            """
        )
    connection.close()

    schema = {table.table: table for table in make_reader(path).get_schema()}
    invoice = {column.name: column for column in schema["Invoice"].columns}

    assert invoice["InvoiceDate"].value_range == ("2021-01-01 00:00:00", "2025-12-22 00:00:00")
    assert invoice["Note"].value_range is None
    assert invoice["InvoiceId"].value_range is None
    assert schema["Empty"].columns[1].value_range is None
