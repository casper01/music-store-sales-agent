from pathlib import Path

import pytest

from sales_agent.config import Settings
from sales_agent.errors import ConfigurationError


def test_defaults_are_applied_when_variables_are_missing():
    settings = Settings.from_env({})

    assert settings.database_path == (Path.cwd() / "data/Chinook_Sqlite.sqlite").resolve()
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.llm_temperature is None
    assert settings.agent_max_iterations == 8
    assert settings.sql_max_rows == 200


def test_values_are_parsed_from_environment(tmp_path):
    settings = Settings.from_env(
        {
            "DATABASE_PATH": str(tmp_path / "db.sqlite"),
            "LLM_API_KEY": " secret ",
            "LLM_TEMPERATURE": "0",
            "AGENT_MAX_ITERATIONS": "3",
            "SQL_TIMEOUT_SECONDS": "1.5",
        }
    )
    assert settings.database_path == tmp_path / "db.sqlite"
    assert settings.require_api_key() == "secret"
    assert settings.llm_temperature == 0.0
    assert settings.agent_max_iterations == 3
    assert settings.sql_timeout_seconds == 1.5


def test_api_key_is_not_exposed_in_repr():
    assert "secret" not in repr(Settings.from_env({"LLM_API_KEY": "secret"}))


def test_missing_api_key_gives_actionable_error():
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        Settings.from_env({}).require_api_key()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AGENT_MAX_ITERATIONS", "0"),
        ("AGENT_MAX_ITERATIONS", "many"),
        ("SQL_MAX_ROWS", "-5"),
        ("SQL_TIMEOUT_SECONDS", "0"),
        ("KNOWLEDGE_TOP_K", "100"),
        ("LLM_TEMPERATURE", "5"),
    ],
)
def test_invalid_numbers_are_rejected(name, value):
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env({name: value})
