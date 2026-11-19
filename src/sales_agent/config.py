"""Application settings loaded from environment variables.

The agent talks to any LLM provider that exposes an OpenAI-compatible API. The default is
Google Gemini (free tier available at https://aistudio.google.com/api-keys); switching to OpenAI
or another provider only means changing LLM_BASE_URL, LLM_MODEL and EMBEDDING_MODEL.

Secrets (the API key) are read only from the environment; ``.env`` is loaded by the CLI via
python-dotenv for local convenience and is git-ignored. Relative paths are resolved against the
current working directory, so run the CLI from the project root (as documented in README.md).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from sales_agent.errors import ConfigurationError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_LLM_MODEL = "gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

_T = TypeVar("_T", int, float)


def _read_path(env: Mapping[str, str], name: str, default: str) -> Path:
    return Path(env.get(name) or default).expanduser().resolve()


def _read_text(env: Mapping[str, str], name: str, default: str = "") -> str:
    return (env.get(name) or "").strip() or default


def _read_number(
    env: Mapping[str, str],
    name: str,
    default: _T,
    parse: Callable[[str], _T],
    minimum: _T,
    maximum: _T,
) -> _T:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = parse(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _read_optional_float(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return _read_number(env, name, 0.0, float, 0.0, 2.0)


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    knowledge_path: Path
    vector_store_path: Path
    llm_api_key: str = field(repr=False)
    llm_base_url: str
    llm_model: str
    embedding_model: str
    llm_temperature: float | None
    llm_timeout_seconds: float
    llm_max_retries: int
    agent_max_iterations: int
    sql_max_rows: int
    sql_timeout_seconds: float
    knowledge_top_k: int
    max_tool_output_chars: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            database_path=_read_path(env, "DATABASE_PATH", "data/Chinook_Sqlite.sqlite"),
            knowledge_path=_read_path(env, "KNOWLEDGE_PATH", "knowledge"),
            vector_store_path=_read_path(env, "VECTOR_STORE_PATH", ".vector_store"),
            llm_api_key=_read_text(env, "LLM_API_KEY"),
            llm_base_url=_read_text(env, "LLM_BASE_URL", GEMINI_BASE_URL),
            llm_model=_read_text(env, "LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_model=_read_text(env, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            llm_temperature=_read_optional_float(env, "LLM_TEMPERATURE"),
            llm_timeout_seconds=_read_number(env, "LLM_TIMEOUT_SECONDS", 90.0, float, 1.0, 600.0),
            # Free tiers allow only a few requests per minute; the SDK retries HTTP 429 with
            # exponential backoff and honours the server's Retry-After header.
            llm_max_retries=_read_number(env, "LLM_MAX_RETRIES", 6, int, 0, 20),
            agent_max_iterations=_read_number(env, "AGENT_MAX_ITERATIONS", 8, int, 1, 30),
            sql_max_rows=_read_number(env, "SQL_MAX_ROWS", 200, int, 1, 5000),
            sql_timeout_seconds=_read_number(env, "SQL_TIMEOUT_SECONDS", 5.0, float, 0.1, 120.0),
            knowledge_top_k=_read_number(env, "KNOWLEDGE_TOP_K", 4, int, 1, 20),
            max_tool_output_chars=_read_number(
                env, "MAX_TOOL_OUTPUT_CHARS", 20_000, int, 1_000, 200_000
            ),
        )

    def require_api_key(self) -> str:
        """Return the API key or fail with an actionable message."""
        if not self.llm_api_key:
            raise ConfigurationError(
                "LLM_API_KEY is not set. Copy .env.gemini.example or .env.openai.example to .env "
                "and paste your API key (or export it in the environment)."
            )
        return self.llm_api_key
