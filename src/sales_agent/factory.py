"""Composition root: the only place that wires concrete adapters together."""

from __future__ import annotations

from sales_agent.adapters.chat_model import OpenAICompatibleChatModel
from sales_agent.adapters.embeddings import OpenAICompatibleEmbedder
from sales_agent.adapters.sqlite_reader import SqliteReader
from sales_agent.adapters.vector_index import ChromaVectorIndex
from sales_agent.agent.orchestrator import (
    AgentOrchestrator,
    ModelCallObserver,
    ToolCallObserver,
)
from sales_agent.config import Settings
from sales_agent.tools.knowledge import SearchKnowledgeTool
from sales_agent.tools.sql import GetSchemaTool, RunSqlTool


def build_reader(settings: Settings) -> SqliteReader:
    return SqliteReader(
        settings.database_path,
        max_rows=settings.sql_max_rows,
        timeout_seconds=settings.sql_timeout_seconds,
    )


def build_vector_index(settings: Settings) -> ChromaVectorIndex:
    embedder = OpenAICompatibleEmbedder(
        settings.require_api_key(),
        settings.embedding_model,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return ChromaVectorIndex(settings.vector_store_path, embedder)


def build_agent(
    settings: Settings,
    on_tool_call: ToolCallObserver | None = None,
    on_model_call: ModelCallObserver | None = None,
) -> AgentOrchestrator:
    reader = build_reader(settings)
    model = OpenAICompatibleChatModel(
        settings.require_api_key(),
        settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return AgentOrchestrator(
        model=model,
        tools=[
            GetSchemaTool(reader),
            RunSqlTool(reader),
            SearchKnowledgeTool(build_vector_index(settings), settings.knowledge_top_k),
        ],
        max_iterations=settings.agent_max_iterations,
        max_tool_output_chars=settings.max_tool_output_chars,
        on_tool_call=on_tool_call,
        on_model_call=on_model_call,
    )
