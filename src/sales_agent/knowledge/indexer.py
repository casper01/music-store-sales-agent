"""Turns the ``knowledge/`` directory into vector-index entries."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sales_agent.adapters.markdown_chunker import split_markdown
from sales_agent.adapters.vector_index import IndexEntry
from sales_agent.errors import ConfigurationError


class RebuildableIndex(Protocol):
    def rebuild(self, entries: list[IndexEntry]) -> int: ...


def load_knowledge_entries(
    directory: Path, max_chars: int = 1200, overlap: int = 150
) -> list[IndexEntry]:
    """Chunk every Markdown file below ``directory`` into index entries."""
    if not directory.is_dir():
        raise ConfigurationError(f"Knowledge directory not found: {directory}")
    paths = sorted(directory.rglob("*.md"))
    if not paths:
        raise ConfigurationError(f"No Markdown files found in {directory}")

    entries: list[IndexEntry] = []
    for path in paths:
        document = path.relative_to(directory).as_posix()
        text = path.read_text(encoding="utf-8")
        for number, chunk in enumerate(split_markdown(text, max_chars, overlap)):
            entries.append(
                IndexEntry(
                    id=f"{document}#{number}",
                    text=chunk.text,
                    document=document,
                    heading=chunk.heading,
                    chunk=number,
                )
            )
    return entries


def build_knowledge_index(directory: Path, index: RebuildableIndex) -> int:
    return index.rebuild(load_knowledge_entries(directory))
