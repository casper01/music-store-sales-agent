"""Heading-aware Markdown chunking.

Documents are first split into sections at Markdown headings, so a chunk never mixes two rules
(e.g. "Revenue" and "Order"). Each chunk is prefixed with its heading path, which gives the
embedding the section context even for short paragraphs. Sections longer than ``max_chars`` are
packed paragraph by paragraph with a character overlap; single paragraphs longer than the limit
are split with a sliding window. ``max_chars`` bounds the body of a chunk; the heading prefix is
added on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True, slots=True)
class MarkdownChunk:
    heading: str
    body: str

    @property
    def text(self) -> str:
        """Text that is embedded and returned to the agent."""
        return f"{self.heading}\n\n{self.body}" if self.heading else self.body


def split_markdown(text: str, max_chars: int = 1200, overlap: int = 150) -> list[MarkdownChunk]:
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0")
    chunks: list[MarkdownChunk] = []
    for heading, body in _split_sections(text):
        chunks.extend(
            MarkdownChunk(heading=heading, body=piece)
            for piece in split_paragraphs(body, max_chars, overlap)
        )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return ``(heading path, body)`` pairs; headings inside code fences are ignored."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(title for _, title in stack), content))
        body.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
    flush()
    return sections


def split_paragraphs(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack blank-line separated paragraphs into chunks of at most ``max_chars`` characters."""
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_sliding_window(paragraph, max_chars, overlap))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        carry = current[-overlap:].strip() if overlap else ""
        with_carry = f"{carry}\n\n{paragraph}" if carry else paragraph
        current = with_carry if len(with_carry) <= max_chars else paragraph

    if current:
        chunks.append(current)
    return chunks


def _sliding_window(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max_chars - overlap
    pieces = []
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars].strip()
        if piece:
            pieces.append(piece)
        if start + max_chars >= len(text):
            break
    return pieces
