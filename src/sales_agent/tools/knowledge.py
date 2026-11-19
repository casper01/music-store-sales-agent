from __future__ import annotations

from typing import Any, Protocol

from sales_agent.adapters.vector_index import SearchHit
from sales_agent.tools.validation import validate_arguments

MAX_QUERY_CHARS = 1_000


class KnowledgeSearcher(Protocol):
    def search(self, query: str, top_k: int) -> list[SearchHit]: ...


class SearchKnowledgeTool:
    name = "search_knowledge"
    description = (
        "Semantic search over the FICTIONAL business-rule documents (business_rules.md, "
        "refund_policy.md, sales_playbook.md). Use it for definitions (revenue, order, active "
        "customer), the refund policy, and guidance on interpreting sales results. Returns the "
        "best-matching fragments with their document name, section heading and similarity."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up, phrased as a question or topic.",
                "minLength": 1,
                "maxLength": MAX_QUERY_CHARS,
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, searcher: KnowledgeSearcher, top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._searcher = searcher
        self._top_k = top_k

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validate_arguments(self.name, arguments, self.parameters)
        hits = self._searcher.search(arguments["query"].strip(), self._top_k)
        if not hits:
            return {"matches": [], "note": "No matching knowledge fragments were found."}
        return {"matches": [hit.to_dict() for hit in hits]}
