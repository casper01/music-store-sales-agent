import pytest

from sales_agent.adapters.vector_index import ChromaVectorIndex, IndexEntry
from sales_agent.errors import ConfigurationError, EmbeddingError, KnowledgeIndexError
from sales_agent.knowledge.indexer import build_knowledge_index, load_knowledge_entries

from conftest import KNOWLEDGE_PATH
from fakes import HashingEmbedder


@pytest.fixture
def index(tmp_path):
    return ChromaVectorIndex(tmp_path / "store", HashingEmbedder())


def test_loads_all_documents_with_unique_ids():
    entries = load_knowledge_entries(KNOWLEDGE_PATH)

    documents = {entry.document for entry in entries}
    assert documents == {"business_rules.md", "refund_policy.md", "sales_playbook.md"}
    assert len({entry.id for entry in entries}) == len(entries)
    assert any(entry.heading.endswith("> Revenue") for entry in entries)


def test_every_document_is_marked_as_fictional():
    for path in KNOWLEDGE_PATH.glob("*.md"):
        assert "fictional" in path.read_text(encoding="utf-8").lower(), path.name


def test_missing_or_empty_directory_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError):
        load_knowledge_entries(tmp_path / "missing")
    with pytest.raises(ConfigurationError):
        load_knowledge_entries(tmp_path)


@pytest.mark.parametrize(
    ("query", "expected_document", "expected_heading"),
    [
        ("revenue means the sum of unitprice and quantity", "business_rules.md", "Revenue"),
        ("refund request window days after invoice", "refund_policy.md", "Request window"),
        ("interpreting a top revenue country", "sales_playbook.md", "top-revenue country"),
    ],
)
def test_search_returns_the_relevant_document_first(
    index, query, expected_document, expected_heading
):
    build_knowledge_index(KNOWLEDGE_PATH, index)

    hits = index.search(query, top_k=3)

    assert hits[0].document == expected_document
    assert expected_heading in hits[0].heading
    assert hits[0].similarity >= hits[-1].similarity


def test_rebuild_replaces_previous_content(index):
    index.rebuild([IndexEntry("a#0", "old text", "a.md", "A", 0)])
    count = index.rebuild([IndexEntry("b#0", "new text", "b.md", "B", 0)])

    assert count == 1
    assert [hit.document for hit in index.search("text", top_k=5)] == ["b.md"]


def test_top_k_is_capped_by_collection_size(index):
    index.rebuild([IndexEntry("a#0", "one", "a.md", "A", 0)])
    assert len(index.search("one", top_k=10)) == 1


def test_search_before_indexing_asks_to_run_index(index):
    with pytest.raises(KnowledgeIndexError, match="sales-agent index"):
        index.search("revenue", top_k=3)


def test_search_with_a_different_embedding_model_requires_reindex(tmp_path):
    ChromaVectorIndex(tmp_path / "store", HashingEmbedder(model_name="model-a")).rebuild(
        [IndexEntry("a#0", "text", "a.md", "A", 0)]
    )
    other = ChromaVectorIndex(tmp_path / "store", HashingEmbedder(model_name="model-b"))

    with pytest.raises(KnowledgeIndexError, match="model-a"):
        other.search("text", top_k=1)


def test_failed_embedding_keeps_existing_index(tmp_path):
    class FailingEmbedder(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("API down")

    path = tmp_path / "store"
    ChromaVectorIndex(path, HashingEmbedder()).rebuild([IndexEntry("a#0", "kept", "a.md", "A", 0)])

    with pytest.raises(EmbeddingError):
        ChromaVectorIndex(path, FailingEmbedder()).rebuild(
            [IndexEntry("b#0", "new", "b.md", "B", 0)]
        )
    assert ChromaVectorIndex(path, HashingEmbedder()).search("kept", top_k=1)[0].document == "a.md"


def test_rejects_invalid_search_arguments(index):
    with pytest.raises(ValueError):
        index.search("  ", top_k=1)
    with pytest.raises(ValueError):
        index.search("revenue", top_k=0)


def test_rejects_duplicate_ids(index):
    entry = IndexEntry("a#0", "text", "a.md", "A", 0)
    with pytest.raises(ValueError, match="unique"):
        index.rebuild([entry, entry])
