import pytest

from sales_agent.adapters.markdown_chunker import split_markdown, split_paragraphs

DOCUMENT = """# Rules

> Fictional.

## Revenue

Revenue is the sum of invoice lines.

## Order

One order is one invoice.

```text
# not a heading inside a code fence
```
"""


def test_splits_on_headings_and_keeps_heading_path():
    chunks = split_markdown(DOCUMENT)

    assert [chunk.heading for chunk in chunks] == ["Rules", "Rules > Revenue", "Rules > Order"]
    assert chunks[1].body == "Revenue is the sum of invoice lines."
    assert chunks[1].text.startswith("Rules > Revenue\n\n")
    assert "# not a heading" in chunks[2].body


def test_heading_stack_pops_to_the_correct_level():
    chunks = split_markdown("# A\n\n## B\n\nb\n\n### C\n\nc\n\n## D\n\nd")
    assert [chunk.heading for chunk in chunks] == ["A > B", "A > B > C", "A > D"]


def test_document_without_headings_is_one_section():
    chunks = split_markdown("plain text")
    assert len(chunks) == 1
    assert chunks[0].heading == ""
    assert chunks[0].text == "plain text"


def test_long_paragraph_is_split_into_bounded_overlapping_pieces():
    text = "".join(chr(ord("a") + index % 26) for index in range(40))
    pieces = split_paragraphs(text, max_chars=16, overlap=4)

    assert len(pieces) > 1
    assert all(len(piece) <= 16 for piece in pieces)
    assert pieces[0][-4:] == pieces[1][:4]
    assert "".join(piece[4:] if index else piece for index, piece in enumerate(pieces)) == text


def test_paragraphs_are_packed_without_exceeding_the_limit():
    text = "\n\n".join(["alpha " * 5, "beta " * 5, "gamma " * 5])
    pieces = split_paragraphs(text, max_chars=40, overlap=10)
    assert all(len(piece) <= 40 for piece in pieces)
    assert "gamma" in pieces[-1]


@pytest.mark.parametrize(("max_chars", "overlap"), [(10, 10), (0, 0), (10, -1)])
def test_rejects_invalid_parameters(max_chars, overlap):
    with pytest.raises(ValueError):
        split_markdown("text", max_chars=max_chars, overlap=overlap)
