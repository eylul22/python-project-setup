"""Tests for the text preprocessing utilities."""

from my_first_project.preprocessing import (
    clean_documents,
    normalize_whitespace,
    truncate,
)


def test_normalize_whitespace_collapses_spaces() -> None:
    assert normalize_whitespace("hello    world") == "hello world"


def test_normalize_whitespace_strips_edges() -> None:
    assert normalize_whitespace("  hello world  ") == "hello world"


def test_truncate_leaves_short_text_unchanged() -> None:
    assert truncate("short text", max_length=100) == "short text"


def test_truncate_shortens_long_text() -> None:
    result = truncate("a" * 50, max_length=10)
    assert len(result) == 10
    assert result.endswith("...")


def test_clean_documents_removes_empty_entries() -> None:
    documents = ["  hello  ", "", "   ", "world"]
    assert clean_documents(documents) == ["hello", "world"]