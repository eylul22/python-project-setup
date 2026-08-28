"""Text preprocessing utilities for AI pipelines."""


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(text.split())


def truncate(text: str, max_length: int = 100) -> str:
    """Shorten text to max_length characters, adding an ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def clean_documents(documents: list[str]) -> list[str]:
    """Normalize a batch of documents, dropping empty ones."""
    cleaned = [normalize_whitespace(document) for document in documents]
    return [document for document in cleaned if document]
