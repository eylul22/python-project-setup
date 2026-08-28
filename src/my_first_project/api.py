"""FastAPI application exposing text preprocessing utilities."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from my_first_project.preprocessing import (
    clean_documents,
    normalize_whitespace,
    truncate,
)

app = FastAPI(
    title="Text Preprocessing API",
    description="A small service that cleans and shortens text for AI pipelines.",
    version="0.2.0",
)


class TextRequest(BaseModel):
    """Input payload for cleaning a single piece of text."""

    text: str = Field(min_length=1, max_length=10_000)
    max_length: int = Field(default=100, ge=10, le=5_000)


class TextResponse(BaseModel):
    """Result of cleaning a single piece of text."""

    original_length: int
    cleaned_length: int
    result: str


class DocumentsRequest(BaseModel):
    """Input payload for cleaning a batch of documents."""

    documents: list[str] = Field(min_length=1, max_length=500)


class DocumentsResponse(BaseModel):
    """Result of cleaning a batch of documents."""

    received: int
    kept: int
    documents: list[str]


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Text Preprocessing API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report service health."""
    return {"status": "ok"}


@app.post("/clean", response_model=TextResponse)
def clean_text(payload: TextRequest) -> TextResponse:
    """Normalize whitespace in the text, then shorten it."""
    normalized = normalize_whitespace(payload.text)
    shortened = truncate(normalized, max_length=payload.max_length)
    return TextResponse(
        original_length=len(payload.text),
        cleaned_length=len(shortened),
        result=shortened,
    )


@app.post("/clean-batch", response_model=DocumentsResponse)
def clean_batch(payload: DocumentsRequest) -> DocumentsResponse:
    """Clean a batch of documents, dropping the empty ones."""
    cleaned = clean_documents(payload.documents)
    return DocumentsResponse(
        received=len(payload.documents),
        kept=len(cleaned),
        documents=cleaned,
    )