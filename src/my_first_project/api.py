"""FastAPI application exposing text preprocessing utilities."""

from fastapi import FastAPI

from my_first_project.preprocessing import normalize_whitespace, truncate

app = FastAPI(
    title="Text Preprocessing API",
    description="A small service that cleans and shortens text for AI pipelines.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Text Preprocessing API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report service health."""
    return {"status": "ok"}


@app.get("/normalize")
def normalize(text: str) -> dict[str, str]:
    """Collapse repeated whitespace in the given text."""
    return {"original": text, "normalized": normalize_whitespace(text)}


@app.get("/truncate")
def shorten(text: str, max_length: int = 50) -> dict[str, str | int]:
    """Shorten the given text to max_length characters."""
    return {"result": truncate(text, max_length=max_length), "max_length": max_length}