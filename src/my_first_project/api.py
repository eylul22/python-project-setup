"""FastAPI application exposing text preprocessing and iris prediction."""

from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from my_first_project.preprocessing import (
    clean_documents,
    normalize_whitespace,
    truncate,
)
from my_first_project.train import MODEL_PATH

ml_bundle: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and release it at shutdown."""
    if MODEL_PATH.exists():
        ml_bundle.update(joblib.load(MODEL_PATH))
    yield
    ml_bundle.clear()


app = FastAPI(
    title="Iris & Text API",
    description="Cleans text and predicts iris species from measurements.",
    version="0.3.0",
    lifespan=lifespan,
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


class IrisRequest(BaseModel):
    """The four measurements the classifier expects, in centimetres."""

    sepal_length: float = Field(ge=0, le=20, examples=[5.1])
    sepal_width: float = Field(ge=0, le=20, examples=[3.5])
    petal_length: float = Field(ge=0, le=20, examples=[1.4])
    petal_width: float = Field(ge=0, le=20, examples=[0.2])


class IrisResponse(BaseModel):
    """The predicted species with the model's confidence."""

    species: str
    confidence: float
    trained_accuracy: float


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Iris & Text API is running"}


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    """Report service health and whether the model is loaded."""
    return {"status": "ok", "model_loaded": bool(ml_bundle)}


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


@app.post("/predict", response_model=IrisResponse)
def predict_species(payload: IrisRequest) -> IrisResponse:
    """Predict the iris species from four flower measurements."""
    if not ml_bundle:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run the training script first.",
        )

    features = [
        [
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]
    ]

    probabilities = ml_bundle["model"].predict_proba(features)[0]
    best = int(probabilities.argmax())

    return IrisResponse(
        species=ml_bundle["target_names"][best],
        confidence=float(probabilities[best]),
        trained_accuracy=ml_bundle["accuracy"],
    )
