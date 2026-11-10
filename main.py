from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_IS_VERCEL = bool(os.environ.get("VERCEL"))
_RUNTIME_DIR = Path(tempfile.gettempdir()) / "nl_mf_faq"
if _IS_VERCEL:
    os.environ.setdefault("HF_HOME", str(_RUNTIME_DIR / "huggingface"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(_RUNTIME_DIR / "huggingface" / "transformers")
    )
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        str(_RUNTIME_DIR / "huggingface" / "sentence-transformers"),
    )
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from src.config import Settings, settings as default_settings
from src.pipeline.query_pipeline import QueryPipeline


LOGGER = logging.getLogger(__name__)
_BUNDLED_CHROMA_DIR = Path(__file__).resolve().parent / "data" / "chroma"
_RUNTIME_CHROMA_DIR = _RUNTIME_DIR / "chroma"
_RUNTIME_TRACE_DIR = _RUNTIME_DIR / "traces"
_pipeline: Optional[QueryPipeline] = None
_init_lock = threading.RLock()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    category: str
    reason: str
    source_url: Optional[str]
    source_date: str


def _runtime_settings() -> Settings:
    if not _IS_VERCEL:
        return default_settings

    chroma_dir = _RUNTIME_CHROMA_DIR
    if not chroma_dir.exists():
        with _init_lock:
            if not chroma_dir.exists():
                if not _BUNDLED_CHROMA_DIR.exists():
                    raise RuntimeError(
                        f"Bundled ChromaDB data is missing: {_BUNDLED_CHROMA_DIR}"
                    )
                _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copytree(_BUNDLED_CHROMA_DIR, chroma_dir)
    return default_settings.model_copy(
        update={
            "chroma_dir": chroma_dir,
            "trace_dir": _RUNTIME_TRACE_DIR,
        }
    )


def _get_pipeline() -> QueryPipeline:
    global _pipeline
    if _pipeline is None:
        with _init_lock:
            if _pipeline is None:
                _pipeline = QueryPipeline(settings=_runtime_settings())
    return _pipeline


app = FastAPI(title="Groww Mutual Fund FAQ Assistant")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = _get_pipeline().answer(request.question)
    except Exception:
        LOGGER.exception("Chat request failed")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable.",
        )
    return ChatResponse(
        answer=result.answer,
        category=result.category,
        reason=result.reason,
        source_url=result.source_url,
        source_date=result.source_date,
    )
