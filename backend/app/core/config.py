"""Centralized configuration via environment variables with sensible defaults.

Keep all tunable knobs here so the rest of the codebase never reads os.environ
directly — this makes the system reproducible, which matters for the eval harness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

# Load variables from a local .env file (if present) into the environment.
# This MUST run before any _env() call below reads os.environ.
try:
    from dotenv import load_dotenv
    # .env lives at the project root (one level above backend/).
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    pass  # python-dotenv missing or no .env — env vars still work normally


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- App ---
    app_name: str = "PageSense AI"
    env: str = field(default_factory=lambda: _env("APP_ENV", "dev"))
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- Models (routing) ---
    # Cheap/fast model handles easy queries + routing decisions.
    fast_model: str = field(default_factory=lambda: _env("FAST_MODEL", "gemini-2.5-flash"))
    # Strong model handles reasoning-heavy / comparison queries.
    strong_model: str = field(default_factory=lambda: _env("STRONG_MODEL", "gpt-4.1"))
    # Provider API keys — read lazily, never logged.
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))

    # --- Retrieval ---
    # Hybrid fusion weights (must sum to 1.0). Tuned empirically via the eval harness.
    bm25_weight: float = field(default_factory=lambda: _env_float("BM25_WEIGHT", 0.35))
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 0.65))

    # Cross-encoder re-ranker HF model id. Start from MS MARCO MiniLM for speed.
    reranker_model: str = field(default_factory=lambda: _env(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ))
    # Path to a locally fine-tuned re-ranker (overrides reranker_model when set).
    reranker_finetuned_path: str = field(default_factory=lambda: _env("RERANKER_FINETUNED_PATH", ""))

    top_k_retrieval: int = field(default_factory=lambda: _env_int("TOP_K_RETRIEVAL", 20))
    top_k_rerank: int = field(default_factory=lambda: _env_int("TOP_K_RERANK", 5))

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 400))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 60))

    # --- Embeddings ---
    embedding_model: str = field(default_factory=lambda: _env(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ))

    @property
    def reranker_source(self) -> str:
        """Which re-ranker weights to actually load."""
        return self.reranker_finetuned_path or self.reranker_model


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — call this everywhere instead of constructing."""
    return Settings()
