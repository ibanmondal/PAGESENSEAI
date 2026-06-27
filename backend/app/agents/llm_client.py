"""LLM client abstraction with a routing layer.

Routing is the cheap-but-visible engineering story: easy queries go to a fast
cheap model, hard reasoning goes to a strong model. The eval/latency numbers
that come out of this ("Flash routing cut p95 from 4.1s -> 1.3s") are resume fuel.

Each provider is a thin adapter implementing the same `LLMClient` protocol, so
swapping providers (or adding a local Ollama one) doesn't touch the agent code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.config import get_settings
from ..core.logging import get_logger

log = get_logger(__name__)


class LLMClient(Protocol):
    """Minimal chat-completion interface every provider implements."""
    name: str
    def complete(self, system: str, user: str, temperature: float = 0.2) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (works for OpenAI + DeepSeek + local vLLM/Ollama).
# ---------------------------------------------------------------------------

@dataclass
class OpenAICompatibleClient:
    name: str
    model: str
    api_key: str
    base_url: str | None = None

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


@dataclass
class GeminiClient:
    name: str = "gemini"
    model: str = "gemini-2.5-flash"

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        # Suppress protobuf version mismatch warnings from a conflicting
        # TensorFlow install on the system. Must happen BEFORE google.genai imports.
        import os as _os
        _os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        import warnings as _w
        _w.filterwarnings("ignore", message=".*Protobuf.*")
        from google import genai
        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=user,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
        return resp.text or ""


# ---------------------------------------------------------------------------
# Router — picks the model per query.
# ---------------------------------------------------------------------------

@dataclass
class ModelRouter:
    """Route queries to fast vs strong model based on simple intent signals.

    This is intentionally a rule-based router, not a learned classifier yet —
    it's the baseline. A fine-tuned intent classifier is a documented milestone
    (ROADMAP.md) that can be A/B-tested against THIS router with the same harness.
    """
    fast: LLMClient | None = None
    strong: LLMClient | None = None

    # Signals that indicate expensive reasoning is warranted.
    _STRONG_SIGNALS = (
        "compare", "comparison", "versus", " vs ", "trade-off", "tradeoff",
        "design", "architect", "why does", "critique", "analyze", "deep dive",
    )

    def classify(self, query: str, mode: str = "ask") -> str:
        """Return 'strong' or 'fast'. Exposed for tracing + eval."""
        if mode == "compare":
            return "strong"
        q = query.lower()
        if any(sig in q for sig in self._STRONG_SIGNALS):
            return "strong"
        return "fast"

    def route(self, query: str, mode: str = "ask") -> tuple[LLMClient, str]:
        tier = self.classify(query, mode)
        if tier == "strong" and self.strong is not None:
            return self.strong, "strong"
        if self.fast is not None:
            return self.fast, tier
        raise RuntimeError("No LLM client configured (need fast or strong)")


def build_router() -> ModelRouter:
    """Construct the router from settings. Gracefully handles missing keys."""
    settings = get_settings()
    fast: LLMClient | None = None
    strong: LLMClient | None = None
    if settings.gemini_api_key:
        fast = GeminiClient(model=settings.fast_model)
        log.info("Router fast tier: Gemini %s", settings.fast_model)
    if settings.openai_api_key:
        strong = OpenAICompatibleClient(
            name="openai", model=settings.strong_model, api_key=settings.openai_api_key,
        )
        log.info("Router strong tier: OpenAI %s", settings.strong_model)
    # Fallback: if only one key is set, use it for both tiers.
    if fast is None and strong is not None:
        fast = strong
    if strong is None and fast is not None:
        strong = fast
    return ModelRouter(fast=fast, strong=strong)
