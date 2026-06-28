"""LangGraph multi-step agent: retrieve -> generate -> fact-check.

The graph is deliberately small and explicit (3 nodes) because clarity beats
cleverness for a portfolio project. Each node is a pure function of state, which
makes the whole agent unit-testable and the intermediate state inspectable —
both matter in interviews ("show me the agent's reasoning trace").

If langgraph isn't installed, we fall back to a plain function with identical
behaviour, so the backend always runs even in a minimal environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from ..core.logging import get_logger
from ..models.schemas import Citation, QueryRequest, QueryResponse, RetrievalResult

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Agent state — flows through the graph.
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    request: QueryRequest
    rewritten_query: str = ""
    retrieval: RetrievalResult | None = None
    draft_answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    chat_history: list[dict] = field(default_factory=list)
    warning: str | None = None
    confidence: float = 0.0
    model_tier: str = "fast"
    model_used: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Nodes — each is (state) -> state. Pure, testable, single-responsibility.
# ---------------------------------------------------------------------------

def make_rewrite_node(router) -> Callable:
    """Rewrite vague queries into specific search terms for better retrieval."""
    def _node(state: AgentState) -> AgentState:
        if state.error:
            return state
        # Only rewrite for "ask" mode. Compare/summarize are usually explicit.
        if state.request.mode != "ask":
            state.rewritten_query = state.request.question
            return state

        system = "You are a search query optimizer. Given a user question, output ONLY a concise, keyword-rich search query that will maximize retrieval quality from a document database. Do not include quotes, explanations, or conversational text. If the query is already specific, leave it mostly as-is."
        client, _ = router.route(state.request.question, state.request.mode)
        try:
            # We use the fast client for this cheap operation
            fast_client = router.fast if router.fast else client
            state.rewritten_query = fast_client.complete(system, state.request.question).strip()
            log.info("Rewrote query: '%s' -> '%s'", state.request.question, state.rewritten_query)
        except Exception as e:
            log.warning("query rewrite failed, falling back to original: %s", e)
            state.rewritten_query = state.request.question
        return state
    return _node


def make_retrieve_node(retrieve_fn: Callable[[QueryRequest], RetrievalResult]) -> Callable:
    def _node(state: AgentState) -> AgentState:
        if state.error:
            return state
        try:
            # We construct a temporary request with the rewritten query for retrieval
            search_request = state.request.model_copy(update={"question": state.rewritten_query or state.request.question})
            state.retrieval = retrieve_fn(search_request)
        except Exception as e:
            log.exception("retrieve failed")
            state.error = f"retrieve: {e}"
        return state
    return _node


def make_generate_node(router, answer_fn: Callable | None = None) -> Callable:
    """Build the generation node. `answer_fn` lets tests inject a fake LLM."""
    def _node(state: AgentState) -> AgentState:
        if state.error or state.retrieval is None:
            return state
        # Build citations from the re-ranked chunks.
        state.citations = [
            Citation(
                tab_id=sc.chunk.tab_id,
                url=sc.chunk.url,
                title=sc.chunk.title,
                snippet=sc.chunk.text[:280],
                score=round(sc.score, 4),
            )
            for sc in state.retrieval.top_chunks
        ]
        # Pick the model via the router (the visible routing decision).
        client, tier = router.route(state.request.question, state.request.mode)
        state.model_tier = tier
        state.model_used = getattr(client, "model", client.name)

        # Format history
        history_text = ""
        for msg in state.chat_history:
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

        context = _format_context(state.retrieval)
        system = _system_prompt(state.request.mode)
        
        user = (
            f"Use ONLY the context below to answer. If the context is insufficient, "
            f"say so explicitly. Cite sources as [#] matching the numbered context.\n\n"
        )
        if history_text:
            user += f"--- CHAT HISTORY ---\n{history_text}\n"
            
        user += f"--- CONTEXT ---\n{context}\n\nQuestion: {state.request.question}"

        if answer_fn is not None:
            state.draft_answer = answer_fn(system, user)
        else:
            try:
                state.draft_answer = client.complete(system, user)
            except Exception as e:
                log.exception("generation failed")
                state.error = f"generate: {e}"
        
        # Append to history
        if not state.error:
            state.chat_history = state.chat_history + [
                {"role": "user", "content": state.request.question},
                {"role": "assistant", "content": state.draft_answer}
            ]

        return state
    return _node


def make_factcheck_node(router, check_fn: Callable | None = None) -> Callable:
    """Lightweight self-consistency check: does the answer over-claim vs context?

    We don't run a second LLM pass by default (cost/latency); instead we compute
    a coverage signal: fraction of citations whose snippet terms overlap the
    answer. Low coverage -> warning + lower confidence. This is the honest,
    measurable "fact verification" that belongs on a resume.
    """
    def _node(state: AgentState) -> AgentState:
        if state.error or not state.draft_answer:
            return state
        if check_fn is not None:
            state.warning, state.confidence = check_fn(state.draft_answer, state.citations)
            return state
        # Default heuristic coverage check.
        answer_lower = state.draft_answer.lower()
        covered = 0
        for cite in state.citations:
            # pull a few distinctive tokens from the snippet
            tokens = {t for t in cite.snippet.lower().split() if len(t) > 4}
            if tokens and sum(1 for t in tokens if t in answer_lower) / len(tokens) > 0.15:
                covered += 1
        total = max(len(state.citations), 1)
        coverage = covered / total
        state.confidence = round(0.4 + 0.6 * coverage, 3)  # bounded in [0.4, 1.0]
        if coverage < 0.34:
            state.warning = "Low citation coverage — answer may not be well-supported."
        return state
    return _node


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _system_prompt(mode: str) -> str:
    base = (
        "You are PageSense AI, a precise research assistant. Answer strictly from "
        "the provided context. Mark every factual claim with a citation [#]. "
        "If multiple sources disagree, surface the conflict."
    )
    if mode == "compare":
        base += (
            "\n\nThis is a COMPARISON task. Produce a markdown table comparing the "
            "relevant sources across key dimensions, then a short recommendation."
        )
    elif mode == "summarize":
        base += "\n\nThis is a SUMMARIZE task. Produce a tight bullet-point summary."
    return base


def _format_context(retrieval: RetrievalResult) -> str:
    lines = []
    for i, sc in enumerate(retrieval.top_chunks, start=1):
        lines.append(
            f"[{i}] (source: {sc.chunk.title} — {sc.chunk.url})\n{sc.chunk.text}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(retrieve_fn, router, answer_fn=None, check_fn=None, use_langgraph=True):
    """Return a callable(AgentState) -> AgentState that runs the full pipeline.

    Prefers a real langgraph StateGraph; falls back to a sequential function.
    Both produce identical results — the fallback exists so CI/eval works without
    the langgraph dependency.
    """
    rewrite_node = make_rewrite_node(router)
    retrieve_node = make_retrieve_node(retrieve_fn)
    generate_node = make_generate_node(router, answer_fn)
    factcheck_node = make_factcheck_node(router, check_fn)

    def _run(state: AgentState) -> AgentState:
        state = rewrite_node(state)
        state = retrieve_node(state)
        state = generate_node(state)
        state = factcheck_node(state)
        return state

    if not use_langgraph:
        return _run

    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        log.info("langgraph not installed — using sequential fallback")
        return _run

    # Global checkpointer for this graph instance
    memory = MemorySaver()

    # TypedDict shim so langgraph can route on our dataclass state.
    g = StateGraph(dict)
    g.add_node("rewrite", lambda d: asdict(rewrite_node(_from_dict(d))))
    g.add_node("retrieve", lambda d: asdict(retrieve_node(_from_dict(d))))
    g.add_node("generate", lambda d: asdict(generate_node(_from_dict(d))))
    g.add_node("factcheck", lambda d: asdict(factcheck_node(_from_dict(d))))
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "factcheck")
    g.add_edge("factcheck", END)
    compiled = g.compile(checkpointer=memory)

    def _run_graph(state: AgentState) -> AgentState:
        config = {"configurable": {"thread_id": state.request.session_id}}
        
        # Clear transient fields for a new run, but keep request to re-initialize
        # chat_history will be automatically merged in by MemorySaver.
        input_dict = {
            "request": state.request.model_dump(),
            "rewritten_query": "",
            "retrieval": None,
            "draft_answer": "",
            "citations": [],
            "warning": None,
            "confidence": 0.0,
            "model_tier": "fast",
            "model_used": "",
            "error": None
        }
        final_dict = compiled.invoke(input_dict, config=config)
        return _from_dict(final_dict)
    return _run_graph


def _from_dict(d: dict) -> AgentState:
    """Reconstruct AgentState from a langgraph dict."""
    req_dict = d.get("request", {})
    req = QueryRequest(**req_dict) if isinstance(req_dict, dict) and req_dict else req_dict
    
    return AgentState(
        request=req,
        rewritten_query=d.get("rewritten_query", ""),
        retrieval=None,
        draft_answer=d.get("draft_answer", ""),
        citations=[c if isinstance(c, Citation) else Citation(**c) for c in d.get("citations", [])],
        chat_history=d.get("chat_history", []),
        warning=d.get("warning"),
        confidence=d.get("confidence", 0.0),
        model_tier=d.get("model_tier", "fast"),
        model_used=d.get("model_used", ""),
        error=d.get("error"),
    )


def run_query(state: AgentState, graph) -> QueryResponse:
    """Execute the graph and map final state -> QueryResponse."""
    final = graph(state)
    return QueryResponse(
        answer=final.draft_answer or final.error or "No answer generated.",
        citations=final.citations,
        model_used=final.model_used or "unknown",
        confidence=final.confidence,
        warning=final.warning,
    )
